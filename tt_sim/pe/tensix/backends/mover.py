from enum import IntEnum

from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.perf.model import mover_cost_model, unit_cost_model
from tt_sim.util.conversion import conv_to_bytes


class MoverUnit(TensixBackendUnit):
    """
    This unit moves data between L1 and other memory spaces, such as the
    16KB NCRISC private IRAM. This can be used to accelerate memcpy.

    Based on description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/XMOV.md
    """

    class XMOV_DIRECTION(IntEnum):
        XMOV_L0_TO_L1 = 0
        XMOV_L1_TO_L0 = 1
        XMOV_L0_TO_L0 = 2
        XMOV_L1_TO_L1 = 3

    OPCODE_TO_HANDLER = {
        "XMOV": "handle_xmov",
    }

    TENSIX_CFG_BASE = 0xFFEF0000
    MEM_NCRISC_IRAM_BASE = 0xFFC00000

    #: Which ``mover.transfer`` rate (tt_sim/perf/unit_costs.yaml) prices each
    #: XMOV mode. The two "L1_TO_" modes are the memcpy the table's
    #: ``l1_to_l1`` entry covers ("Covers XMOV_L1_TO_L1 and XMOV_L1_TO_L0");
    #: the "L0_TO_" modes are memsets, split by whether the destination is L1.
    MODE_TO_TRANSFER_KIND = {
        XMOV_DIRECTION.XMOV_L1_TO_L1: "l1_to_l1",
        XMOV_DIRECTION.XMOV_L1_TO_L0: "l1_to_l1",
        XMOV_DIRECTION.XMOV_L0_TO_L1: "l1_memset",
        XMOV_DIRECTION.XMOV_L0_TO_L0: "non_l1_memset",
    }

    def __init__(self, backend):
        self.tdma_commands = []
        super().__init__(backend, MoverUnit.OPCODE_TO_HANDLER, "Mover")
        # Phase 5 of docs/plans/event-driven-pump.md, wired 2026-08-06. Two
        # models, because the cost splits in two exactly as the ISA doc states
        # it: the XMOV entry's 1 is the *issue* cost ("XMOV will execute in a
        # single cycle - the mover proceeds with the task in the background"),
        # and the background task's duration is bandwidth-derived from the
        # ``mover.transfer`` rates. The duration is charged as unit occupancy
        # — "the thread issuing an XMOV instruction will be automatically
        # stalled until the mover is able to start work" is what the
        # issue-refusal machinery already models — so a second XMOV, or a
        # queued TDMA command, waits out the transfer in flight. Both are
        # ``None`` unless TT_SIM_COST_MODEL is set, and every transfer then
        # completes in the tick it was issued exactly as before.
        arch = "blackhole" if backend.blackhole else "wormhole"
        self.cost_model = unit_cost_model("XMOV", arch)
        self.transfer_model = mover_cost_model(arch)

    def append_command_from_tdma(self, command):
        self.tdma_commands.append(command)

    def is_clock_idle(self):
        return super().is_clock_idle() and not self.tdma_commands

    def transfer_cycles(self, mode, count):
        """Cycles the mover is busy with a ``count``-byte transfer, or ``None``."""
        model = self.transfer_model
        if model is None:
            return None
        kind = MoverUnit.MODE_TO_TRANSFER_KIND.get(mode)
        if kind is None:
            return None
        return model.transfer_cycles(kind, count)

    def instruction_occupancy(self, instruction_name, issue_thread):
        """XMOV's charge: the greater of the issue cost and the transfer time.

        The transfer this XMOV starts is described entirely by config registers
        (``handle_xmov`` reads the same four fields immediately after), so the
        size and mode are knowable here, at issue, before the handler runs.
        A transfer the table cannot price falls back to the entry's documented
        1-cycle issue cost — the floor.

        The two config fields are read *exactly* as :meth:`handle_xmov` reads
        them, quirks included, so the charge and the transfer can never
        disagree about which mode or how many bytes: whatever this bills,
        that is what runs.
        """
        base = super().instruction_occupancy(instruction_name, issue_thread)
        if self.transfer_model is None or instruction_name != "XMOV":
            return base
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )
        count = (
            self.getConfigValue(stateID, "THCON_SEC0_REG6_Buffer_size") & 0xFFFF
        ) << 4
        mode = self.getConfigValue(stateID, "THCON_SEC0_REG6_Destination_address")
        cycles = self.transfer_cycles(mode, count)
        if cycles is None:
            return base
        return max(base or 1, cycles)

    def clock_tick(self, cycle_num):
        if self.busy_until is not None:
            # Release any transfer whose deadline has arrived before deciding
            # whether the mover can start the next command; the base drain does
            # the same thing first for the instruction path.
            self._release_expired(cycle_num)
        if len(self.tdma_commands) > 0:
            if self.is_occupied():
                # Mid-transfer: the mover "is able to start work" only once
                # the current transfer's occupancy expires, so a queued TDMA
                # command waits exactly as a refused XMOV issue does. Never
                # armed with the cost model off.
                return
            command = self.tdma_commands.pop(0)
            self.move(*command)
            # The TDMA path bypasses the instruction machinery, so its charge
            # is armed here: the transfer starts this tick.
            cycles = self.transfer_cycles(command[3], command[2])
            if cycles is not None and cycles > 1:
                self.occupy_for(cycle_num, cycles)
        else:
            super().clock_tick(cycle_num)

    def checkForOutstandingInstructions(self):
        if self.busy_until is not None:
            # A transfer in flight. This is what STALLWAIT's mover condition
            # (Wormhole C12, Blackhole C9) and the TDMA "mover wait" command
            # poll, and on hardware both wait for the background transfer to
            # finish — so the occupancy window counts as outstanding work.
            # Never true with the cost model off, since nothing arms it.
            return True
        if len(self.tdma_commands) > 0:
            return True
        if len(self.next_instruction) > 0:
            return True
        return False

    def handle_xmov(self, instruction_info, issue_thread, instr_args):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        dst = self.getConfigValue(stateID, "THCON_SEC0_REG6_Destination_address") << 4
        src = self.getConfigValue(stateID, "THCON_SEC0_REG6_Source_address") << 4
        count = (
            self.getConfigValue(stateID, "THCON_SEC0_REG6_Buffer_size") & 0xFFFF
        ) << 4
        mode = self.getConfigValue(stateID, "THCON_SEC0_REG6_Destination_address")

        self.move(dst, src, count, mode)

    def move(self, dst, src, count, mode):
        """
        This is based on the functional model description at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/Mover.md
        """
        assert self.backend.getAddressableMemory() is not None
        if (
            mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1
            or mode == MoverUnit.XMOV_DIRECTION.XMOV_L0_TO_L1
        ):
            # In the "_TO_L1" modes, dst must be an address in L1.
            assert dst < 1024 * 1464
        else:
            if dst <= 0xFFFF:
                dst += MoverUnit.TENSIX_CFG_BASE
            elif 0x40000 <= dst and dst <= 0x4FFFF:
                dst = (dst - 0x40000) + MoverUnit.MEM_NCRISC_IRAM_BASE
            else:
                dst = None  # Operation still happens, but the writes get discarded.

            if (dst & 0xFFFF) + count > 0x10000:
                raise NotImplementedError(
                    "Can not access more than one region at a time"
                )

        # Perform the operation.
        if (
            mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1
            or mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L0
        ):
            # In the "L1_TO_" modes, a memcpy is done, and src must be an address in L1.
            if src >= (1024 * 1464):
                raise NotImplementedError("")
            self.backend.getAddressableMemory().write(
                dst, self.backend.getAddressableMemory().read(src, count)
            )
        else:
            # In the "L0_TO_" modes, a memset is done.
            zero_val = conv_to_bytes(0, count)
            self.backend.getAddressableMemory().write(dst, zero_val)
