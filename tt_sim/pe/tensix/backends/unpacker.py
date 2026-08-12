from math import ceil, floor

import numpy as np

from tt_sim.network.tt_noc import NoCOverlay
from tt_sim.pe.tensix.backends.backend_base import (
    DATA_FORMAT_TO_BITS,
    DATA_FORMAT_TO_NAME,
    DataFormat,
    TensixBackendUnit,
)
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.util import DataFormatConversions
from tt_sim.perf.model import unit_cost_model
from tt_sim.util.bits import get_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class UnPackerUnit(TensixBackendUnit):
    """
    Unpacker unit, which unpacks from L1 into either srcA/srcB or dst.

    Based on the description and functional code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/UNPACR.md
    """

    OPCODE_TO_HANDLER = {"UNPACR": "handle_unpacr", "UNPACR_NOP": "handle_unpacr_nop"}

    # Datum widths a block of L1 can be reinterpreted as directly. Datums are
    # little-endian and so is the host, so this is the same value per datum as
    # ``conv_to_uint32`` of the datum's bytes -- without assembling each one.
    _BLOCK_DTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32}

    def __init__(self, backend, unpacker_id):
        self.unpacker_id = unpacker_id
        self.context_counter = [0] * 3
        self.srcBank = 0
        self.srcRow = [0] * 3
        self.blocked = False
        self.blocked_wait_bank = None
        self.repeat_instruction = None
        self.pending_unpack = None
        self.setRegBase = 0
        self.setRegAcc = 0
        # Occupancy the current tick's handler wants charged, armed by
        # ``clock_tick`` once the handler has run. The unpacker cannot use the
        # base per-opcode charge because its cost is not a constant: a
        # datum-moving UNPACR costs a >= 2-cycle address phase plus a data
        # phase of transfer-bytes / throttle-rate, both knowable only after
        # the instruction's config has been decoded. See ``handle_regular``.
        self._pending_occupancy = 0
        # The thread whose ``UNPACR`` owes the documented address-phase issue
        # interlock, armed by ``clock_tick`` from the same anchor the occupancy
        # uses. ``None`` unless one is owed, which with no cost model is never.
        self._pending_thread_block = None
        # Who owns ``_pending_occupancy`` / the hold it becomes: the thread that
        # issued the ``UNPACR`` still in the unpacker's pipeline. Read only by
        # :meth:`hasInflightInstructionsFromThread`, and only meaningful while
        # ``is_occupied()`` -- once the hold expires the last writer is stale
        # and the predicate stops consulting it.
        self._pending_occupancy_thread = None
        self._occupied_thread = None
        # The Src bank hand-over an ``UNPACR`` with ``FlipSrc`` owes the matrix
        # unit, and the cycle it becomes visible. See ``flip_src_banks`` and
        # ``_hand_over_src_bank``: the transfer is pipelined, so the hand-over
        # lands at the *end* of it rather than in the tick the instruction
        # retired. ``(src, out_data_format, issue_thread, src_row_base)`` while
        # owed, ``None`` otherwise; the deadline is ``None`` until the arming
        # site knows which cycle the transfer runs to.
        self._deferred_dvalid = None
        self._deferred_dvalid_cycle = None
        super().__init__(backend, UnPackerUnit.OPCODE_TO_HANDLER, "Unpacker")
        # Phase 5 of docs/plans/event-driven-pump.md, wired 2026-08-06 as the
        # seventh unit (the last-but-one; TDMA stays out deliberately).
        # ``None`` unless TT_SIM_COST_MODEL is set, in which case every unpack
        # completes in the tick it was issued exactly as before.
        self.cost_model = unit_cost_model(
            "UNPACK", "blackhole" if backend.blackhole else "wormhole"
        )

    def issueInstruction(self, instruction, from_thread):
        if self.blocked:
            # Not ``unit_busy``: the unpacker is idle, and is waiting for the
            # *matrix unit* to release the Src bank its latched instruction
            # wants to write. The exact mirror of the wait gate's
            # ``src_reserved_by_unpacker``, and the other half of the Src
            # ping-pong a code generator is trying to overlap -- so it is
            # attributed to MATH, not to the unpacker the gate offered to.
            return self._refuse("src_reserved_by_matrix", blocked_on="MATH")
        else:
            return super().issueInstruction(instruction, from_thread)

    def is_clock_idle(self):
        # A blocked unpacker re-runs its latched instruction every cycle until
        # the Src bank it is waiting on frees up, and an unpacker still owing a
        # Src hand-over has a state change of its own left to make -- neither is
        # idle even with an empty issue queue.
        return (
            not self.next_instruction
            and not self.blocked
            and self._deferred_dvalid is None
        )

    def hasInflightInstructionsFromThread(self, from_thread):
        """A blocked *or still transferring* unpacker has not retired anything.

        Two states here are "an instruction is in this unpacker's pipeline"
        that the base implementation's issue-queue scan cannot see.

        **Still transferring.** ``STALLWAIT.md`` defines conditions C1 and C2 as
        "The current thread has an instruction in any stage of Unpacker 0's [/
        Unpacker 1's] pipeline", and ``UNPACR_Regular.md``'s Performance section
        says where those stages are: an ``UNPACR`` "spends at least two cycles
        calculating the initial input address ... Once these cycles are
        complete, execution proceeds in a pipelined fashion, with the primary
        bottleneck being the fetching of bytes from L1". So the L1 fetch is a
        stage of the pipeline, and the condition is *not* met while it runs --
        the address phase and the data phase are both "in the pipeline", and
        together they are exactly the occupancy this unit charges itself in
        ``_arm_pending_occupancy``. Reporting only the issue queue let C1/C2
        clear while the transfer was still moving datums, which is the same
        window ``_hand_over_src_bank`` exists to protect: a ``STALLWAIT`` is
        how a thread with no semaphore waits for its own unpack to land.

        Scoped to this unit deliberately. Occupancy is throughput
        back-pressure, and for a pipelined unit that is not the same as
        residency -- an instruction can be in a stage of the pipeline long
        after the unit will accept the next one. The unpacker is the case
        where the two coincide, because the doc's bottleneck *is* the transfer
        and tt-sim charges the whole address+data phase as one hold, so
        "occupied by this thread" and "this thread's instruction is in a stage"
        are the same statement. Nothing here licenses reading another unit's
        ``busy_until`` as residency; that would need each unit's own latency.

        **Blocked.** All three blocking sites here wait for a Src bank's
        ``AllowedClient`` to come back to the unpackers, and re-run the latched
        instruction every cycle until it does. The base implementation only
        looks at ``next_instruction``, which the issue queue has already drained
        by the time the handler blocks, so a blocked unpacker used to report the
        thread as *done* -- and a unit that can never make progress became
        invisible to both consumers of this predicate: the PC-buffer drain
        (``TTSync``/``CoprocessorDoneCheck``, i.e. what a kernel's end-of-thread
        sync reads) and the deadlock watchdog.

        That is not a cosmetic gap. ``perfbench/tensixbench
        --dvalid-unpacr-nop`` issues ``UNPACR_NOP``+``set_dvalid`` and then
        deliberately never clears dvalid, so the Src bank is never handed back;
        on Blackhole silicon the *next* execution of that setup -- the next
        program launch in the same process, or the first launch of the next
        process on the same card -- waits at the unpacker for ever and only a
        board reset clears it. tt-sim reached the identical blocked state and
        ran to completion anyway. See ROADMAP.md, "Unpacker dvalid deadlock".
        """
        if self._occupied_thread == from_thread and self.is_occupied():
            return True
        if (
            self.blocked
            and self.repeat_instruction is not None
            and self.repeat_instruction[1] == from_thread
        ):
            return True
        return super().hasInflightInstructionsFromThread(from_thread)

    def blocked_on(self):
        """``(thread, opcode, which_src, bank)`` while blocked, else ``None``.

        For the deadlock watchdog: "backend busy" is a symptom, and the thing a
        reader needs is which Src bank the unpacker is waiting to be given back.
        """
        if not self.blocked or self.repeat_instruction is None:
            return None
        instruction_info, issue_thread = self.repeat_instruction
        which = "SrcB" if self.unpacker_id == 1 else "SrcA"
        # The bank waited on is not always the unpacker's own: the Blackhole
        # ZEROSRC form waits on the MATRIX unit's bank unless `stall_clr_cntrl`
        # says otherwise, and confusing the two is what makes this deadlock hard
        # to read, so report the one actually latched at the blocking site.
        return (issue_thread, instruction_info["name"], which, self.blocked_wait_bank)

    def instruction_occupancy(self, instruction_name, issue_thread):
        """``None`` for UNPACR: its cost is charged from the handler instead.

        The base hook runs before the handler and sees only the opcode name,
        which for the unpacker is not enough twice over: ``UNPACR`` is three
        instruction forms sharing one opcode (regular / increment-context /
        flush-cache), and only the datum-moving regular form has published
        timing (``UNPACR_Regular.md``'s Performance section); and the regular
        form's cost depends on the transfer size and the throttle config,
        which exist only after ``read_unpack_state`` has decoded them. So the
        charge is computed there and armed by :meth:`clock_tick` through
        ``_pending_occupancy``. ``UNPACR_NOP`` keeps the flat table lookup
        (a documented 1, which ``occupy_for`` no-ops on).
        """
        if instruction_name == "UNPACR":
            return None
        return super().instruction_occupancy(instruction_name, issue_thread)

    def clock_tick(self, cycle_num):
        # A hand-over owed by an earlier UNPACR comes first: it belongs to a
        # transfer that finished before anything this tick does, and the unit
        # cannot have accepted new work while it was outstanding (the same
        # occupancy that sets the deadline refuses the issue).
        if self._deferred_dvalid is not None:
            self._hand_over_src_bank(cycle_num)
        if self.blocked:
            if self.busy_until is not None:
                # The base drain releases expired holds at the top of its own
                # tick; the blocked path never reaches it, so an address-phase
                # hold armed just before the block would otherwise outlive its
                # deadline for as long as the wait lasted.
                self._release_expired(cycle_num)
            assert self.repeat_instruction is not None
            instruction_info, issue_thread = self.repeat_instruction
            assert instruction_info["name"] in UnPackerUnit.OPCODE_TO_HANDLER
            getattr(self, UnPackerUnit.OPCODE_TO_HANDLER[instruction_info["name"]])(
                instruction_info, issue_thread, instruction_info["instr_args"]
            )
            # An unpack that just came unblocked completed above; its data
            # phase starts at this tick, so the hold is anchored here — the
            # address phase was already charged at issue, and the blocked
            # cycles in between were the unit *waiting* on a Src bank, not
            # busy, so nothing was charged for them and nothing is
            # double-counted now.
            self._arm_pending_occupancy(cycle_num, cycle_num)
        else:
            super().clock_tick(cycle_num)
            # Anchored at the acceptance cycle (one before this retire tick),
            # exactly as the base batch arming anchors its charges.
            self._arm_pending_occupancy(cycle_num - 1, cycle_num)

    def _arm_pending_occupancy(self, anchor_cycle, cycle_num):
        cycles = self._pending_occupancy
        thread = self._pending_thread_block
        if thread is not None:
            self._pending_thread_block = None
            self.backend.block_thread_issue(
                thread, anchor_cycle + self._address_phase_cycles(), "UNPACK"
            )
        if cycles:
            self._pending_occupancy = 0
            self.occupy_for(anchor_cycle, cycles)
            # Whose instruction the hold represents, for the "in any stage of
            # this unpacker's pipeline" answer. Written only when a hold is
            # actually armed, so it stays ``None`` for the whole of a run with
            # the cost model off and the predicate never reaches
            # ``is_occupied()``.
            self._occupied_thread = self._pending_occupancy_thread
        if self._deferred_dvalid is not None and self._deferred_dvalid_cycle is None:
            # End of the transfer: the same deadline the occupancy just armed.
            # Set once, when the hand-over is first owed -- this method runs
            # again on every later tick, and recomputing the deadline from a
            # by-then-empty ``_pending_occupancy`` would collapse it onto the
            # next cycle. With no cost model ``cycles`` is 0, the deadline is
            # the anchor, and the hand-over below happens in this very tick --
            # exactly what the unpacker did before any of this existed.
            self._deferred_dvalid_cycle = anchor_cycle + cycles
            self._hand_over_src_bank(cycle_num)

    def _hand_over_src_bank(self, cycle_num):
        """Give the pending Src bank to the matrix unit, once the transfer ends.

        The ISA docs place the hand-over after the datum loop, not beside the
        instruction's issue: ``UNPACR_Regular.md``'s functional model runs its
        whole "Main unpack loop" and only then assigns ``(WhichUnpacker ? SrcB :
        SrcA)[...].AllowedClient = SrcClient::MatrixUnit``, and its Performance
        section says the instruction spends its address phase and then
        "execution proceeds in a pipelined fashion, with the primary bottleneck
        being the fetching of bytes from L1". So the bank becomes the matrix
        unit's when the last datum has been fetched, which is
        ``address phase + data phase`` cycles after the ``UNPACR`` was accepted
        -- precisely the occupancy the unit charges itself.

        tt-sim moves every datum in the retire tick, which is unobservable
        (nothing may read the bank until it changes hands) -- but flipping
        ``AllowedClient`` there too let the matrix unit start consuming the
        bank up to a whole data phase early. That is not a small error in one
        number: it is what a producer/consumer handshake is *made of*, and it
        left the LLK's unpack/math ping-pong resting on one-cycle margins that
        any timing change spends.
        """
        if (
            self._deferred_dvalid_cycle is None
            or cycle_num < self._deferred_dvalid_cycle
        ):
            return
        src, out_data_format, issue_thread, src_row_base = self._deferred_dvalid
        self._deferred_dvalid = None
        self._deferred_dvalid_cycle = None
        src.setAllowedClient(SrcRegister.SrcClient.MatrixUnit)
        # Latch the format the bank was written in, for the matrix unit's
        # implied-format read (see ``latch_src_data_format``).
        if out_data_format is not None:
            src.setDataFormat(out_data_format)
        self.srcBank ^= 1
        self.srcRow[issue_thread] = src_row_base

    def _address_phase_cycles(self):
        """The UNPACR entry's own occupancy: the >= 2-cycle address phase.

        Charged at its low end by the model (2, exact for uncompressed data —
        the only kind tt-sim unpacks), and 0 with no model or no entry.
        """
        model = self.cost_model
        if model is None:
            return 0
        return model.occupancy("UNPACR") or 0

    def handle_unpacr_nop(self, instruction_info, issue_thread, instr_args):
        if self.backend.blackhole:
            self._handle_unpacr_nop_blackhole(instruction_info, issue_thread)
            return
        args = instr_args["NoOp"]

        mode1 = args & 0x3
        mode2 = args & 0x7

        if mode1 == 0x1:
            # Set srcA or srcB to zero
            self.handle_set_src_to_zero(instruction_info, issue_thread, args)
        else:
            match mode2:
                case 0x2:
                    # Occupy Unpacker for one cycle
                    pass
                case 0x3 | 0x0:
                    # MMIO register write to Overlay STREAM_MSG_DATA_CLEAR_REG_INDEX
                    self.handle_write_stream_data_clear_reg_index(issue_thread, args)
                case 0x4:
                    # MMIO register write
                    self.handle_mmio_register_write(args)
                case 0x7:
                    # Give srcA or srcB banks to matrix unit
                    self.handle_give_src_to_fpu(issue_thread, args)
                case _:
                    raise NotImplementedError()

    def _handle_unpacr_nop_blackhole(self, instruction_info, issue_thread):
        # Blackhole re-lays out UNPACR_NOP entirely (ttsim data/bh); the WH
        # ``NoOp`` mode-select bits mean different things, so decode the BH fields
        # from the raw word: unpack_pop(1:0), src_clr_val_ctrl(3:2), bank_clr_ctrl(4),
        # stall_clr_cntrl(5), clr_to1_fmt_ctrl(7:6), set_dvalid(8), msg_clr_cnt(14:12),
        # stream_id(21:16), unpacker_select(23 = block select, srcA vs srcB).
        #
        # unpack_pop==1 implies stall-and-clear: wait for the matrix unit to have
        # consumed the bank we're about to reuse, then zero the unpack bank. The
        # FP32 copy path is ELWADD(unpacked srcA, *zeroed* srcB), so this clear is
        # what supplies srcB's zero operand — skipping it leaves srcB with stale
        # data and periodically corrupts the copied tile. An optional set_dvalid
        # then hands the (now-zeroed) bank to the matrix unit. Assert the control
        # fields we don't model are absent rather than mis-handle them silently.
        raw = instruction_info["raw_instruction"]
        unpack_pop = get_bits(raw, 0, 1)
        src_clr_val_ctrl = get_bits(raw, 2, 3)
        stall_clr_cntrl = get_nth_bit(raw, 5)
        set_dvalid = get_nth_bit(raw, 8)
        assert unpack_pop == 1, f"UNPACR_NOP unpack_pop={unpack_pop} not modelled"
        assert get_nth_bit(raw, 4) == 0, "UNPACR_NOP bank_clr_ctrl not modelled"
        assert get_bits(raw, 6, 7) == 0, "UNPACR_NOP clr_to1_fmt_ctrl not modelled"
        assert get_bits(raw, 12, 14) == 0, "UNPACR_NOP msg_clr_cnt not modelled"
        assert get_bits(raw, 16, 21) == 0, "UNPACR_NOP stream_id not modelled"

        is_srcb = self.unpacker_id == 1
        get_src = self.backend.getSrcB if is_srcb else self.backend.getSrcA
        matrix_bank = (
            self.backend.matrix_unit.srcBBank
            if is_srcb
            else self.backend.matrix_unit.srcABank
        )
        unpack_bank = self.srcBank
        wait_bank = unpack_bank if stall_clr_cntrl else matrix_bank

        # Stall until the matrix unit has released the bank we wait on.
        if get_src(wait_bank).getAllowedClient() != SrcRegister.SrcClient.Unpackers:
            self.blocked = True
            self.blocked_wait_bank = wait_bank
            self.repeat_instruction = (instruction_info, issue_thread)
            return
        self.blocked = False
        self.blocked_wait_bank = None
        self.repeat_instruction = None

        # Clear the unpack bank (0, or SrcA "negative inf" 0xFFFFE000 when asked).
        clear_val = 0xFFFFE000 if (not is_srcb and src_clr_val_ctrl) else 0
        src = get_src(unpack_bank)
        for i in range(64):
            for j in range(16):
                src[i, j] = clear_val

        if set_dvalid:
            self.handle_give_src_to_fpu(issue_thread, raw)

    def handle_give_src_to_fpu(self, issue_thread, args):
        # Unlike the regular UNPACR, this path moves no datums, so there is no
        # latched unpack state to take the format from -- read it from config,
        # as the hardware does.
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )
        outDataFormat = DataFormat(
            self.getConfigValue(
                stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Out_data_format"
            )
        )

        if self.unpacker_id == 0:
            src = self.backend.getSrcA(self.srcBank)
            setBase = "SRCA_SET_Base"
        else:
            src = self.backend.getSrcB(self.srcBank)
            setBase = "SRCB_SET_Base"

        src.setAllowedClient(SrcRegister.SrcClient.MatrixUnit)
        src.setDataFormat(outDataFormat)
        self.srcBank ^= 1
        self.srcRow[issue_thread] = (
            self.backend.getThreadConfigValue(issue_thread, setBase) << 4
        )

    def handle_mmio_register_write(self, args):
        accumulate = get_nth_bit(args, 3)
        value11 = (args >> 4) & 0x3FF
        addrMid = (args >> 16) & 0x3F
        addrSel = get_nth_bit(args, 22)

        addr = 0xFFB00000 + self.setRegBase[addrSel] + (addrMid << 12)
        if accumulate:
            accValue = self.setRegAcc
            if value11 == 0:
                accValue = 0
            else:
                accValue = (accValue + value11) & 0x1FFFF
                self.backend.getAddressableMemory().write(addr, conv_to_bytes(accValue))
            self.setRegAcc = accValue
        else:
            self.backend.getAddressableMemory().write(addr, conv_to_bytes(value11))

    def handle_write_stream_data_clear_reg_index(self, issue_thread, args):
        clearCount = (args >> 4) & 0x3FF
        whichStream = (args >> 16) & 0x1F

        if clearCount != 0:
            streamId = whichStream
        else:
            streamId = self.backend.getThreadConfigValue(
                issue_thread, "NOC_OVERLAY_MSG_CLEAR_StreamId_" + str(self.unpacker_id)
            )

        overlay_addr = NoCOverlay.NOC_STREAM_REG_SPACE_SIZE * streamId + (
            NoCOverlay.STREAM_MSG_DATA_CLEAR_REG_INDEX << 2
        )
        self.backend.getAddressableMemory().write(
            0xFFB40000 + overlay_addr, conv_to_bytes(1)
        )

    def handle_set_src_to_zero(self, instruction_info, issue_thread, args):
        negativeInfSrcA = get_nth_bit(args, 2)
        bothBanks = get_nth_bit(args, 3)
        waitLikeUnpacr = get_nth_bit(args, 4)

        unpackBank = self.srcBank

        if self.unpacker_id == 0:
            if waitLikeUnpacr:
                srcBank = self.srcBank
            else:
                srcBank = self.backend.matrix_unit.srcABank
            if (
                self.backend.getSrcA(srcBank).getAllowedClient()
                != SrcRegister.SrcClient.Unpackers
            ):
                self.blocked = True
                self.blocked_wait_bank = srcBank
                self.repeat_instruction = (instruction_info, issue_thread)
                return
        else:
            if waitLikeUnpacr:
                srcBank = self.srcBank
            else:
                srcBank = self.backend.matrix_unit.srcBBank
            if (
                self.backend.getSrcB(srcBank).getAllowedClient()
                != SrcRegister.SrcClient.Unpackers
            ):
                self.blocked = True
                self.blocked_wait_bank = srcBank
                self.repeat_instruction = (instruction_info, issue_thread)
                return

        self.blocked = False
        self.blocked_wait_bank = None
        self.repeat_instruction = None

        for bank in range(2):
            if bothBanks or bank == unpackBank:
                if self.unpacker_id == 0:
                    clearVal = ~0 if negativeInfSrcA else 0
                    for i in range(64):
                        for j in range(16):
                            self.backend.getSrcA(bank)[i, j] = clearVal
                else:
                    for i in range(64):
                        for j in range(16):
                            self.backend.getSrcB(bank)[i, j] = 0

    def handle_unpacr(self, instruction_info, issue_thread, instr_args):
        one_bit = instr_args["SearchCacheFlush"]
        thirteen_bit = instr_args["CfgContextCntInc"]
        if one_bit:
            self.handle_flush(instruction_info, issue_thread, instr_args)
        else:
            if thirteen_bit:
                self.handle_increment_context_counter(
                    instruction_info, issue_thread, instr_args
                )
            else:
                self.handle_regular(instruction_info, issue_thread, instr_args)

    def handle_flush(self, instruction_info, issue_thread, instr_args):
        pass

    def handle_increment_context_counter(
        self, instruction_info, issue_thread, instr_args
    ):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        increment_ctr = self.context_counter[issue_thread]
        thcon_context_count = self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Context_count"
        )
        if increment_ctr >= (1 << thcon_context_count):
            increment_ctr = 0
        self.context_counter[issue_thread] = increment_ctr

    def wrapAddr(self, stateID, addr):
        if addr is None:
            return None
        if (
            addr
            > self.getConfigValue(
                stateID,
                "THCON_SEC" + str(self.unpacker_id) + "_REG2_Unpack_limit_address",
            )
            * 16
        ):
            addr -= (
                self.getConfigValue(
                    stateID,
                    "THCON_SEC" + str(self.unpacker_id) + "_REG2_Unpack_fifo_size",
                )
                * 16
            )
        return addr

    def read_unpack_configuration(
        self,
        issue_thread,
        multiContextMode,
        useContextCounter,
        contextNumber,
        contextADC,
    ):
        if multiContextMode:
            if useContextCounter:
                whichContext = self.context_counter[issue_thread]
            else:
                whichContext = contextNumber
            whichContext += self.backend.getThreadConfigValue(
                issue_thread,
                "UNPACK_MISC_CFG_CfgContextOffset_" + str(self.unpacker_id),
            )

            whichADC = contextADC
            assert not (self.unpacker_id == 1 and whichContext >= 2)
            assert whichADC != 3
        else:
            whichContext = 0
            whichADC = issue_thread

        return whichContext, whichADC

    def get_isUncompressed(
        self, configDescriptor, stateID, multiContextMode, whichContext
    ):
        return True
        if multiContextMode:
            return self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG2_Disable_zero_compress_cntx"
                + str(whichContext),
            )
        else:
            return get_nth_bit(configDescriptor[0], 4)

    def get_xyzw_dim(self, configDescriptor, stateID, multiContextMode, whichContext):
        if multiContextMode and self.unpacker_id == 0:
            xdim = self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG5_Tile_x_dim_cntx"
                + str(whichContext & 3),
            )
        else:
            xdim = get_bits(configDescriptor[0], 16, 31)

        ydim = get_bits(configDescriptor[1], 0, 7)
        zdim = get_bits(configDescriptor[1], 16, 23)
        if not zdim:
            zdim = 1
        wdim = get_bits(configDescriptor[2], 0, 7)
        if not wdim:
            wdim = 1
        if not ydim:
            ydim = 1

        return xdim, ydim, zdim, wdim

    def get_inaddr(self, configDescriptor, stateID, multiContextMode, whichContext):
        if multiContextMode and whichContext != 0:
            inAddr = self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG3_Base_cntx"
                + str(whichContext)
                + "_address",
            ) + (
                self.getConfigValue(
                    stateID,
                    "THCON_SEC"
                    + str(self.unpacker_id)
                    + "_REG7_Offset_cntx"
                    + str(whichContext & 3)
                    + "_address",
                )
                & 0xFFFF
            )
        else:
            inAddr = self.getConfigValue(
                stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG3_Base_address"
            ) + (
                self.getConfigValue(
                    stateID,
                    "THCON_SEC" + str(self.unpacker_id) + "_REG7_Offset_address",
                )
                & 0xFFFF
            )
        inAddr = (inAddr + 1 + get_bits(configDescriptor[3], 24, 31)) * 16
        return inAddr

    def get_first_datum_and_inputNumDatums(
        self,
        configDescriptor,
        stateID,
        issue_thread,
        whichADC,
        whichContext,
        isUncompressed,
        rowSearch,
        multiContextMode,
        blobsPerXYPlane,
        xdim,
        ydim,
        zdim,
        wdim,
        inAddr_RowStart,
    ):
        adc_xy = self.backend.getADC(whichADC).Unpacker[self.unpacker_id].Channel[0]
        adc_zw = self.backend.getADC(issue_thread).Unpacker[self.unpacker_id].Channel[0]
        if isUncompressed:
            if not rowSearch:
                xpos = adc_xy.X
                ypos = adc_xy.Y
                xend = (
                    self.backend.getADC(whichADC)
                    .Unpacker[self.unpacker_id]
                    .Channel[1]
                    .X
                    + 1
                )
            elif blobsPerXYPlane:
                if multiContextMode and self.unpacker_id == 0:
                    blobsYStart = self.getConfigValue(
                        stateID,
                        "UNP0_BLOBS_Y_START_CNTX_"
                        + str(whichContext & 2)
                        + "_blobs_y_start",
                    )
                else:
                    blobsYStart = get_bits(configDescriptor[2], 16, 32)
                xpos = get_bits(blobsYStart, adc_xy.X & 7, (adc_xy.X & 7) + 4) << 4
                ypos = 0
                x71 = (adc_xy.X & 7) + 1
                if x71 == blobsPerXYPlane:
                    xend = xdim & 0x1F0
                else:
                    xend = get_bits(blobsYStart, x71, x71 + 4) << 4
            else:
                xpos = 0
                ypos = adc_xy.Y
                xend = (
                    self.backend.getADC(whichADC)
                    .Unpacker[self.unpacker_id]
                    .Channel[1]
                    .X
                )
            firstDatum = ((adc_zw.W * zdim + adc_zw.Z) * ydim + ypos) * xdim + xpos
            inputNumDatums = xend - xpos
        else:
            raise NotImplementedError()

        return firstDatum, inputNumDatums

    def generate_input_addresses_and_sizes(
        self, issue_thread, stateID, multiContextMode, whichContext, whichADC, rowSearch
    ):
        configDescriptor = self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG0_TileDescriptor", 4
        )

        isUncompressed = self.get_isUncompressed(
            configDescriptor, stateID, multiContextMode, whichContext
        )
        xdim, ydim, zdim, wdim = self.get_xyzw_dim(
            configDescriptor, stateID, multiContextMode, whichContext
        )

        if multiContextMode and self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Ovrd_data_format"
        ):
            inDataFormat = DataFormat(
                self.getConfigValue(
                    stateID,
                    "THCON_SEC"
                    + str(self.unpacker_id)
                    + "_REG7_Unpack_data_format_cntx"
                    + str(whichContext),
                )
            )
        else:
            inDataFormat = DataFormat(get_bits(configDescriptor[0], 0, 3))

        datumSizeBytes = int(DATA_FORMAT_TO_BITS[inDataFormat] / 8)

        inAddr = self.get_inaddr(
            configDescriptor, stateID, multiContextMode, whichContext
        )

        blobsPerXYPlane = get_bits(configDescriptor[3], 8, 11)
        if not isUncompressed:
            inAddr_RowStart = inAddr
            if blobsPerXYPlane:
                numBlobs = blobsPerXYPlane * zdim * wdim
                inAddr += ceil((numBlobs + 1) * 2 / 16) * 16
            else:
                numRows = ydim * zdim * wdim
                inAddr += ceil((numRows + 1) * 2 / 16) * 16
        else:
            inAddr_RowStart = 0

        inAddr_Exponents = 0
        if inDataFormat.isBFPFormat() and not self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Force_shared_exp"
        ):
            inAddr_Exponents = inAddr
            if inDataFormat == DataFormat.BFP8:
                # missing BFP8a and ConfigDescriptor.NoBFPExpSection
                numElements = xdim * ydim * zdim * wdim
                numExponents = ceil(numElements / 16)
                inAddr += ceil(numExponents / 16) * 16

        firstDatum, inputNumDatums = self.get_first_datum_and_inputNumDatums(
            configDescriptor,
            stateID,
            issue_thread,
            whichADC,
            whichContext,
            isUncompressed,
            rowSearch,
            multiContextMode,
            blobsPerXYPlane,
            xdim,
            ydim,
            zdim,
            wdim,
            inAddr_RowStart,
        )

        inAddr_Datums = inAddr
        inAddr_Exponents += int(firstDatum / 16)
        if isUncompressed:
            inAddr_Datums += firstDatum * datumSizeBytes
            inAddr_Deltas = None
        else:
            inAddr_Datums += int(firstDatum / 32) * int(32 * datumSizeBytes + 32 * 0.5)
            inAddr_Deltas = inAddr_Datums + 32 * datumSizeBytes
            inAddr_Datums += (firstDatum % 32) * datumSizeBytes
            inAddr_Deltas += int((firstDatum % 32) * 0.5)

        inAddr_Exponents = self.wrapAddr(stateID, inAddr_Exponents)
        inAddr_Datums = self.wrapAddr(stateID, inAddr_Datums)
        inAddr_Deltas = self.wrapAddr(stateID, inAddr_Deltas)

        # Number of datums read from L1 before the input address advances by
        # RowStride rather than by one datum. Blackhole reads wider input rows
        # for anything above a byte per datum (UNPACR_Regular.md,
        # "UnpackRowWidth"), which is what makes a contiguous RowStride
        # arch-dependent.
        if self.backend.blackhole and datumSizeBytes > 1:
            unpackRowWidth = 32
        else:
            unpackRowWidth = 16

        discontiguousInputRows = self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Tileize_mode"
        )
        if discontiguousInputRows:
            # Note that each Shift_amount_cntx is a 4-bit field, so there's 12 bits of
            # precision here, and therefore the maximum RowStride is 65520 bytes.
            rowStride = (
                (
                    self.getConfigValue(
                        stateID,
                        "THCON_SEC"
                        + str(self.unpacker_id)
                        + "_REG2_Shift_amount_cntx0",
                    )
                    << 4
                )
                | (
                    self.getConfigValue(
                        stateID,
                        "THCON_SEC"
                        + str(self.unpacker_id)
                        + "_REG2_Shift_amount_cntx1",
                    )
                    << 8
                )
                | (
                    self.getConfigValue(
                        stateID,
                        "THCON_SEC"
                        + str(self.unpacker_id)
                        + "_REG2_Shift_amount_cntx2",
                    )
                    << 12
                )
            )
        else:
            rowStride = datumSizeBytes * unpackRowWidth

        return (
            inAddr_Datums,
            datumSizeBytes,
            inputNumDatums,
            inAddr_Deltas,
            inAddr_Exponents,
            rowStride,
            unpackRowWidth,
            discontiguousInputRows,
            isUncompressed,
            inDataFormat,
        )

    def generate_output_address(
        self, stateID, issue_thread, multiContextMode, whichContext
    ):
        if multiContextMode and self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Ovrd_data_format"
        ):
            outDataFormat = DataFormat(
                self.getConfigValue(
                    stateID,
                    "THCON_SEC"
                    + str(self.unpacker_id)
                    + "_REG7_Unpack_out_data_format_cntx"
                    + str(whichContext),
                )
            )
        else:
            outDataFormat = DataFormat(
                self.getConfigValue(
                    stateID,
                    "THCON_SEC" + str(self.unpacker_id) + "_REG2_Out_data_format",
                )
            )

        if self.unpacker_id == 0:
            if multiContextMode:
                unpackToDst = self.getConfigValue(
                    stateID,
                    "THCON_SEC"
                    + str(self.unpacker_id)
                    + "_REG2_Unpack_if_sel_cntx"
                    + str(whichContext),
                )
            else:
                unpackToDst = self.getConfigValue(
                    stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Unpack_If_Sel"
                )
            transpose = self.getConfigValue(
                stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Haloize_mode"
            )
        else:
            unpackToDst = False
            transpose = False

        adc_out = (
            self.backend.getADC(issue_thread).Unpacker[self.unpacker_id].Channel[1]
        )
        outAddr = (
            self.getConfigValue(
                stateID, "UNP" + str(self.unpacker_id) + "_ADDR_BASE_REG_1_Base"
            )
            + adc_out.Y
            * self.getConfigValue(
                stateID, "UNP" + str(self.unpacker_id) + "_ADDR_CTRL_XY_REG_1_Ystride"
            )
            + adc_out.Z
            * self.getConfigValue(
                stateID, "UNP" + str(self.unpacker_id) + "_ADDR_CTRL_ZW_REG_1_Zstride"
            )
            + adc_out.W
            * self.getConfigValue(
                stateID, "UNP" + str(self.unpacker_id) + "_ADDR_CTRL_ZW_REG_1_Wstride"
            )
        )

        if (
            outDataFormat == DataFormat.FP32
            or outDataFormat == DataFormat.TF32
            or outDataFormat == DataFormat.INT32
        ):
            assert not outAddr & 3
            outAddr >>= 2
        elif (
            outDataFormat == DataFormat.FP16
            or outDataFormat == DataFormat.BF16
            or outDataFormat == DataFormat.UINT16
        ):
            assert not outAddr & 1
            outAddr >>= 1

        if multiContextMode and self.unpacker_id == 0:
            ctxOutAddr = self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG5_Dest_cntx"
                + str(whichContext & 3)
                + "_address",
            )
            if unpackToDst or self.getConfigValue(
                stateID,
                "UNP"
                + str(self.unpacker_id)
                + "_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr",
            ):
                outAddr += ctxOutAddr
            else:
                outAddr = ctxOutAddr

        return outAddr, outDataFormat, unpackToDst, transpose

    def check_unpacker_settings(
        self,
        transpose,
        discontiguousInputRows,
        inAddr_Datums,
        upsampleZeroes,
        isUncompressed,
        unpackToDst,
        colShift,
    ):
        if transpose or discontiguousInputRows:
            # These modes require that InAddr_Datums start at an aligned 16 byte boundary.
            assert inAddr_Datums == floor(inAddr_Datums / 16) * 16

        assert not (
            discontiguousInputRows and (upsampleZeroes > 0 or not isUncompressed)
        )
        assert not (unpackToDst and (colShift or transpose))

    def check_modelled_settings(
        self,
        rowStride,
        unpackRowWidth,
        datumSizeBytes,
        discontiguousInputRows,
        upsampleRate,
        upsampleZeroes,
        upsampleInterleave,
        colShift,
        outAddr,
    ):
        """Reject the UNPACR modes whose datum walk this unpacker does not model.

        Deliberately *not* suppressible by an environment variable, unlike the
        NoC alignment checks: those guard against hardware ``UndefinedBehavior``
        that a user may legitimately want to explore, whereas these are simply
        "tt-sim does not implement this", where carrying on reads the wrong L1
        addresses or writes the wrong Src/Dst rows with nothing to notice it by.

        ``RowStride`` itself is now modelled (see ``perform_unpack``), so what
        is left here is the sub-byte datum the strided walk cannot address, and
        the two "shift the datums about" modes the walk still knows nothing of.
        """
        if outAddr & 15:
            # The doc's output loop derives both coordinates from one running
            # counter -- ``Row = OutAddr / 16; Col = OutAddr & 15`` -- and marks
            # a start address that is not a whole row UnsupportedFunctionality
            # ("no known usage, confidence in specification is weak"); the
            # reference simulator refuses it too. The walk below splits the
            # counter into a row index and a column of 16, which is exact only
            # while the low nibble is zero, so refuse rather than silently drop
            # the column offset (and the row carry it would produce).
            raise NotImplementedError(
                f"Unpacker {self.unpacker_id}: an output address of {outAddr} "
                f"datums does not start on a 16-datum row boundary "
                f"(OutAddr & 15 == {outAddr & 15}). UNPACR_Regular.md marks a "
                f"misaligned OutAddr UnsupportedFunctionality; only whole-row "
                f"output addresses are modelled."
            )

        if discontiguousInputRows and datumSizeBytes < 1:
            # The doc calls this UndefinedBehavior outright ("BFP2(a) has no
            # valid Throttle_mode with tileize; BFP4(a) addressing is
            # incorrect"), and the walk below indexes L1 in whole datums.
            raise NotImplementedError(
                f"Unpacker {self.unpacker_id}: Tileize_mode with a sub-byte "
                f"datum ({datumSizeBytes} bytes per datum, i.e. a BFP2/BFP4 "
                f"format) is not modelled -- UNPACR_Regular.md marks the "
                f"combination UndefinedBehavior."
            )

        if upsampleZeroes:
            raise NotImplementedError(
                f"Unpacker {self.unpacker_id}: Upsample_rate={upsampleRate} "
                f"(UpsampleZeroes={upsampleZeroes}, Upsample_and_interleave="
                f"{int(bool(upsampleInterleave))}) is not modelled. Upsampling "
                f"emits {upsampleZeroes + 1} output datums per input datum -- the "
                f"datum then {upsampleZeroes} zeroes, or, with interleave, the "
                f"datum then {upsampleZeroes} output positions left untouched for "
                f"a later unpack to fill -- so the walk here, which emits one "
                f"output datum per input datum, would put every datum after the "
                f"first in the wrong SrcA/SrcB/Dst position. Only Upsample_rate=0 "
                f"is modelled; Upsample_and_interleave is a no-op at that rate and "
                f"is accepted. Note the ISA docs mark upsampling "
                f"UnsupportedFunctionality themselves ('no known usage, confidence "
                f"in specification is weak') -- see UNPACR_Regular.md."
            )

        if colShift:
            raise NotImplementedError(
                f"Unpacker {self.unpacker_id}: ColShift={colShift} "
                f"(THCON_SEC{self.unpacker_id}_REG2_Shift_amount_cntx, with "
                f"Tileize_mode clear) is not modelled. It shifts each datum "
                f"{colShift} columns towards column 0 and *drops* the datums "
                f"whose source column is below {colShift} -- 'if (Row < 4 || Col "
                f"< ColShift) continue;' -- leaving the top {colShift} columns of "
                f"every SrcA row untouched. Only ColShift=0 is modelled. Note the "
                f"ISA docs mark ColShift UnsupportedFunctionality themselves in "
                f"the same breath as upsampling ('no known usage, confidence in "
                f"specification is weak'), and the reference simulator declines "
                f"it too, so there is nothing to validate an implementation "
                f"against -- see UNPACR_Regular.md."
            )

    def perform_unpack(
        self,
        stateID,
        issue_thread,
        inputNumDatums,
        inAddr_Datums,
        outAddr,
        datumSizeBytes,
        inDataFormat,
        outDataFormat,
        unpackToDst,
        transpose,
        allDatumsAreZero,
        rowStride,
        unpackRowWidth,
    ):
        # The input walk reads ``unpackRowWidth`` datums contiguously and then
        # advances by ``rowStride`` bytes rather than by one datum -- i.e. input
        # datum ``i`` comes from ``inAddr_Datums + datumSizeBytes * (i %
        # unpackRowWidth) + rowStride * (i // unpackRowWidth)``. With
        # ``Tileize_mode`` clear ``rowStride`` *is* ``datumSizeBytes *
        # unpackRowWidth``, so the rows abut and this degenerates to a flat read;
        # with it set the input rows are a tile's worth of L1 apart, which is how
        # tt-metal's ``tilize`` LLK gathers a row-major block into faces.
        # ``unpackRowWidth`` is 16 on Wormhole and 32 on Blackhole above one byte
        # per datum, so the *same* RowStride means different things per arch.
        #
        # The output side is untouched by any of this: one output datum per
        # input datum, at its own column, in destination rows of 16.
        # UpsampleZeroes / UpsampleInterleave / ColShift, which would change
        # that, are still rejected by ``check_modelled_settings``.
        start_row = int(outAddr / 16)
        if self.unpacker_id == 0:
            assert start_row >= 4
            start_row -= 4

        if self.getDiagnosticSettings().reportUnpacking():
            tgt = (
                "srcB" if self.unpacker_id == 1 else ("dst" if unpackToDst else "srcA")
            )
            stride_note = ""
            if rowStride != datumSizeBytes * unpackRowWidth:
                stride_note = (
                    f", strided (RowStride {rowStride} bytes every "
                    f"{unpackRowWidth} datums)"
                )
            print(
                f"Unpacker {self.unpacker_id}: start read at {hex(inAddr_Datums)} for "
                f"{inputNumDatums} datums of bytes size {datumSizeBytes} "
                f"starting write to {tgt} at row {start_row}, read data type "
                f"{DATA_FORMAT_TO_NAME[inDataFormat]} -> write data type "
                f"{DATA_FORMAT_TO_NAME[outDataFormat]}{stride_note}"
            )

        numRows = int(inputNumDatums / 16)
        if self._unpack_block(
            stateID,
            issue_thread,
            numRows,
            inAddr_Datums,
            start_row,
            datumSizeBytes,
            inDataFormat,
            outDataFormat,
            unpackToDst,
            transpose,
            allDatumsAreZero,
            rowStride,
            unpackRowWidth,
        ):
            return

        # The input-row cursor of the doc's loop: it walks forward a datum at a
        # time and, every ``unpackRowWidth`` datums, rewinds the row and jumps by
        # RowStride instead.
        datumIndex = 0
        for row in range(numRows):
            for col in range(16):
                assert datumSizeBytes <= 4
                raw_datum = conv_to_uint32(
                    self.backend.addressable_memory.read(inAddr_Datums, datumSizeBytes)
                )

                datum = self.formatConversion(
                    stateID, inDataFormat, outDataFormat, raw_datum, unpackToDst
                )

                if allDatumsAreZero:
                    datum = 0
                inAddr_Datums += datumSizeBytes
                datumIndex += 1
                if datumIndex % unpackRowWidth == 0:
                    inAddr_Datums += rowStride - datumSizeBytes * unpackRowWidth

                # Destination row/column for this datum. Kept in locals: `row`
                # and `col` are the loop variables, and the adjustments below
                # used to write back into `row`, so every datum after the first
                # in a row saw an already-adjusted row (harmless while the
                # adjustment was zero, wrong as soon as it is not).
                outRow, outCol = row, col
                if self.unpacker_id == 1:
                    # always srcB
                    outRow = (outRow + self.srcRow[issue_thread] + start_row) & 0x3F
                    self.backend.getSrcB(self.srcBank)[outRow, outCol] = datum
                else:
                    # Always srcA
                    if not unpackToDst:
                        # ``Row`` is ``OutAddr / 16`` (less the four-row skew),
                        # and OutAddr advances one per datum from the *initial*
                        # output address -- so the row this UNPACR starts at is
                        # part of the row index whichever way the doc's two
                        # branches then adjust it. Dropping ``start_row`` on the
                        # SetOvrdWithAddr branch (which every current LLK takes)
                        # collapsed every UNPACR of an unpack-side untilize onto
                        # SrcA row 0: llk_unpack_untilize steps the *output*
                        # address generator, bumping ADC channel 1's Y by one per
                        # UNPACR against a 16-datum Ystride, and that Y is the
                        # only thing that says which SrcA row a face row lands in.
                        if self.backend.getThreadConfigValue(
                            issue_thread, "SRCA_SET_SetOvrdWithAddr"
                        ):
                            # SrcA is 64 rows and the row index is six bits, so
                            # a row past the end wraps to the start. Blackhole's
                            # UNPACR_Regular.md says so outright ("Row &= 63;
                            # allowed for BH fast tilize"); Wormhole's calls it
                            # UndefinedBehavior, but its own LLK relies on the
                            # wrap -- the SrcA clear ahead of the int32/fp32
                            # SFPU kernels (examples five, five-fp, loopback)
                            # reaches this with ADC channel 1's Z accumulated to
                            # exactly 64 rows, i.e. row 64 == row 0.
                            outRow = (outRow + start_row) & 0x3F
                        else:
                            assert outRow < 16
                            outRow += self.srcRow[issue_thread] + start_row

                        # Haloize transposes each 16x16 block on the way into
                        # SrcA. It is driven by the destination row/column, so
                        # it applies whichever way the row was derived -- the
                        # SetOvrdWithAddr path (which every current LLK takes)
                        # included. reduce_tile's REDUCE_ROW is the first path
                        # to reach it: it transposes the data face so a
                        # column-wise GMPOOL/MVMUL reduces along rows.
                        if transpose:
                            rowLowBits = outCol
                            outCol = outRow & 0xF
                            outRow = (outRow & ~0xF) | rowLowBits
                        self.backend.getSrcA(self.srcBank)[outRow, outCol] = datum
                    else:
                        if self.backend.getThreadConfigValue(
                            issue_thread, "SRCA_SET_SetOvrdWithAddr"
                        ):
                            outRow &= 15
                        else:
                            outRow &= 0x3FF
                        if DATA_FORMAT_TO_BITS[outDataFormat] == 32:
                            self.backend.getDst().setDst32b(
                                outRow + start_row, outCol, datum
                            )
                        else:
                            self.backend.getDst().setDst16b(
                                outRow + start_row, outCol, datum
                            )
                outAddr += 1

    def _unpack_block(
        self,
        stateID,
        issue_thread,
        numRows,
        inAddr_Datums,
        start_row,
        datumSizeBytes,
        inDataFormat,
        outDataFormat,
        unpackToDst,
        transpose,
        allDatumsAreZero,
        rowStride,
        unpackRowWidth,
    ):
        """Move the whole ``numRows x 16`` rectangle at once, or decline it.

        The datum loop in ``perform_unpack`` is a rectangle on the *output*
        side -- every destination index is arithmetic on ``(row, col)`` -- and
        on the input side it is either one contiguous run or, under
        ``Tileize_mode``, a run of ``unpackRowWidth`` datums repeated every
        ``rowStride`` bytes. Both are index arithmetic numpy can do in one go:
        the block reads out of L1 in one slice (gathering by an index array when
        the rows do not abut), converts in one call -- ``formatConversion`` below
        is the *same* function, handed an int64 array instead of an int, exactly
        as the matrix unit's operand gather does -- and lands in one indexed
        assignment. Returns True if it did the unpack.

        The cases it declines, and why, are all "this is no longer a rectangle
        the index arithmetic describes":

        * a datum that is not a whole 1, 2 or 4 bytes (the BFP formats, with
          their shared exponents, which the scalar loop does not handle either);
        * ``FP32 -> FP16`` out, the one conversion on this path that still
          branches per datum (``FP32ToFP16`` saturates and flushes);
        * a row count large enough for the destination row map to alias, where
          an indexed assignment would depend on numpy's ordering rather than the
          scalar loop's explicit last-write-wins.
        """
        dtype = self._BLOCK_DTYPE.get(datumSizeBytes)
        if dtype is None or numRows <= 0:
            return False
        if inDataFormat == DataFormat.FP32 and outDataFormat == DataFormat.FP16:
            return False
        if rowStride % datumSizeBytes:
            # The gather below indexes in whole datums. RowStride is always a
            # multiple of 16 bytes so this cannot fire today, but if it ever
            # does, the scalar loop -- which walks in bytes -- is still exact.
            return False

        # Destination indices, mirroring the scalar loop's row/column arithmetic
        # with `row` as an array and `col` as the 16-element axis.
        rows = np.arange(numRows)
        if unpackToDst:
            # Unpacker 0 only, and check_unpacker_settings has already ruled out
            # transpose here, so this writes whole Dst rows.
            if self.backend.getThreadConfigValue(
                issue_thread, "SRCA_SET_SetOvrdWithAddr"
            ):
                if numRows > 16:
                    return False
                rows = rows & 15
            else:
                if numRows > 1024:
                    return False
                rows = rows & 0x3FF
            rows = rows + start_row
        elif self.unpacker_id == 1:
            # Always srcB, and the row index wraps within the 64-row bank.
            if numRows > 64:
                return False
            rows = (rows + self.srcRow[issue_thread] + start_row) & 0x3F
        elif self.backend.getThreadConfigValue(
            issue_thread, "SRCA_SET_SetOvrdWithAddr"
        ):
            # Same as the scalar loop: the row this UNPACR's output address
            # starts at, wrapped within the 64-row bank. Consecutive rows stay
            # distinct under the wrap as long as there are at most 64 of them,
            # so the indexed assignment below cannot alias.
            assert numRows <= 64
            rows = (rows + start_row) & 0x3F
        else:
            assert numRows <= 16  # ... and its `outRow < 16`
            rows = rows + self.srcRow[issue_thread] + start_row

        # Read the datums, convert them, and zero them if asked -- in the same
        # order as the scalar loop, so a format combination it rejects is still
        # rejected here.
        numDatums = numRows * 16
        if rowStride == datumSizeBytes * unpackRowWidth:
            # Contiguous: the whole walk is one slice of L1.
            raw = np.frombuffer(
                self.backend.addressable_memory.read(
                    inAddr_Datums, numDatums * datumSizeBytes
                ),
                dtype=dtype,
            ).astype(np.int64)
        else:
            # Strided (Tileize_mode): the source offsets are the scalar loop's
            # `datumSizeBytes * (i % unpackRowWidth) + rowStride * (i //
            # unpackRowWidth)`, as an index array -- in whole datums, which
            # RowStride always is (it is a multiple of 16 bytes, and a datum here
            # is 1, 2 or 4). One read spanning exactly what the walk touches,
            # then a gather; nothing outside the walk's own footprint is read.
            index = np.arange(numDatums)
            offsets = (index % unpackRowWidth) + (index // unpackRowWidth) * (
                rowStride // datumSizeBytes
            )
            span = (int(offsets.max()) + 1) * datumSizeBytes
            raw = np.frombuffer(
                self.backend.addressable_memory.read(inAddr_Datums, span),
                dtype=dtype,
            )[offsets].astype(np.int64)
        values = self.formatConversion(
            stateID, inDataFormat, outDataFormat, raw, unpackToDst
        )
        if allDatumsAreZero:
            values = np.zeros_like(raw)
        values = values.reshape(numRows, 16)

        if unpackToDst:
            dst = self.backend.getDst()
            if DATA_FORMAT_TO_BITS[outDataFormat] == 32:
                dst.setDst32bRows(rows, values)
            else:
                dst.setDst16bRows(rows, values)
            return True

        outRows = rows[:, np.newaxis]
        outCols = np.arange(16)[np.newaxis, :]
        if transpose:
            # Haloize transposes each 16x16 block on the way into SrcA; see the
            # scalar loop for what drives it. Same two assignments, done at once
            # (the tuple's right-hand side is evaluated before either lands, as
            # the scalar version's `rowLowBits` temporary ensures).
            outRows, outCols = (outRows & ~0xF) | outCols, outRows & 0xF

        src = self.backend.getSrcB if self.unpacker_id == 1 else self.backend.getSrcA
        src(self.srcBank).writeDatums(outRows, outCols, values)
        return True

    def increment_counter(
        self, stateID, issue_thread, whichContext, multiContextMode, useContextCounter
    ):
        if multiContextMode and useContextCounter:
            incrementedCounter = whichContext + 1
            if incrementedCounter >= (
                1
                << self.getConfigValue(
                    stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Context_count"
                )
            ):
                incrementedCounter = 0
            self.context_counter[issue_thread] = incrementedCounter

    def update_ADC(self, issue_thread, whichADC, ch0YInc, ch0ZInc, ch1YInc, ch1ZInc):
        for i in range(3):
            if i == issue_thread or i == whichADC:
                self.backend.getADC(i).Unpacker[self.unpacker_id].Channel[
                    0
                ].Y += ch0YInc
                self.backend.getADC(i).Unpacker[self.unpacker_id].Channel[
                    0
                ].Z += ch0ZInc
                self.backend.getADC(i).Unpacker[self.unpacker_id].Channel[
                    1
                ].Y += ch1YInc
                self.backend.getADC(i).Unpacker[self.unpacker_id].Channel[
                    1
                ].Z += ch1ZInc

    def flip_src_banks(self, flipSrc, issue_thread, outDataFormat=None):
        srcRowBase = (
            self.backend.getThreadConfigValue(issue_thread, "SRCB_SET_Base")
            if self.unpacker_id
            else self.backend.getThreadConfigValue(issue_thread, "SRCA_SET_Base")
        ) << 4
        if flipSrc:
            # Release the current bank to the matrix unit and advance to the
            # other bank — the same effect as the explicit "give src to fpu"
            # UNPACR_NOP (``handle_give_src_to_fpu``). This is a *set*, not a
            # toggle: Blackhole kernels mark dvalid inline on the regular UNPACR
            # (rather than issuing the Wormhole-only SETDVALID NoOp), so this
            # runs once per unpacked face; toggling would flip an even number of
            # times and leave the bank owned by the unpackers, starving the FPU.
            if self.unpacker_id == 0:
                src = self.backend.getSrcA(self.srcBank)
            else:
                src = self.backend.getSrcB(self.srcBank)
            # Owed, not done: the hand-over lands at the end of the pipelined
            # transfer, which ``_hand_over_src_bank`` resolves once the caller
            # knows the cycle the data phase runs to. The bank pointer and the
            # row base go with it, because the functional model moves all three
            # together -- ``AllowedClient = MatrixUnit; SrcBank ^= 1;
            # SrcRow[CurrentThread] = SrcRowBase`` is one block at the end of
            # UNPACR_Regular.md's pseudocode -- and because they are what
            # ``STALLWAIT``'s C8/C9 ("SrcA/SrcB available for unpacker writes")
            # read to decide which bank they are asking about. Splitting them
            # would leave the unpack thread asking about the *next* bank while
            # the previous one had not yet changed hands.
            self._deferred_dvalid = (src, outDataFormat, issue_thread, srcRowBase)
            self._deferred_dvalid_cycle = None
        else:
            self.srcRow[issue_thread] += 16 + srcRowBase

    def formatConversion(
        self, stateID, inDataFormat, outDataFormat, raw_datum, unpackToDst
    ):
        if inDataFormat == DataFormat.FP32:
            match outDataFormat:
                case DataFormat.FP32:
                    pass
                case DataFormat.TF32:
                    if unpackToDst:
                        # when unpacking to Dst TF32 means FP32
                        return DataFormatConversions.FP32ToDstFormatFP32(raw_datum)
                    else:
                        return DataFormatConversions.TF32ToSrcFormatTF32(
                            raw_datum >> 13
                        )
                case DataFormat.BF16:
                    # Flush denormals to signed zero, then truncate toward zero
                    # -- which is precisely ``FP32ToBF16``, written there
                    # without a branch so that this converts a whole block as
                    # readily as one datum (see DataFormatConversions).
                    raw_datum = DataFormatConversions.FP32ToBF16(raw_datum)
                    inDataFormat = DataFormat.BF16
                case DataFormat.FP16:
                    raw_datum = DataFormatConversions.FP32ToFP16(raw_datum)
                    inDataFormat = DataFormat.FP16
                case _:
                    raise NotImplementedError()
        else:
            assert inDataFormat == outDataFormat

            match inDataFormat:
                case DataFormat.INT8:
                    # INT8 is either uint8_t or 8 bit sign-magnitude, and becomes "Integer 8",
                    # which is then overlaid onto FP16
                    int8MeansUnsigned = (
                        self.getConfigValue(
                            stateID, "ALU_FORMAT_SPEC_REG0_SrcBUnsigned"
                        )
                        if self.unpacker_id == 1
                        else self.getConfigValue(
                            stateID, "ALU_FORMAT_SPEC_REG0_SrcAUnsigned"
                        )
                    )
                    sign = 0 if int8MeansUnsigned else raw_datum & 0x80
                    raw_datum = raw_datum - sign
                    # ``* (raw_datum != 0)`` is the "if the magnitude is
                    # non-zero" guard written as arithmetic, so a whole block
                    # converts in one pass; for a single datum it is the same
                    # int (see DataFormatConversions on branch-free form).
                    raw_datum = raw_datum | (16 << 10) * (raw_datum != 0)
                    raw_datum = raw_datum | (sign << 8)
                    inDataFormat = DataFormat.FP16
                case DataFormat.TF32:
                    if unpackToDst:
                        #  When unpacking to Dst, TF32 means FP32
                        return DataFormatConversions.FP32ToDstFormatFP32(raw_datum)
                    else:
                        # Otherwise, TF32 is not valid as InDataFormat, but software can instead
                        # specify InDataFormat == FP32 and OutDataFormat == TF32
                        raise ValueError()

        # Now rearrange bits to the format expected by Dst or by SrcA / SrcB
        match inDataFormat:
            case DataFormat.UINT16:
                if unpackToDst:
                    return (raw_datum & 0xFF00) << 3
                else:
                    return raw_datum & 0xFF
            case DataFormat.INT32 | DataFormat.FP32:
                if unpackToDst:
                    # INT32 is stored verbatim in Dst — getDst32b/setDst32b split
                    # it across the hi/lo 16-bit planes and the packer reads it
                    # back unchanged (packer.py's INT32 case returns raw_datum).
                    # Only FP32 gets rearranged into the Dst float storage format;
                    # applying that rearrangement to an integer scrambles its high
                    # 16 bits (low 16 survive), corrupting every 32-bit int datum.
                    if inDataFormat == DataFormat.INT32:
                        return raw_datum
                    return DataFormatConversions.FP32ToDstFormatFP32(raw_datum)
                # Unpacking a 32-bit datum to the 19-bit SrcA/SrcB registers
                # narrows it to the Src TF32 storage format (the widest Src
                # representation), the same way an explicit FP32->TF32 unpack to
                # Src is handled at the top of this function. tt-metal's LLK
                # emits such an unpack (e.g. clearing SrcA to zero) ahead of
                # SFPU int32/fp32 kernels, which then compute out of Dst.
                return DataFormatConversions.TF32ToSrcFormatTF32(raw_datum >> 13)
            case DataFormat.BF16:
                if unpackToDst:
                    return DataFormatConversions.BF16ToDstFormatBF16(raw_datum)
                else:
                    return DataFormatConversions.BF16ToSrcBF16(raw_datum)
            case DataFormat.FP16:
                if unpackToDst:
                    return DataFormatConversions.FP16ToDstFormatFP16(raw_datum)
                else:
                    return DataFormatConversions.FP16ToSrcFP16(raw_datum)
            case _:
                raise NotImplementedError()

    def handle_regular(self, instruction_info, issue_thread, instr_args):
        # An UNPACR reads all of its configuration -- state ID, context
        # selection, input/output addresses, formats -- before it starts moving
        # datums, and only then waits for the Src bank it writes into. That
        # order is load-bearing, not incidental: the issuing thread carries on
        # while the unpack is in flight, and the LLK's matmul unpack flips
        # UNPACK_MISC_CFG_CfgContextOffset (and bumps the SEC0 base address via
        # the MOP) immediately after issuing its UNPACRs. Reading the
        # configuration afresh when a stalled unpack finally runs would pick up
        # the *next* matmul's context and base address, so latch it here and
        # reuse it for as long as the unpack is blocked.
        fresh = self.pending_unpack is None
        if fresh:
            self.pending_unpack = self.read_unpack_state(issue_thread, instr_args)

        src = self.backend.getSrcA if self.unpacker_id == 0 else self.backend.getSrcB
        if src(self.srcBank).getAllowedClient() != SrcRegister.SrcClient.Unpackers:
            self.blocked = True
            self.blocked_wait_bank = self.srcBank
            self.repeat_instruction = (instruction_info, issue_thread)
            if fresh:
                # The address phase happens up front, before the wait for the
                # Src bank ("spends at least two cycles calculating the
                # initial input address" — then "the primary bottleneck being
                # the fetching of bytes from L1"). Charge it now, once; the
                # blocked re-runs charge nothing, because a blocked unit is
                # waiting, not busy.
                self._pending_occupancy = self._address_phase_cycles()
                self._pending_thread_block = issue_thread
                self._pending_occupancy_thread = issue_thread
            return

        self.blocked = False
        self.blocked_wait_bank = None
        self.repeat_instruction = None
        state = self.pending_unpack
        self.pending_unpack = None
        if self.cost_model is not None:
            # The data phase, from the size and throttle config latched at
            # decode; serial with the address phase, which is only still
            # chargeable here when the unpack never blocked (``fresh``) — a
            # resumed unpack paid it at issue.
            address = self._address_phase_cycles() if fresh else 0
            self._pending_occupancy = address + (state["data_phase_cycles"] or 0)
            self._pending_occupancy_thread = issue_thread
            if fresh and address:
                self._pending_thread_block = issue_thread
        self.perform_unpack_state(issue_thread, state)

    def read_unpack_state(self, issue_thread, instr_args):
        """Read everything an UNPACR needs, before it waits on its Src bank."""
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rowSearch = instr_args["RowSearch"]
        useContextCounter = instr_args["AutoIncContextID"]
        allDatumsAreZero = instr_args["ZeroWrite2"]
        flipSrc = instr_args["SetDatValid"]
        multiContextMode = instr_args["OvrdThreadId"]
        contextADC = instr_args["AddrCntContextId"]
        contextNumber = instr_args["CfgContextId"]

        addrMode = instr_args["AddrMode"]
        ch0ZInc = get_bits(addrMode, 0, 1)
        ch0YInc = get_bits(addrMode, 2, 3)
        ch1ZInc = get_bits(addrMode, 4, 5)
        ch1YInc = get_bits(addrMode, 6, 7)

        # Determine initial input address(es) and input datum count
        whichContext, whichADC = self.read_unpack_configuration(
            issue_thread, multiContextMode, useContextCounter, contextNumber, contextADC
        )

        # Determine initial output address
        (
            inAddr_Datums,
            datumSizeBytes,
            inputNumDatums,
            inAddr_Deltas,
            inAddr_Exponents,
            rowStride,
            unpackRowWidth,
            discontiguousInputRows,
            isUncompressed,
            inDataFormat,
        ) = self.generate_input_addresses_and_sizes(
            issue_thread, stateID, multiContextMode, whichContext, whichADC, rowSearch
        )

        outAddr, outDataFormat, unpackToDst, transpose = self.generate_output_address(
            stateID, issue_thread, multiContextMode, whichContext
        )

        upsampleRate = self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Upsample_rate"
        )
        upsampleZeroes = (1 << upsampleRate) - 1
        upsampleInterleave = self.getConfigValue(
            stateID,
            "THCON_SEC" + str(self.unpacker_id) + "_REG2_Upsample_and_interleave",
        )
        if discontiguousInputRows or self.unpacker_id == 1:
            colShift = 0
        else:
            colShift = self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG2_Shift_amount_cntx"
                + str(whichContext & 3),
            )

        # Check that various settings are compatible with each other:
        self.check_unpacker_settings(
            transpose,
            discontiguousInputRows,
            inAddr_Datums,
            upsampleZeroes,
            isUncompressed,
            unpackToDst,
            colShift,
        )

        # ... and that the ones which shape the datum walk are ones we model.
        # Checked here, at decode, so an unmodelled unpack fails before it moves
        # a single datum (and before it can block on a Src bank).
        self.check_modelled_settings(
            rowStride,
            unpackRowWidth,
            datumSizeBytes,
            discontiguousInputRows,
            upsampleRate,
            upsampleZeroes,
            upsampleInterleave,
            colShift,
            outAddr,
        )

        # The data-phase charge, priced while the throttle config that governs
        # this UNPACR is in hand ("computed at issue"): transfer bytes over the
        # documented fetch rate. ``None`` — charge nothing — with the model off
        # or where the rate selection has no opinion (e.g. an untabulated
        # throttle mode).
        data_phase_cycles = None
        model = self.cost_model
        if model is not None:
            throttle_mode = self.getConfigValue(
                stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Throttle_mode"
            )
            default_overridden = True
            if self.backend.blackhole:
                # Blackhole-only bit, and the doc indexes THCON_SEC[0] for
                # both unpackers. Clear (the tt-metal default) means the
                # config mode is ignored in favour of x4/x8.
                default_overridden = bool(
                    self.getConfigValue(
                        stateID, "THCON_SEC0_REG1_ovrd_default_throttle_mode"
                    )
                )
            data_phase_cycles = model.unpack_data_phase_cycles(
                inputNumDatums * datumSizeBytes,
                throttle_mode,
                datumSizeBytes,
                tileize=bool(discontiguousInputRows),
                default_throttle_overridden=default_overridden,
            )

        # "Update counters in preparation for next instruction" -- the last
        # block of UNPACR_Regular.md's functional model, less the Src hand-over
        # (which ``flip_src_banks`` still does at the end of the transfer).
        # These belong to the *address phase*, with the input address generator
        # that consumes them and with the configuration read above, for the
        # reason the docstring of ``handle_regular`` gives for latching that
        # configuration: the issuing thread carries on while the unpack is in
        # flight and reprograms this very state for the next instruction. The
        # ADCs are the write side of exactly that hazard --
        # ``_llk_unpack_reduce_`` opens each call with a ``SETADCZW`` that
        # resets the Z counter, and an UNPACR still waiting on a Src bank when
        # that lands used to apply its ``Ch0.Z += Ch0ZInc`` *afterwards*, so
        # the next reduction read every face one row late. The counters are
        # also what the doc says the update is for ("in preparation for next
        # instruction"), and the next instruction cannot start before this
        # unpacker's address phase is over, so nothing inside the unit can tell
        # the difference.
        self.increment_counter(
            stateID, issue_thread, whichContext, multiContextMode, useContextCounter
        )
        self.update_ADC(issue_thread, whichADC, ch0YInc, ch0ZInc, ch1YInc, ch1ZInc)

        # Exactly what ``perform_unpack_state`` (the data phase) still needs,
        # and nothing else. The context/ADC selectors and the four address
        # increments used to be carried here too, for the counter update that
        # now runs above in the address phase; nothing read them afterwards, and
        # a latched copy of state the address phase has already consumed is an
        # invitation to consume it a second time.
        return {
            "stateID": stateID,
            "data_phase_cycles": data_phase_cycles,
            "allDatumsAreZero": allDatumsAreZero,
            "flipSrc": flipSrc,
            "inAddr_Datums": inAddr_Datums,
            "datumSizeBytes": datumSizeBytes,
            "inputNumDatums": inputNumDatums,
            "rowStride": rowStride,
            "unpackRowWidth": unpackRowWidth,
            "inDataFormat": inDataFormat,
            "outAddr": outAddr,
            "outDataFormat": outDataFormat,
            "unpackToDst": unpackToDst,
            "transpose": transpose,
        }

    def perform_unpack_state(self, issue_thread, state):
        """Move the datums, using the configuration latched at decode."""
        # Main unpack loop
        self.perform_unpack(
            state["stateID"],
            issue_thread,
            state["inputNumDatums"],
            state["inAddr_Datums"],
            state["outAddr"],
            state["datumSizeBytes"],
            state["inDataFormat"],
            state["outDataFormat"],
            state["unpackToDst"],
            state["transpose"],
            state["allDatumsAreZero"],
            state["rowStride"],
            state["unpackRowWidth"],
        )

        # The context counter and the ADCs were advanced in the address phase,
        # by ``read_unpack_state``; see the comment there.

        # Flip src banks
        self.flip_src_banks(state["flipSrc"], issue_thread, state["outDataFormat"])
