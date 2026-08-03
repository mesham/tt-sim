from tt_sim.memory.memory import VisibleMemory
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.register.register import Register, RegisterAccessMode
from tt_sim.pe.register.register_file import RegisterFile
from tt_sim.pe.rv.isa.i_isa import RV_I_ISA
from tt_sim.pe.rv.isa.m_isa import RV_M_ISA
from tt_sim.pe.rv.isa.tt_isa import RV_TT_ISA
from tt_sim.trace import EventCategory, InstrEvent, get_bus
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

# The 32 integer registers occupy indices 0-31, with pc/nextpc at 32/33. When a
# core is built with a floating-point unit, the 32 f-registers occupy 34-65 and
# fcsr is 66 — kept as module constants so the F/Zfh ISAs can address them.
FP_REGISTER_BASE = 34
FCSR_INDEX = FP_REGISTER_BASE + 32  # 66

REGISTER_NAME_MAPPING = {
    "x0": 0,
    "x1": 1,
    "x2": 2,
    "x3": 3,
    "x4": 4,
    "x5": 5,
    "x6": 6,
    "x7": 7,
    "x8": 8,
    "x9": 9,
    "x10": 10,
    "x11": 11,
    "x12": 12,
    "x13": 13,
    "x14": 14,
    "x15": 15,
    "x16": 16,
    "x17": 17,
    "x18": 18,
    "x19": 19,
    "x20": 20,
    "x21": 21,
    "x22": 22,
    "x23": 23,
    "x24": 24,
    "x25": 25,
    "x26": 26,
    "x27": 27,
    "x28": 28,
    "x29": 29,
    "x30": 30,
    "x31": 31,
    "zero": 0,
    "ra": 1,
    "sp": 2,
    "gp": 3,
    "tp": 4,
    "t0": 5,
    "t1": 6,
    "t2": 7,
    "s0": 8,
    "s1": 9,
    "a0": 10,
    "a1": 11,
    "a2": 12,
    "a3": 13,
    "a4": 14,
    "a5": 15,
    "a6": 16,
    "a7": 17,
    "s2": 18,
    "s3": 19,
    "s4": 20,
    "s5": 21,
    "s6": 22,
    "s7": 23,
    "s8": 24,
    "s9": 25,
    "s10": 26,
    "s11": 27,
    "t3": 28,
    "t4": 29,
    "t5": 30,
    "t6": 31,
    "fp": 8,
    "pc": 32,
    "nextpc": 33,
    # Floating-point register file (F/Zfh, FLEN=32). Present only on cores built
    # with fp_registers=True (Blackhole baby cores); indices are stable so the
    # FP ISAs address them as FP_REGISTER_BASE + n. fcsr follows the 32 f-regs.
    **{f"f{i}": FP_REGISTER_BASE + i for i in range(32)},
    "fcsr": FCSR_INDEX,
}


class RV32I(ProcessingElement):
    def __init__(
        self,
        start_address,
        memory_spaces,
        extensions=None,
        unknown_instr_is_error=False,
        snoop=False,
        core_id=0,
        fp_registers=False,
    ):
        if extensions is None:
            extensions = []
        self.isas = [RV_I_ISA] + extensions
        assert isinstance(memory_spaces, list)
        self.start_address = start_address
        self.active = False
        self.unknown_instructions = 0
        self.snoop = snoop
        self.core_id = core_id
        self.core_label = str(core_id)
        self.unit_id: tuple | None = None

        # 32 registers plus the PC
        registers = []
        # Register 0 is read only and hardcoded to zero
        registers.append(Register(4, conv_to_bytes(0), RegisterAccessMode.R, False))
        for i in range(33):
            registers.append(Register(4))

        # Floating-point register file (F/Zfh): 32 f-registers + fcsr, appended
        # at FP_REGISTER_BASE so the FP ISAs find them. Only cores with an FP
        # unit (Blackhole baby cores) allocate these; RV32IM cores never do.
        if fp_registers:
            assert len(registers) == FP_REGISTER_BASE
            for i in range(32):
                registers.append(Register(4))
            registers.append(Register(4))  # fcsr

        # A ``tt_sim.pe.rv.cost.RiscvCostState`` once a core opts into the
        # cycle-cost tables *and* ``TT_SIM_COST_MODEL`` is set; ``None``
        # otherwise, which is the default. ``clock_tick`` reads it once per
        # instruction, so the switched-off cost is one attribute read and one
        # predicted branch on the hottest path in the simulator. Only the baby
        # cores set it (they are the ones with an architecture and a memory
        # map to classify addresses against); see ``tt_sim/pe/rv/cost.py``.
        self.rv_cost = None
        self.register_file = RegisterFile(registers, REGISTER_NAME_MAPPING)
        # PC and next-PC are touched several times per simulated cycle; resolve
        # them once instead of going through the name map every time.
        self.pc_register = self.register_file["pc"]
        self.nextpc_register = self.register_file["nextpc"]
        # The event bus is a process-wide singleton that is never replaced, so
        # hold it directly rather than calling get_bus() per instruction.
        self.bus = get_bus()
        self.unknown_instr_is_error = unknown_instr_is_error

        # Now determine the visible memory for the core, this will either be a combination of
        # global device memory and local PE memory, or one of these if the other is not supplied
        if len(memory_spaces) == 0:
            raise Exception(
                "An RV32 core must have access to at-least one memory space"
            )
        elif len(memory_spaces) == 1:
            self.visible_memory = memory_spaces[0]
        else:
            self.visible_memory = VisibleMemory.merge(*memory_spaces)

    def print_snoop(self, pc, nextpc, actioned):
        addr = pc.read_uint()
        instr = self.visible_memory.read(addr, 4)

        opcode_bin = RV_I_ISA.get_bits(instr, 0, 6)
        opcode_bin.reverse()

        print(opcode_bin)

    def clock_tick(self, cycle_num):
        if not self.active:
            return

        register_file = self.register_file
        pc = self.pc_register
        nextpc = self.nextpc_register
        pc_val = pc.read_uint()
        nextpc.write(conv_to_bytes(pc_val + 4))
        self.visible_memory.caller_context = (self.unit_id, self.core_label, pc_val)

        # Fetch once per cycle and hand the word to every ISA in the list —
        # each one used to re-read it from memory, which cost a full memory-map
        # traversal per ISA tried.
        instr = int.from_bytes(self.visible_memory.read(pc_val, 4), "little")

        # Memory-stall back-pressure (ROADMAP section I). Consulted before any
        # tracing or snoop output is produced, because a stalled cycle must
        # look like a cycle in which this core did nothing at all: no
        # instruction retires, the PC does not advance, and the same word is
        # re-offered on a later cycle. ``rv_cost`` is ``None`` unless
        # ``TT_SIM_COST_MODEL`` is set, so this is one attribute read otherwise.
        cost = self.rv_cost
        if cost is not None and not cost.can_issue(instr, cycle_num, register_file):
            return

        # The InstrEvent below reports which GPR the instruction wrote, which
        # needs a recording hook on every Register.write — the hottest call in
        # the simulator. Install it only while a subscriber is listening, and
        # clear the last-write record so the event sees this instruction alone.
        trace_instr = self.unit_id is not None and self.bus.is_enabled(
            EventCategory.INSTR
        )
        if trace_instr != register_file.write_recording:
            register_file.set_write_recording(trace_instr)
        if trace_instr:
            register_file.clear_write_record()

        if self.snoop:
            print(f"[{self.core_id}-> {cycle_num}][{hex(pc_val)}] ", end="")

        actioned = False
        pe_stall = False
        for isa in self.isas:
            actioned = isa.run(register_file, self.visible_memory, self.snoop, instr)
            pe_stall = actioned == ProcessingElement.PEStall
            if actioned or pe_stall:
                break

        if not actioned and not pe_stall:
            self.unknown_instructions += 1
            if self.unknown_instr_is_error:
                raise Exception("Unknown instruction")
            if self.snoop:
                print(f"unknown # {instr:032b}", end="")

        if self.snoop:
            if pe_stall:
                print("[Stalled]")
            print("")

        if trace_instr:
            # Capture the architectural register write (x1-x31). Index 0
            # (x0) is read-only and not logged by Spike-style commitlogs;
            # indices 32+ are PC/nextpc, not architectural state.
            wi = register_file.last_write_idx
            if 1 <= wi <= 31:
                reg_idx = wi
                reg_val = conv_to_uint32(register_file.last_write_value)
            else:
                reg_idx = -1
                reg_val = 0
            # How long the cost model held this instruction before it could
            # issue, drained here rather than counted on the issue path so the
            # non-tracing interpreter is unchanged. Always (0, "") with
            # TT_SIM_COST_MODEL unset, because nothing stalls then.
            if cost is None:
                stall_cycles, stall_reason = 0, ""
            else:
                stall_cycles, stall_reason = cost.take_pending_stall()
            self.bus.publish(
                InstrEvent(
                    cycle=cycle_num,
                    unit_id=self.unit_id,
                    pc=pc_val,
                    instruction=instr,
                    stalled=pe_stall,
                    reg_write_idx=reg_idx,
                    reg_write_value=reg_val,
                    stall_cycles=stall_cycles,
                    stall_reason=stall_reason,
                )
            )

        if not pe_stall:
            pc.write(nextpc.read())

    def reset(self):
        self.stop()
        self.start()

    def getRegisterFile(self):
        return self.register_file

    def get_start_address(self):
        return self.start_address

    def initialise_core(self):
        pc = self.register_file["pc"]
        pc.write(conv_to_bytes(self.get_start_address()))

        self.unknown_instructions = 0
        if self.rv_cost is not None:
            self.rv_cost.reset()

    def start(self):
        self.initialise_core()
        self.active = True

    def stop(self):
        self.active = False


class RV32IM(RV32I):
    def __init__(
        self,
        start_address,
        memory_spaces,
        extensions=None,
        unknown_instr_is_error=False,
        snoop=False,
        core_id=0,
        fp_registers=False,
    ):
        if extensions is None:
            extensions = []
        if RV_M_ISA not in extensions:
            extensions.append(RV_M_ISA)
        super().__init__(
            start_address,
            memory_spaces,
            extensions,
            unknown_instr_is_error,
            snoop,
            core_id,
            fp_registers,
        )


class RV32IM_TT(RV32IM):
    def __init__(
        self,
        start_address,
        memory_spaces,
        extensions=None,
        unknown_instr_is_error=False,
        snoop=False,
        core_id=0,
        fp_registers=False,
    ):
        if extensions is None:
            extensions = []
        if RV_TT_ISA not in extensions:
            extensions.append(RV_TT_ISA)
        super().__init__(
            start_address,
            memory_spaces,
            extensions,
            unknown_instr_is_error,
            snoop,
            core_id,
            fp_registers,
        )
