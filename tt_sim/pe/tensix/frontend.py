from abc import ABC

from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.util import DiagnosticsSettings, TensixInstructionDecoder
from tt_sim.util.bits import extract_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_uint32


class TensixFrontend(MemMapable):
    def __init__(self, thread_id, backend, diags_settings=None):
        self.thread_id = thread_id
        self.backend = backend
        self.mop_instruction_fifo = []
        self.replay_instruction_fifo = []
        self.wait_gate_instruction_fifo = []
        self.mop_expander = TensixMOPExpander(self)
        self.replay_expander = TensixReplayExpander(self)
        self.wait_gate = WaitGate(self, backend)
        self.diags_settings = (
            diags_settings if diags_settings is not None else DiagnosticsSettings()
        )

    def getClocks(self):
        return [self.mop_expander, self.replay_expander, self.wait_gate]

    def getDiagnosticSettings(self):
        return self.diags_settings

    def hasInflightInstructions(self):
        return (
            len(self.mop_instruction_fifo) > 0
            or len(self.replay_instruction_fifo)
            or len(self.wait_gate_instruction_fifo)
        )

    def MOPExpanderDoneCheck(self):
        return len(self.mop_instruction_fifo) > 0

    def pop_mop_instruction(self):
        if len(self.mop_instruction_fifo) > 0:
            return self.mop_instruction_fifo.pop(0)
        else:
            return None

    def push_mop_instruction(self, instruction):
        return self.mop_instruction_fifo.append(instruction)

    def pop_replay_instruction(self):
        if len(self.replay_instruction_fifo) > 0:
            return self.replay_instruction_fifo.pop(0)
        else:
            return None

    def push_replay_instruction(self, instruction):
        return self.replay_instruction_fifo.append(instruction)

    def pop_wait_gate_instruction(self):
        if len(self.wait_gate_instruction_fifo) > 0:
            return self.wait_gate_instruction_fifo.pop(0)
        else:
            return None

    def inspect_wait_gate_instruction(self):
        if len(self.wait_gate_instruction_fifo) > 0:
            return self.wait_gate_instruction_fifo[0]
        else:
            return None

    def push_wait_gate_instruction(self, instruction):
        return self.wait_gate_instruction_fifo.append(instruction)

    def getMOPExpander(self):
        return self.mop_expander

    def read(self, addr, size):
        raise NotImplementedError(
            f"Reading at {hex(addr)} from tensix thread {self.thread_id}"
        )

    def write(self, addr, value, size=None):
        instruction = conv_to_uint32(value)
        if TensixInstructionDecoder.isInstructionRecognised(instruction):
            self.push_mop_instruction(instruction)
        else:
            opcode = extract_bits(instruction, 8, 24)
            raise NotImplementedError(
                f"Unknown op code issued {hex(opcode)} on tensix thread {self.thread_id}"
            )

    def getSize(self):
        return 0xFFFF


class TensixFrontendUnit(Clockable, ABC):
    def __init__(self, frontend):
        self.frontend = frontend

    def getDiagnosticSettings(self):
        return self.frontend.getDiagnosticSettings()


class WaitGate(TensixFrontendUnit):
    # The math instructions to check allowedclient for, first element is
    # list of applicable instructions that need to check srcA, second element
    # is list of applicable instructions that need to check srcB
    MATH_ALLOWED_CLIENT_INSTRUCTIONS = [
        (
            "MVMUL",
            "DOTPV",
            "GAPOOL",
            "GMPOOL",
            "ELWMUL",
            "ELWADD",
            "ELWSUB",
            "MOVA2D",
            "MOVDBGA2D",
            "SHIFTXA",
        ),
        (
            "MVMUL",
            "DOTPV",
            "GAPOOL",
            "GMPOOL",
            "ELWMUL",
            "ELWADD",
            "ELWSUB",
            "MOVB2D",
            "MOVB2A",
            "SHIFTXB",
            "TRANSPSRCB",
        ),
    ]

    class LatchedInstruction:
        BLOCKED_INSTRUCTION_TYPES = [
            (
                "TDMA",
                "XMOV",
                "THCON",
                "PACK",
                "UNPACK",
            ),
            ("SYNC",),
            ("PACK",),
            ("UNPACK",),
            ("XMOV",),
            ("THCON",),
            ("MATH",),
            ("CFG",),
            ("SFPU",),
        ]

        def __init__(self, opcode, condition_mask, block_mask, semaphore_mask=None):
            self.opcode = opcode
            self.condition_mask = condition_mask
            self.block_mask = block_mask
            self.semaphore_mask = semaphore_mask

        def doesInstructionMatchBlockMask(self, instruction_info):
            tgt_backend_unit = instruction_info["ex_resource"]
            for bit_idx in range(9):
                do_check = get_nth_bit(self.block_mask, bit_idx)
                if do_check:
                    if (
                        tgt_backend_unit
                        in WaitGate.LatchedInstruction.BLOCKED_INSTRUCTION_TYPES[
                            bit_idx
                        ]
                    ):
                        return True
            return False

        def getConditionCheck(self, idx):
            return get_nth_bit(self.condition_mask, idx)

        def isSemaphoreMode(self):
            if self.opcode == "SEMWAIT":
                assert self.semaphore_mask > 0
                return True
            return False  # is STALLWAIT

        def getSemaphoresToCheck(self):
            sem_check = []
            for i in range(8):
                if get_nth_bit(self.semaphore_mask, i):
                    sem_check.append(i)
            return sem_check

    def __init__(self, frontend, backend):
        super().__init__(frontend)
        self.mutex_stall = False
        self.backend_enforced_stall = False
        self.latchedWaitInstruction = None
        self.latch_wait = False
        self.backend = backend

    def setBackendEnforcedStall(self):
        self.backend_enforced_stall = True

    def clearBackendEnforcedStall(self):
        self.backend_enforced_stall = False

    def setLatchedWaitInstruction(
        self, opcode, condition_mask, block_mask, semaphore_mask=None
    ):
        self.latchedWaitInstruction = WaitGate.LatchedInstruction(
            opcode, condition_mask, block_mask, semaphore_mask
        )

    def check_for_wait_condition_met(self):
        assert self.latchedWaitInstruction is not None
        if self.latchedWaitInstruction.isSemaphoreMode():
            sem_checks = self.latchedWaitInstruction.getSemaphoresToCheck()
            for sem in sem_checks:
                semaphore = self.backend.getSyncUnit().getSemaphore(sem)
                if (
                    self.latchedWaitInstruction.getConditionCheck(0)
                    and semaphore.value == 0
                ):
                    return False
                if (
                    self.latchedWaitInstruction.getConditionCheck(1)
                    and semaphore.value >= semaphore.max
                ):
                    return False
                return True
        else:
            for idx in range(15):
                if self.latchedWaitInstruction.getConditionCheck(idx):
                    if not self.check_for_semwait_condition_match(idx):
                        return False
            return True

    def check_for_semwait_condition_match(self, cond_idx):
        # Some of these conditions do not apply, as per
        # https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/STALLWAIT.md
        match cond_idx:
            case 0:
                return True
            case 1:
                return not self.backend.unpacker_units[
                    0
                ].hasInflightInstructionsFromThread(self.frontend.thread_id)
            case 2:
                return not self.backend.unpacker_units[
                    0
                ].hasInflightInstructionsFromThread(self.frontend.thread_id)
            case 3 | 4 | 5 | 6:
                return not self.backend.packer_unit.hasInflightInstructionsFromThread(
                    self.frontend.thread_id
                )
            case 7:
                return not self.backend.matrix_unit.hasInflightInstructionsFromThread(
                    self.frontend.thread_id
                )
            case 8:
                return (
                    self.backend.getSrcA(
                        self.backend.unpacker_units[0].srcBank
                    ).allowedClient
                    == SrcRegister.SrcClient.Unpackers
                )
            case 9:
                return (
                    self.backend.getSrcB(
                        self.backend.unpacker_units[1].srcBank
                    ).allowedClient
                    == SrcRegister.SrcClient.Unpackers
                )
            case 10:
                return (
                    self.backend.getSrcA(self.backend.matrix_unit.srcBank).allowedClient
                    == SrcRegister.SrcClient.MatrixUnit
                )
            case 11:
                return (
                    self.backend.getSrcB(self.backend.matrix_unit.srcBank).allowedClient
                    == SrcRegister.SrcClient.MatrixUnit
                )
            case 12:
                return not self.backend.mover_unit.checkForOutstandingInstructions()
            case 13:
                return True
            case 14:
                return not self.backend.vector_unit.hasInflightInstructionsFromThread(
                    self.frontend.thread_id
                )
            case _:
                return True

    def clock_tick(self, cycle_num):
        if not self.mutex_stall and not self.backend_enforced_stall:
            instruction = self.frontend.inspect_wait_gate_instruction()
            if not self.latch_wait and self.latchedWaitInstruction is not None:
                if instruction is not None:
                    instruction_info = TensixInstructionDecoder.getInstructionInfo(
                        instruction
                    )
                    self.latch_wait = (
                        self.latchedWaitInstruction.doesInstructionMatchBlockMask(
                            instruction_info
                        )
                    )
            elif self.latch_wait:
                condition_met = self.check_for_wait_condition_met()
                if condition_met:
                    self.latch_wait = False
                    self.latchedWaitInstruction = None
                    return  # One cycle latency here

            if not self.latch_wait:
                if instruction is not None:
                    instruction_info = TensixInstructionDecoder.getInstructionInfo(
                        instruction
                    )
                    if instruction_info["name"] == "ATGETM":
                        # Stall due to mutex, but still process this instruction to ensure
                        # it reaches the sync unit
                        self.mutex_stall = True
                    if instruction_info["ex_resource"] == "MATH":
                        # For FPU instructions need to ensure that srcA and srcB
                        # being consumed has allowed client of MatrixUnit
                        if self.checkIfFPUInstructionShouldStall(
                            instruction_info["name"]
                        ):
                            return
                    instruction_accepted = self.frontend.backend.issueInstruction(
                        instruction, self.frontend.thread_id
                    )
                    if instruction_accepted:
                        # If the instruction was accepted then remove it,
                        # otherwise retry next cycle
                        if self.getDiagnosticSettings().reportIssuedInstructions():
                            print(
                                f"Issued {instruction_info['name']} to {instruction_info['ex_resource']} "
                                f"from thread {self.frontend.thread_id}"
                            )
                        self.frontend.pop_wait_gate_instruction()

    def checkIfFPUInstructionShouldStall(self, opcode):
        if opcode in WaitGate.MATH_ALLOWED_CLIENT_INSTRUCTIONS[0]:
            if (
                self.backend.getMatrixUnit().getSrcA().allowedClient
                == SrcRegister.SrcClient.Unpackers
            ):
                return True

        if opcode in WaitGate.MATH_ALLOWED_CLIENT_INSTRUCTIONS[1]:
            if (
                self.backend.getMatrixUnit().getSrcB().allowedClient
                == SrcRegister.SrcClient.Unpackers
            ):
                return True

        return False

    def informMutexAcquired(self):
        self.mutex_stall = False


class TensixReplayExpander(TensixFrontendUnit):
    def __init__(self, frontend):
        self.replay_buffer = [0] * 32
        self.append_instruction_to_buffer = False
        self.exec_while_load = False
        self.replay_len = 0
        self.replay_start_idx = 0
        self.replay_idx = 0
        super().__init__(frontend)

    def clock_tick(self, cycle_num):
        instruction = self.frontend.pop_replay_instruction()
        if instruction is not None:
            if self.append_instruction_to_buffer:
                if self.replay_idx < self.replay_len:
                    self.replay_buffer[
                        (self.replay_start_idx + self.replay_idx) % 32
                    ] = instruction
                    self.replay_idx += 1
                    if self.exec_while_load:
                        self.frontend.push_wait_gate_instruction(instruction)
                else:
                    self.append_instruction_to_buffer = False

            if not self.append_instruction_to_buffer:
                instruction_info = TensixInstructionDecoder.getInstructionInfo(
                    instruction
                )
                if instruction_info["name"] == "REPLAY":
                    instr_args = instruction_info["instr_args"]
                    if instr_args["load_mode"] == 0:
                        index = instr_args["start_idx"]
                        for i in range(instr_args["len"] or 64):
                            self.frontend.push_wait_gate_instruction(
                                self.replay_buffer[(index + i) % 32]
                            )
                    else:
                        self.replay_start_idx = instr_args["start_idx"]
                        self.exec_while_load = instr_args["execute_while_loading"]
                        self.replay_len = instr_args["len"] or 64
                        self.replay_idx = 0
                        self.append_instruction_to_buffer = True
                else:
                    self.frontend.push_wait_gate_instruction(instruction)


class TensixMOPExpander(TensixFrontendUnit, MemMapable):
    def __init__(self, frontend):
        self.mop_cfg = [0] * 9
        self.mask_hi = 0
        super().__init__(frontend)

    def read(self, address, size):
        raise NotImplementedError("Can not read from Tensix MOP expander configuration")

    def write(self, addr, value, size=None):
        idx = int(addr / 4)
        assert idx < 9

        self.mop_cfg[idx] = conv_to_uint32(value)

    def clock_tick(self, cycle_num):
        instruction = self.frontend.pop_mop_instruction()
        if instruction is not None:
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            if instruction_info["name"] == "MOP":
                instr_args = instruction_info["instr_args"]
                if instr_args["mop_type"] == 0:
                    for new_instr in self.expand_template_zero(
                        (self.mask_hi << 16) + instr_args["zmask_lo16"],
                        instr_args["loop_count"],
                    ):
                        self.frontend.push_replay_instruction(new_instr)
                else:
                    for new_instr in self.expand_template_one():
                        self.frontend.push_replay_instruction(new_instr)
            elif instruction_info["name"] == "MOP_CFG":
                self.mask_hi = instruction_info["instr_args"]["zmask_hi16"]
            else:
                self.frontend.push_replay_instruction(instruction)

    def expand_template_zero(self, mask, count1):
        flags = self.mop_cfg[1]
        insnb = self.mop_cfg[2]
        insna0 = self.mop_cfg[3]
        insna1 = self.mop_cfg[4]
        insna2 = self.mop_cfg[5]
        insna3 = self.mop_cfg[6]
        skipa0 = self.mop_cfg[7]
        skipb = self.mop_cfg[8]
        hasb = flags & 1
        hasa123 = flags & 2
        for i in range(count1 + 1):
            if mask & 1 == 0:
                yield insna0
                if hasa123:
                    yield insna1
                    yield insna2
                    yield insna3
                if hasb:
                    yield insnb
            else:
                yield skipa0
                if hasb:
                    yield skipb
            mask >>= 1

    def expand_template_one(self):
        outercount = self.mop_cfg[0] & 127
        innercount = self.mop_cfg[1] & 127
        startop = self.mop_cfg[2]
        endop0 = self.mop_cfg[3]
        endop1 = self.mop_cfg[4]
        loopop = self.mop_cfg[5]
        loopop1 = self.mop_cfg[6]
        loop0last = self.mop_cfg[
            7
        ]  # overrides loopop or loopop1 on last iteration of inner loop, if also last iteration of outer loop
        loop1last = self.mop_cfg[
            8
        ]  # overrides loopop or loopop1 on last iteration of inner loop, if not last iteration of outer loop

        if self.isnop(loopop1):
            loopopflip = 0
        else:
            loopopflip = (
                loopop ^ loopop1
            )  # inner loop will alternate between the two instructions and will
            innercount *= (
                2  # execute for twice as many iterations. it is expressed like this
            )
            # because loop0last / loop1last override the last iteration.
        if (
            outercount == 1
            and self.isnop(startop)
            and innercount == 0
            and not self.isnop(endop0)
        ):
            outercount += 128  # hardware bug
        for j in range(outercount):
            if not self.isnop(startop):
                yield startop
            for i in range(innercount):
                if i != innercount - 1:
                    yield loopop
                elif j != outercount - 1:
                    yield loop1last
                else:
                    yield loop0last
                loopop ^= loopopflip
            if not self.isnop(endop0):
                yield endop0
                if not self.isnop(endop1):
                    yield endop1

    def isnop(self, instruction):
        # this only recognises plain nop (opcode 0x02), not dmanop nor sfpnop.
        opcode = extract_bits(instruction, 8, 24)
        return opcode == 0x02

    def getSize(self):
        return 0x23
