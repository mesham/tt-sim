from abc import ABC

from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.bits import extract_bits
from tt_sim.util.conversion import conv_to_uint32


class TensixFrontend(MemMapable):
    def __init__(self, thread_id, tensix_instruction_decoder, backend):
        self.thread_id = thread_id
        self.backend = backend
        self.mop_instruction_fifo = []
        self.replay_instruction_fifo = []
        self.wait_gate_instruction_fifo = []
        self.tensix_instruction_decoder = tensix_instruction_decoder
        self.mop_expander = TensixMOPExpander(self, tensix_instruction_decoder)
        self.replay_expander = TensixReplayExpander(self, tensix_instruction_decoder)
        self.wait_gate = WaitGate(self, tensix_instruction_decoder)

    def getClocks(self):
        return [self.mop_expander, self.replay_expander, self.wait_gate]

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
        if self.tensix_instruction_decoder.isInstructionRecognised(instruction):
            self.push_mop_instruction(instruction)
        else:
            opcode = extract_bits(instruction, 8, 24)
            raise NotImplementedError(
                f"Unknown op code issued {hex(opcode)} on tensix thread {self.thread_id}"
            )

    def getSize(self):
        return 0xFFFF


class TensixFrontendUnit(Clockable, ABC):
    def __init__(self, frontend, tensix_instruction_decoder):
        self.frontend = frontend
        self.tensix_instruction_decoder = tensix_instruction_decoder


class WaitGate(TensixFrontendUnit):
    def __init__(self, frontend, tensix_instruction_decoder):
        super().__init__(frontend, tensix_instruction_decoder)

    def clock_tick(self, cycle_num):
        instruction = self.frontend.pop_wait_gate_instruction()
        if instruction is not None:
            self.frontend.backend.issueInstruction(instruction)


class TensixReplayExpander(TensixFrontendUnit):
    def __init__(self, frontend, tensix_instruction_decoder):
        self.replay_buffer = [0] * 32
        self.append_instruction_to_buffer = False
        self.exec_while_load = False
        self.replay_len = 0
        self.replay_start_idx = 0
        self.replay_idx = 0
        super().__init__(frontend, tensix_instruction_decoder)

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
                instruction_info = self.tensix_instruction_decoder.getInstructionInfo(
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
    def __init__(self, frontend, tensix_instruction_decoder):
        self.mop_cfg = [0] * 9
        self.mask_hi = 0
        super().__init__(frontend, tensix_instruction_decoder)

    def read(self, address, size):
        raise NotImplementedError("Can not read from Tensix MOP expander configuration")

    def write(self, addr, value, size=None):
        idx = int(addr / 4)
        assert idx < 9

        self.mop_cfg[idx] = conv_to_uint32(value)

    def clock_tick(self, cycle_num):
        instruction = self.frontend.pop_mop_instruction()
        if instruction is not None:
            instruction_info = self.tensix_instruction_decoder.getInstructionInfo(
                instruction
            )
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
