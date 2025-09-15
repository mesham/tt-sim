from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.util import TensixInstructionDecoder
from tt_sim.util.bits import get_nth_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TensixSyncUnit(TensixBackendUnit, MemMapable):
    class TTSemaphore:
        def __init__(self):
            self.value = 0
            self.max = 0

    class TTMutex:
        def __init__(self):
            self.held_by = None

    OPCODE_TO_HANDLER = {
        "SEMINIT": "handle_seminit",
        "STALLWAIT": "handle_stallwait",
        "SEMWAIT": "handle_semwait",
        "SEMPOST": "handle_sempost",
        "SEMGET": "handle_semget",
        "ATGETM": "handle_atgetm",
        "ATRELM": "handle_atrelm",
    }

    def __init__(self, backend):
        super().__init__(backend, TensixSyncUnit.OPCODE_TO_HANDLER, "Sync")
        self.semaphores = [TensixSyncUnit.TTSemaphore()] * 8
        # 7 mutexes, but index 1 is ignored
        self.mutexes = [TensixSyncUnit.TTMutex()] * 8
        self.blocked_mutex = []

    def issueInstruction(self, instruction, from_thread):
        instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
        instruction_name = instruction_info["name"]
        if instruction_name == "ATGEM" or instruction_name == "ATRELM":
            # Allowed up to three of these as long as they don't reference the same mutex
            if len(self.next_instruction) < 3:
                index = instruction_info["instr_args"]["mutex_index"]
                for instr, _ in self.next_instruction:
                    instruction_n_info = TensixInstructionDecoder.getInstructionInfo(
                        instr
                    )
                    if instruction_n_info["instr_args"]["mutex_index"] == index:
                        # Same mutex referenced, do not issue this cycle
                        return False
                self.next_instruction.append(
                    (
                        instruction,
                        from_thread,
                    )
                )
                return True
            else:
                # Three or more already issued, do not issue this cycle
                return False
        else:
            # Only one of any other instruction allowed
            if not self.checkIfNextInstructionsContainAnyOtherOpcodes(
                "ATGEM", "ATRELM"
            ):
                self.next_instruction.append(
                    (
                        instruction,
                        from_thread,
                    )
                )
                return True
            else:
                return False

    def clock_tick(self, cycle_num):
        super().clock_tick(cycle_num)
        to_remove = []
        for idx, (issue_thread, index) in enumerate(self.blocked_mutex):
            if (
                self.mutexes[index].held_by is None
                or self.mutexes[index].held_by == issue_thread
            ):
                self.mutexes[index].held_by = issue_thread
                self.backend.getFrontendThread(
                    issue_thread
                ).wait_gate.informMutexAcquired()
                to_remove.append(idx)
        if len(to_remove) > 1:
            to_remove.reverse()
            for idx in to_remove:
                del self.blocked_mutex[idx]

    def getSemaphore(self, idx):
        assert idx <= 7
        return self.semaphores[idx]

    def handle_atrelm(self, instruction_info, issue_thread, instr_args):
        index = instr_args["mutex_index"]
        if self.mutexes[index].held_by == issue_thread:
            # If I own the mutex then release it, otherwise ignore
            self.mutexes[index].held_by = None

    def handle_atgetm(self, instruction_info, issue_thread, instr_args):
        index = instr_args["mutex_index"]
        if (
            self.mutexes[index].held_by is not None
            and self.mutexes[index].held_by != issue_thread
        ):
            self.blocked_mutex.append((issue_thread, index))
        else:
            self.mutexes[index].held_by = issue_thread
            self.backend.getFrontendThread(issue_thread).wait_gate.informMutexAcquired()

    def handle_sempost(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        for i in range(8):
            if get_nth_bit(sem_sel, i) and self.semaphores[i].value < 15:
                self.semaphores[i].value += 1

    def handle_seminit(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        new_value = instr_args["init_value"]
        max_value = instr_args["max_value"]

        for i in range(8):
            if get_nth_bit(sem_sel, i):
                self.semaphores[i].value = new_value
                self.semaphores[i].max = max_value

    def handle_semget(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        for i in range(8):
            if get_nth_bit(sem_sel, i) and self.semaphores[i].value < 15:
                self.semaphores[i].value -= 1

    def handle_stallwait(self, instruction_info, issue_thread, instr_args):
        cond_mask = instr_args["wait_res"]
        block_mask = instr_args["stall_res"]

        self.backend.getFrontendThread(
            issue_thread
        ).wait_gate.setLatchedWaitInstruction(
            "STALLWAIT",
            cond_mask if cond_mask else 0x7F,
            block_mask if block_mask else 1 << 6,
        )

    def handle_semwait(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        cond_mask = instr_args["wait_sem_cond"]
        block_mask = instr_args["stall_res"]
        block_mask = block_mask if block_mask else 1 << 6

        if cond_mask:
            self.backend.getFrontendThread(
                issue_thread
            ).wait_gate.setLatchedWaitInstruction(
                "SEMWAIT", cond_mask, block_mask, sem_sel
            )
        else:
            self.backend.getFrontendThread(
                issue_thread
            ).wait_gate.setLatchedWaitInstruction("STALLWAIT", 0x7F, block_mask)

    def read(self, addr, size):
        # Accesses semaphore[i].value, where each
        # entry is 32 bit
        idx = int(addr / 4)
        assert idx < 8
        return conv_to_bytes(self.semaphores[idx].value)

    def write(self, addr, value, size=None):
        """
        This is taken from the functional model code at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SyncUnit.md#semaphores
        """
        idx = int(addr / 4)
        assert idx < 8
        if conv_to_uint32(value) & 1:
            # This is like a SEMGET instruction
            if self.semaphores[idx].value > 0:
                self.semaphores[idx].value = -1
        else:
            # This is like a SEMPOST instruction
            if self.semaphores[idx].value < 15:
                self.semaphores[idx].value += 1

    def getSize(self):
        return 0xFFDF
