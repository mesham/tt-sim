from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory import MemoryStall
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType


class Mailbox(MemMapable):
    def __init__(self, core_id):
        self.core_id = core_id
        self.mailboxes = [[], [], [], []]
        self.other_mbs = None

    def setOtherMBs(self, other_mbs):
        self.other_mbs = other_mbs

    def sendValue(self, idx, value):
        if len(self.mailboxes[idx]) < 3:
            self.mailboxes[idx].append(value)
            return True
        else:
            return False

    def coreTypeToIdx(self, ct):
        match ct:
            case BabyRISCVCoreType.BRISC:
                return 0
            case BabyRISCVCoreType.TRISC0:
                return 1
            case BabyRISCVCoreType.TRISC1:
                return 2
            case BabyRISCVCoreType.TRISC2:
                return 3
            case _:
                raise ValueError()

    def read(self, addr, size):
        if addr == 0x0:
            mb_idx = 0
            isRead = True
        elif addr == 0x4:
            mb_idx = 0
            isRead = False
        elif addr == 0x1000:
            mb_idx = 1
            isRead = True
        elif addr == 0x1004:
            mb_idx = 1
            isRead = False
        elif addr == 0x2000:
            mb_idx = 2
            isRead = True
        elif addr == 0x2004:
            mb_idx = 2
            isRead = False
        elif addr == 0x3000:
            mb_idx = 3
            isRead = True
        elif addr == 0x3004:
            mb_idx = 3
            isRead = False
        else:
            raise IndexError(
                f"Reading from address {hex(addr)} not yet supported by mailbox"
            )

        if isRead:
            if len(self.mailboxes[mb_idx]) > 0:
                return self.mailboxes[mb_idx].pop(0)
            else:
                return MemoryStall
        else:
            return 0 if len(self.mailboxes[mb_idx]) == 0 else 1

    def write(self, addr, value, size=None):
        if addr == 0x0:
            # B0
            ct = BabyRISCVCoreType.BRISC
        elif addr == 0x1000:
            # T0
            ct = BabyRISCVCoreType.TRISC0
        elif addr == 0x2000:
            # T1
            ct = BabyRISCVCoreType.TRISC1
        elif addr == 0x3000:
            # T2
            ct = BabyRISCVCoreType.TRISC2
        else:
            raise IndexError(
                f"Writing to address {hex(addr)} not yet supported by mailbox"
            )

        my_ct = self.coreTypeToIdx(self.core_id)
        if self.core_id == ct:
            accepted = self.sendValue(my_ct, value)
        else:
            tgt_idx = self.coreTypeToIdx(ct)
            accepted = self.other_mbs[tgt_idx].sendValue(my_ct, value)
        return accepted

    def getSize(self):
        return 0x3FFF
