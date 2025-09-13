import inspect

from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.rv.babyriscv import BabyRISCV, BabyRISCVCoreType


class PCBuf(MemMapable):
    def __init__(self, tile_ctrl, buf_id):
        # In reality this is max 16 32-bit values, but for now allow it to be
        # of unlimited size
        self.fifo = []
        self.buf_id = buf_id
        self.tile_ctrl = tile_ctrl

    def read(self, address, size):
        caller = self.find_caller_instance(BabyRISCV)
        assert caller is not None

        if caller.core_type == BabyRISCVCoreType.BRISC:
            # TODO: Need to add in delay and wait
            return 0
        else:
            # TODO: Are assuming there is a value, need to add waits
            return self.fifo.pop(0)

    def write(self, address, value, size):
        caller = self.find_caller_instance(BabyRISCV)
        assert caller is not None

        if caller.core_type == BabyRISCVCoreType.BRISC:
            self.fifo.append(value)

    def find_caller_instance(self, cls_type):
        for frame_info in inspect.stack()[1:]:  # Skip current frame
            local_self = frame_info.frame.f_locals.get("self")
            if isinstance(local_self, cls_type):
                return local_self
        return None

    def getSize(self):
        # This is for BRISC
        return 0xFFFF
