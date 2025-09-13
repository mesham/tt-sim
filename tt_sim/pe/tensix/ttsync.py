from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.conversion import (
    conv_to_bytes,
)


class TTSync(MemMapable):
    def __init__(self, tile_ctrl, tensix_coprocessor, thread_id):
        self.tile_ctrl = tile_ctrl
        self.thread_id = thread_id
        self.tensix_coprocessor = tensix_coprocessor

    def read(self, address, size):
        # For now just return any value - need to implement memory waiting
        # to handle, and when waiting completes it returns an undefined value
        return conv_to_bytes(0)

    def write(self, address, value, size):
        # Writes are discarded
        pass

    def getSize(self):
        return 0x1B
