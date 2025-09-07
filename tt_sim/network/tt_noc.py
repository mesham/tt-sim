from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32, insert_bytes


class NoCRouter(MemMapable):
    def __init__(self, noc_number, x_coord, y_coord):
        assert noc_number == 0 or noc_number == 1
        self.noc_number = noc_number
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.generate_NIU_and_NoC_config()

    def generate_NIU_and_NoC_config(self):
        # https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/MemoryMap.md#niu-and-noc-router-configuration

        self.niu_cfg_0 = insert_bytes(
            0, 0, 12, 1
        )  # tile clock disable, 1=disable and 0=enable
        self.niu_cfg_0 = insert_bytes(
            self.niu_cfg_0, 0, 13, 1
        )  # double store disable, 1=disable and 0=enable
        self.niu_cfg_0 = insert_bytes(
            self.niu_cfg_0, 0, 14, 1
        )  # coordinate translation enable, 1=enable and 0=disable

        self.router_cfg_0 = 0
        self.router_cfg_1 = 0
        self.router_cfg_2 = 0
        self.router_cfg_3 = 0
        self.router_cfg_4 = 0

        self.noc_id_logical = insert_bytes(0, self.x_coord, 6, 0)
        self.noc_id_logical = insert_bytes(self.noc_id_logical, self.y_coord, 6, 6)

    def read(self, addr, size):
        if addr == 0x0138:
            return conv_to_bytes(self.noc_id_logical)
        elif addr == 0x100:
            return conv_to_bytes(self.niu_cfg_0)
        elif addr == 0x104:
            return conv_to_bytes(self.router_cfg_0)
        elif addr == 0x108:
            return conv_to_bytes(self.router_cfg_1)
        elif addr == 0x10C:
            return conv_to_bytes(self.router_cfg_2)
        elif addr == 0x110:
            return conv_to_bytes(self.router_cfg_3)
        elif addr == 0x114:
            return conv_to_bytes(self.router_cfg_4)
        else:
            raise NotImplementedError(
                f"Reading from address {hex(addr)} not yet supported by NoC"
            )

    def write(self, addr, value, size=None):
        if addr == 0x0138:
            self.noc_id_logical = conv_to_uint32(value)
        elif addr == 0x100:
            self.niu_cfg_0 = conv_to_uint32(value)
        elif addr == 0x104:
            self.router_cfg_0 = conv_to_uint32(value)
        elif addr == 0x108:
            self.router_cfg_1 = conv_to_uint32(value)
        elif addr == 0x10C:
            self.router_cfg_2 = conv_to_uint32(value)
        elif addr == 0x110:
            self.router_cfg_3 = conv_to_uint32(value)
        elif addr == 0x114:
            self.router_cfg_4 = conv_to_uint32(value)
        else:
            raise NotImplementedError(
                f"Writing to address {hex(addr)} not yet supported by NoC"
            )

    def getSize(self):
        return 0xFFFF
