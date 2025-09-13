from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.bits import set_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TDMA(MemMapable, Clockable):
    def __init__(self, tensix_coprocessor):
        self.mover = tensix_coprocessor.getBackend().getMoverUnit()
        self.cmd_params = [0] * 4
        self.command_queue = []

    def read(self, addr, size):
        if addr == 0x14:
            return conv_to_bytes(self.generate_status_bits())
        else:
            raise NotImplementedError(
                f"Reading from tdma address {hex(addr)} not supported"
            )

    def write(self, addr, value, size=None):
        if addr >= 0x0 and addr <= 0xC:
            self.cmd_params[int(addr / 4)] = conv_to_uint32(value)
        elif addr == 0x10:
            # Enqueue command, assume we have unlimited length for simplicity
            cmd = conv_to_uint32(value)
            if cmd >> 31:
                self.command_queue.append([cmd, 0, 0, 0, 0])
            else:
                self.command_queue.append([cmd, *self.cmd_params])
        elif addr == 0x24:
            pass
        else:
            raise NotImplementedError(
                f"Writing to tdma address {hex(addr)} not supported"
            )

    def generate_status_bits(self):
        result = set_bit(0, 3)

        return result

    def clock_tick(self, cycle_num):
        for command in self.command_queue:
            opcode = command[0] & 0xFF
            match opcode:
                case 0x40:
                    # Mover command
                    if command[0] >> 31:
                        raise NotImplementedError()
                    else:
                        src = command[1]
                        dst = command[2]
                        count = command[3] & 0xFFFF
                        mode = (
                            command[4] & 3
                        )  # 0 = XFER_L0_L1, 1 = XFER_L1_L0, 2 = XFER_L0_L0, 3 = XFER_L1_L1
                    self.mover.move(dst << 4, src << 4, count << 4, mode)
                case 0x46:
                    # Mover wait command
                    pass
                case 0x66:
                    # L1 write command (32b or 64b)
                    if command[0] >> 31:
                        raise NotImplementedError()
                    elif (command[0] & 0x600) != 0x600:
                        raise NotImplementedError()
                    else:
                        dst = command[1]
                        if dst >= (1024 * 1464):
                            # Dst must be an address in L1
                            raise NotImplementedError()
                        if command[0] & 0x100:
                            # TODO: write to dst
                            pass
                            # async *(uint64_t*)Dst = (uint64_t(Params[3]) << 32) | uint64_t(Params[2]);
                        else:
                            # TODO: write to dst
                            pass
                            # async *(uint32_t*)Dst = Params[2];
                case 0x89:
                    # NOP command
                    pass
                case _:
                    raise NotImplementedError()

        self.command_queue.clear()

    def getSize(self):
        return 0xFFF
