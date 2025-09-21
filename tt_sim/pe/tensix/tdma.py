from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.bits import set_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TDMA(MemMapable, Clockable):
    def __init__(self, tensix_coprocessor, l1_mem):
        self.mover = tensix_coprocessor.getBackend().getMoverUnit()
        self.cmd_params = [0] * 4
        self.command_queue = []
        self.l1_mem = l1_mem

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
            print(f"SET! {hex(conv_to_uint32(value) & 0xffffff7)}")
        else:
            raise NotImplementedError(
                f"Writing to tdma address {hex(addr)} not supported"
            )

    def generate_status_bits(self):
        result = set_bit(0, 3)

        return result

    def clock_tick(self, cycle_num):
        if len(self.command_queue) > 0:
            command = self.command_queue.pop(0)
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
                    self.mover.append_command_from_tdma(
                        (
                            dst << 4,
                            src << 4,
                            count << 4,
                            mode,
                        )
                    )
                case 0x46:
                    # Mover wait command
                    if self.mover.checkForOutstandingInstructions():
                        # If this is outstanding then insert command into the head of the queue
                        # so it will keep being checked until the mover is free
                        self.command_queue.insert(0, command)
                case 0x66:
                    # L1 write command (32b or 64b)
                    if command[0] >> 31:
                        raise ValueError()
                    elif (command[0] & 0x600) != 0x600:
                        raise ValueError()
                    else:
                        dst = command[1]
                        if dst >= (1024 * 1464):
                            raise ValueError("dst must be an address in L1")
                        if command[0] & 0x100:
                            # 64 bit write to L1
                            self.l1_mem.write(dst, conv_to_bytes(command[2]))
                            self.l1_mem.write(dst + 4, conv_to_bytes(command[3]))
                        else:
                            # 32 bit write to L1
                            self.l1_mem.write(dst, conv_to_bytes(command[2]))
                            pass
                case 0x89:
                    # NOP command
                    pass
                case _:
                    raise NotImplementedError()

    def getSize(self):
        return 0xFFF
