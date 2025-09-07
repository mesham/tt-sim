from enum import Enum

import numpy as np


class RegisterAccessMode(Enum):
    R = 1
    RW = 2


class Register:
    def __init__(
        self,
        size,
        init_val=None,
        access_mode=RegisterAccessMode.RW,
        error_on_write_to_read=True,
    ):
        self.memory = np.empty(size, dtype=np.uint8)
        self.size = size
        self.access_mode = access_mode
        self.error_on_write_to_read = error_on_write_to_read

        if init_val is not None:
            assert isinstance(init_val, bytes)
            byte_buffer = np.frombuffer(init_val, dtype=np.uint8)
            self.memory[: self.size] = byte_buffer[: self.size]

    def read(self):
        return self.memory[: self.size].tobytes()

    def write(self, value):
        if self.access_mode != RegisterAccessMode.RW:
            # Some ISA might error on this, others such as RV simply ignore it
            if self.error_on_write_to_read:
                raise Exception("Can not call set on a read only register")
            else:
                return
        assert isinstance(value, bytes)
        byte_buffer = np.frombuffer(value, dtype=np.uint8)
        self.memory[: self.size] = byte_buffer[: self.size]
