import numpy as np


class Register:
    def __init__(self, size):
        self.memory = np.empty(size, dtype=np.uint8)
        self.size = size

    def read(self):
        return self.memory[: self.size].tobytes()

    def write(self, value):
        assert isinstance(value, bytes)
        byte_buffer = np.frombuffer(value, dtype=np.uint8)
        self.memory[: self.size] = byte_buffer[: self.size]
