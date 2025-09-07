class AddressRange:
    def __init__(self, low, size):
        self.low = low
        self.size = size
        self.high = low + size

    def check_match(self, addr):
        if addr < self.low:
            return False
        if addr > self.high:
            return False
        return True


class MemoryMap:
    def __init__(self):
        self.memory_map = {}

    def __setitem__(self, key, value):
        assert isinstance(key, AddressRange)
        self.memory_map[key] = value

    def __getitem__(self, key):
        return self.memory_map[key]

    def items(self):
        return self.memory_map.items()

    def write(self, address, value):
        pass
