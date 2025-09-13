class AddressRange:
    def __init__(self, low, size):
        self.low = low
        self.size = size
        self.high = low + (size - 1)

    def __hash__(self):
        return hash((self.low, self.high))

    def __eq__(self, other):
        return (self.low, self.high) == (other.low, other.high)

    def check_match(self, addr):
        if addr < self.low:
            return False
        if addr > self.high:
            return False
        return True


class MemoryMap:
    def __init__(self, memory_map=None):
        if memory_map is None:
            self.memory_map = {}
        else:
            self.memory_map = memory_map

    def __setitem__(self, key, value):
        assert isinstance(key, AddressRange)
        self.memory_map[key] = value

    def __getitem__(self, key):
        return self.memory_map[key]

    def __delitem__(self, key):
        del self.memory_map[key]

    def items(self):
        return self.memory_map.items()

    def keys(self):
        return self.memory_map.keys()

    def verify(self):
        for idx1, k1 in enumerate(self.memory_map.keys()):
            for idx2, k2 in enumerate(self.memory_map.keys()):
                if idx1 != idx2:
                    # Test either the low range overlaps (greater than k1 low but smaller than k1 high) or
                    # that the high range overlaps (greater than k1 low but smaller than k1 high)
                    if (k2.low >= k1.low and k2.low < k1.high) or (
                        k2.high >= k1.low and k2.high < k1.high
                    ):
                        raise IndexError(
                            f"Memory range at index {idx1} overlaps with range at index {idx2}"
                        )

    @classmethod
    def merge(cls, *memory_maps):
        new_mem_map = {}
        for memory_map in memory_maps:
            for key, value in memory_map.items():
                if key in new_mem_map:
                    raise KeyError("Can not add duplicate memory range to a memory map")
                new_mem_map[key] = value
        newmm = MemoryMap(new_mem_map)
        newmm.verify()
        return newmm
