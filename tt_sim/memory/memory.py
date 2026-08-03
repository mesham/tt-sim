from abc import ABC

import numpy as np

from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory_map import MemoryMap
from tt_sim.trace import EventCategory, MemEvent, Unit, get_bus
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

_UNKNOWN_UNIT_ID = (0, 0, 0, Unit.UNKNOWN.value)


class MemoryStall:
    pass


class MemorySpace(MemMapable, ABC):
    def __init__(self, memory_map, safe=True, snoop_addresses=None):
        self.memory_map = memory_map
        self.safe = safe
        self.snoop_addresses = [] if snoop_addresses is None else snoop_addresses
        # caller_context is set by callers (typically RV cores) before an
        # access so error messages and trace events can attribute the
        # access. Layout: (unit_id_tuple_or_None, core_label, pc).
        self.caller_context: tuple | None = None

    def add_snoop(self, snoop_addr_low, snoop_addr_high):
        self.snoop_addresses.append((snoop_addr_low, snoop_addr_high))
        return len(self.snoop_addresses)

    def _resolve_caller_unit_id(self) -> tuple:
        ctx = self.caller_context
        if ctx is None:
            return _UNKNOWN_UNIT_ID
        # ctx is (unit_id, core_label, pc); unit_id may be None during
        # construction-period accesses.
        unit_id = ctx[0]
        if unit_id is None:
            return _UNKNOWN_UNIT_ID
        return unit_id

    def _classify_region(self, addr: int) -> str:
        if addr >= 0xFF000000:
            return "MMIO"
        if addr >= 0xFFB00000:
            return "MMIO"
        return "L1"

    def _publish_mem_event(self, op: str, addr: int, size: int):
        bus = get_bus()
        if not bus.is_enabled(EventCategory.MEM):
            return
        # Extract PC from caller_context if available — tuple shape is
        # (unit_id, core_label, pc) when set by an RV core.
        pc = 0
        ctx = self.caller_context
        if ctx is not None and len(ctx) >= 3:
            pc = int(ctx[2]) if ctx[2] is not None else 0
        bus.publish(
            MemEvent(
                cycle=0,
                unit_id=self._resolve_caller_unit_id(),
                op=op,
                address=int(addr),
                size=int(size),
                region=self._classify_region(addr),
                pc=pc,
            )
        )

    def _locate_memory_space(self, addr):
        addr_range, memory_space = self.memory_map.locate(addr)
        if addr_range is not None:
            return addr_range, memory_space

        if self.safe:
            msg = f"Provided address '{hex(addr)}' does not match any registered memory spaces"
            if self.caller_context is not None:
                core_id = (
                    self.caller_context[1] if len(self.caller_context) > 1 else "?"
                )
                pc = self.caller_context[2] if len(self.caller_context) > 2 else 0
                msg += f" (accessed by {core_id} at PC={hex(pc)})"
            if addr >= 0xFF000000:
                msg += (
                    "\nIf the program uses tt-metal CommandQueue dispatch "
                    "(e.g. EnqueueProgram), note that CQ dispatch is unsupported "
                    "in tt-sim — see ROADMAP.md §E."
                )
            raise IndexError(msg)
        else:
            return None, None

    def convert_addr_to_target_range(self, addr_range, addr):
        return addr - addr_range.low

    def check_is_snoop(self, addr):
        for snoop in self.snoop_addresses:
            if addr >= snoop[0] and addr <= snoop[1]:
                return True
        return False

    def read(self, addr, size):
        self._publish_mem_event("read", addr, size)
        addr_range, memory_space = self._locate_memory_space(addr)
        if addr_range is not None and memory_space is not None:
            target_addr = self.convert_addr_to_target_range(addr_range, addr)
            if len(self.snoop_addresses) > 0 and self.check_is_snoop(addr):
                print(
                    f"->>>>>> Read value {hex(conv_to_uint32(memory_space.read(target_addr, size)))} at address {hex(addr)}"
                )
            return memory_space.read(target_addr, size)
        else:
            if len(self.snoop_addresses) > 0 and self.check_is_snoop(addr):
                print(
                    f"->>>>>> Attempt to read at address {hex(addr)} but no memory registered"
                )
            return conv_to_bytes(0)

    def write(self, addr, value, size=None):
        write_size = (
            size
            if size is not None
            else (len(value) if hasattr(value, "__len__") else 0)
        )
        self._publish_mem_event("write", addr, write_size)
        if len(self.snoop_addresses) > 0 and self.check_is_snoop(addr):
            print(
                f"->>>>>> Write value {hex(conv_to_uint32(value))} at address {hex(addr)}"
            )
        addr_range, memory_space = self._locate_memory_space(addr)
        if addr_range is not None and memory_space is not None:
            target_addr = self.convert_addr_to_target_range(addr_range, addr)
            return memory_space.write(target_addr, value, size)

    def getSize(self):
        low_val = None
        high_val = None
        for addr_range in self.memory_map.keys():
            if low_val is None or low_val > addr_range.low:
                low_val = addr_range.low
            if high_val is None or high_val < addr_range.high:
                high_val = addr_range.high
        assert high_val is not None
        assert low_val is not None
        return high_val - low_val

    @classmethod
    def merge(cls, *memory_spaces):
        memory_maps = []
        safe = False
        snoops = []
        for memory_space in memory_spaces:
            memory_maps.append(memory_space.memory_map)
            snoops += memory_space.snoop_addresses
            # Any memory space with safety turned on turns it on for all
            if memory_space.safe:
                safe = True

        new_mm = MemoryMap.merge(*memory_maps)
        return cls(new_mm, safe, snoops)


class VisibleMemory(MemorySpace):
    def __init__(self, memory_map, safe=True, snoop_addresses=None):
        super().__init__(memory_map, safe, snoop_addresses)


class TensixMemory(MemorySpace):
    def __init__(self, memory_map, safe=True, snoop_addresses=None):
        super().__init__(memory_map, safe, snoop_addresses)


class TileMemory(MemorySpace):
    def __init__(self, memory_map, safe=True, snoop_addresses=None):
        super().__init__(memory_map, safe, snoop_addresses)


class AddressableMemory(MemMapable):
    def __init__(self, size, alignment=None):
        # Zero-init: silicon L1/DRAM is uninitialised at power-on but
        # kernels assume zero in places (e.g. mailbox state checks).
        # np.empty would inherit whatever was in the underlying memory
        # pages, which is non-deterministic and process-pattern
        # sensitive (e.g. loading a large ELF for DWARF/LCOV would
        # change the allocator's reuse pattern and start surfacing
        # garbage reads).
        self.memory = np.zeros(size, dtype=np.uint8)
        self.size = size
        self.alignment = alignment

    def read(self, addr, size):
        if addr > self.size:
            raise IndexError(
                f"Start address '{addr}' overflows memory size '{self.size}'"
            )
        if addr + size > self.size:
            raise IndexError(
                f"End address '{addr + size}' overflows memory size '{self.size}'"
            )
        return self.memory[addr : addr + size].tobytes()

    def write(self, addr, value, size=None):
        assert isinstance(value, bytes)

        if size is None:
            size = len(value)

        if addr > self.size:
            raise IndexError(
                f"Start address '{addr}' overflows memory size '{self.size}'"
            )
        if addr + size > self.size:
            raise IndexError(
                f"End address '{addr + size}' overflows memory size '{self.size}'"
            )

        if self.alignment is not None:
            if addr % self.alignment != 0:
                raise IndexError(
                    f"Start address must be aligned to '{self.alignment}' whereas '{addr}' is not"
                )

        byte_buffer = np.frombuffer(value, dtype=np.uint8)
        self.memory[addr : addr + size] = byte_buffer[:size]

    def getSize(self):
        return self.size


class SparseAddressableMemory(MemMapable):
    """An :class:`AddressableMemory` whose backing store is allocated on demand.

    Same read/write contract, but the space is cut into fixed-size chunks and a
    chunk's ``np.zeros`` is only materialised the first time it is *written*;
    reads of an untouched chunk return zeros. That is the same power-on contract
    ``AddressableMemory`` documents (and the same reason it uses ``np.zeros``
    rather than ``np.empty``) — just paid for lazily.

    This exists because a DRAM channel is honestly large: 2 GiB on Wormhole and
    4 GiB on Blackhole, times 6 / 8 channels, so a flat array per channel would
    ask for 12 / 32 GiB at device construction. The vendor reference simulator
    solves it the same way, with a lazily-faulted anonymous ``mmap`` per channel
    (ttsim ``src/sim.cpp``); Python has no equivalent, so chunk it explicitly.
    """

    #: 2 MiB, matching the huge page ttsim ``madvise``s its DRAM mapping to.
    CHUNK_SIZE = 2 * 1024 * 1024

    def __init__(self, size, alignment=None, chunk_size=None):
        self.size = size
        self.alignment = alignment
        self.chunk_size = self.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunks: dict[int, np.ndarray] = {}

    def _check_range(self, addr, size):
        if addr > self.size:
            raise IndexError(
                f"Start address '{addr}' overflows memory size '{self.size}'"
            )
        if addr + size > self.size:
            raise IndexError(
                f"End address '{addr + size}' overflows memory size '{self.size}'"
            )

    def read(self, addr, size):
        self._check_range(addr, size)
        if size <= 0:
            return b""
        index, offset = divmod(addr, self.chunk_size)
        if offset + size <= self.chunk_size:
            # Common case: the access sits inside one chunk.
            chunk = self.chunks.get(index)
            if chunk is None:
                return bytes(size)
            return chunk[offset : offset + size].tobytes()
        out = bytearray(size)
        pos = 0
        while pos < size:
            take = min(self.chunk_size - offset, size - pos)
            chunk = self.chunks.get(index)
            if chunk is not None:
                out[pos : pos + take] = chunk[offset : offset + take].tobytes()
            pos += take
            index += 1
            offset = 0
        return bytes(out)

    def write(self, addr, value, size=None):
        assert isinstance(value, bytes)

        if size is None:
            size = len(value)

        self._check_range(addr, size)

        if self.alignment is not None and addr % self.alignment != 0:
            raise IndexError(
                f"Start address must be aligned to '{self.alignment}' whereas '{addr}' is not"
            )

        if size <= 0:
            return
        byte_buffer = np.frombuffer(value, dtype=np.uint8)[:size]
        index, offset = divmod(addr, self.chunk_size)
        pos = 0
        while pos < size:
            take = min(self.chunk_size - offset, size - pos)
            chunk = self.chunks.get(index)
            if chunk is None:
                chunk = np.zeros(self.chunk_size, dtype=np.uint8)
                self.chunks[index] = chunk
            chunk[offset : offset + take] = byte_buffer[pos : pos + take]
            pos += take
            index += 1
            offset = 0

    def getSize(self):
        return self.size


class DRAM(AddressableMemory):
    def __init__(self, size):
        super().__init__(size, None)


class SparseDRAM(SparseAddressableMemory):
    """A DRAM channel: too large to allocate eagerly, so chunked on demand."""

    def __init__(self, size):
        super().__init__(size, None)


class L1(AddressableMemory):
    def __init__(self, size):
        super().__init__(size, None)
