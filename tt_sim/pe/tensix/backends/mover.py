from enum import IntEnum

from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.util.conversion import conv_to_bytes


class MoverUnit(TensixBackendUnit):
    class XMOV_DIRECTION(IntEnum):
        XMOV_L0_TO_L1 = 0
        XMOV_L1_TO_L0 = 1
        XMOV_L0_TO_L0 = 2
        XMOV_L1_TO_L1 = 3

    OPCODE_TO_HANDLER = {}

    TENSIX_CFG_BASE = 0xFFEF0000
    MEM_NCRISC_IRAM_BASE = 0xFFC00000

    def __init__(self, backend):
        super().__init__(backend, MoverUnit.OPCODE_TO_HANDLER, "Mover")

    def move(self, dst, src, count, mode):
        """
        This is based on the functional model description at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/Mover.md
        """
        assert self.backend.getAddressableMemory() is not None
        if (
            mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1
            or mode == MoverUnit.XMOV_DIRECTION.XMOV_L0_TO_L1
        ):
            # In the "_TO_L1" modes, dst must be an address in L1.
            assert dst < 1024 * 1464
        else:
            if dst <= 0xFFFF:
                dst += MoverUnit.TENSIX_CFG_BASE
            elif 0x40000 <= dst and dst <= 0x4FFFF:
                dst = (dst - 0x40000) + MoverUnit.MEM_NCRISC_IRAM_BASE
            else:
                dst = None  # Operation still happens, but the writes get discarded.

            if (dst & 0xFFFF) + count > 0x10000:
                raise NotImplementedError(
                    "Can not access more than one region at a time"
                )

        # Perform the operation.
        if (
            mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1
            or mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L0
        ):
            # In the "L1_TO_" modes, a memcpy is done, and src must be an address in L1.
            if src >= (1024 * 1464):
                raise NotImplementedError("")
            # print(f"Write to {hex(dst)} from {hex(src)} elements {hex(count)}")
            self.backend.getAddressableMemory().write(
                dst, self.backend.getAddressableMemory().read(src, count)
            )
        else:
            # In the "L0_TO_" modes, a memset is done.
            zero_val = conv_to_bytes(0, count)
            self.backend.getAddressableMemory().write(dst, zero_val)
