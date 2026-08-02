from enum import IntEnum

import numpy as np

from tt_sim.util.conversion import conv_to_uint32


class DstRegister:
    def __init__(self):
        self.dstBits = np.zeros([1024, 16], dtype=np.uint32)
        # DEST_ACCESS_CFG row-remap gates (Blackhole). Both default off, which
        # is the hardware reset state and every path tt-sim currently drives.
        # With them off, Adj16 is the identity and Adj32 reduces to the shared
        # 32-bit row fold, so the addressing is byte-identical to Wormhole.
        # tt-sim does not yet model writes to DEST_ACCESS_CFG (a Blackhole
        # compute path that enables the swizzle is future work), so these stay
        # False today; see BlackholeA0/.../Dst.md for the gated transforms.
        self.dest_remap_addrs = False
        self.dest_swizzle_32b = False

    def adj16(self, r):
        """Blackhole ``Adj16`` Dst16b row map (identity unless the remap gate).

        Per BlackholeA0 Dst.md: preserves bits [1:0] and [6:2], XORing in
        shifted copies of bits [5:4] and bit [3] when remap is enabled.
        """
        if self.dest_remap_addrs:
            r = (r & 0x3C7) ^ ((r & 0x030) >> 1) ^ ((r & 0x008) << 2)
        return r

    def adj32(self, r):
        """Blackhole ``Adj32`` Dst32b base row: ``Adj16``, an optional 32-bit
        swizzle, then the shared fold ``((r & 0x1F8) << 1) | (r & 0x207)`` that
        both architectures apply."""
        r = self.adj16(r)
        if self.dest_swizzle_32b:
            r = (r & 0x3F3) ^ ((r & 0x018) >> 1) ^ ((r & 0x004) << 1)
        return ((r & 0x1F8) << 1) | (r & 0x207)

    def getDst16b(self, idx0, idx1):
        return int(self.dstBits[self.adj16(idx0)][idx1])

    def setDst16b(self, idx0, idx1, value):
        self.dstBits[self.adj16(idx0)][idx1] = value

    def getDst32b(self, idx0, idx1):
        br = self.adj32(idx0)
        v1 = self.dstBits[br][idx1]
        v2 = self.dstBits[br + 8][idx1]
        return int((v1 << 16) | (v2 & 0xFFFF))

    def setDst32b(self, idx0, idx1, value):
        br = self.adj32(idx0)
        self.dstBits[br][idx1] = value >> 16
        self.dstBits[br + 8][idx1] = value & 0xFFFF

    def setUndefinedRow(self, row, isDst32=False):
        # ZEROACC zeroes the accumulator. On real hardware reads after this
        # return zero bits, so just clear the backing storage.
        if isDst32:
            self.dstBits[row * 2, :] = 0
            self.dstBits[row * 2 + 1, :] = 0
        else:
            self.dstBits[row, :] = 0


class SrcRegister:
    class SrcClient(IntEnum):
        MatrixUnit = 0
        Unpackers = 1

    def __init__(self):
        self.allowedClient = SrcRegister.SrcClient.Unpackers
        # Zeroed, not uninitialised: kernels do read Src rows nothing has
        # written this pass (reduce_tile's SrcB rows 16-31 scratch, or the rows
        # a one-row scaler unpack leaves alone), so leaving them as whatever
        # numpy handed back made results depend on unrelated allocations.
        self.data = np.zeros([64, 16], dtype=np.uint32)
        # The format the unpacker wrote this bank in, latched when the bank was
        # handed to the matrix unit. Blackhole's matrix unit reads the operand
        # format from here rather than from ALU_FORMAT_SPEC_REG0_SrcA/1_SrcB;
        # see MatrixUnit.get_dataformat_and_useDst. None until first set.
        self.dataFormat = None

    def flipAllowedClient(self):
        if self.allowedClient == SrcRegister.SrcClient.Unpackers:
            self.allowedClient = SrcRegister.SrcClient.MatrixUnit
        else:
            self.allowedClient = SrcRegister.SrcClient.Unpackers

    def getAllowedClient(self):
        return self.allowedClient

    def setAllowedClient(self, c):
        self.allowedClient = c

    def getDataFormat(self):
        return self.dataFormat

    def setDataFormat(self, fmt):
        self.dataFormat = fmt

    def __getitem__(self, key):
        x, y = key
        return int(self.data[x][y])

    def __setitem__(self, key, value):
        x, y = key
        self.data[x][y] = value


class LReg:
    """One SFPU LReg (32 lanes).

    On Blackhole every lane holds a **uint32 bit pattern**, matching ttsim's
    ``l_regs`` — so a float value is stored as its FP32 bits and integer/bitwise
    results as their bits, and each op reads with the right accessor
    (``conv_to_float`` for the float value, the raw int for bits). This keeps the
    whole SFPU pipeline bit-exact with the reference (e.g. recip's Newton
    refinement). Wormhole keeps the historical mixed float/int model untouched
    (its replay guards encode that behaviour), selected by ``blackhole=False``.
    """

    def __init__(self, blackhole=False):
        self.blackhole = blackhole
        self.read_only = False
        self.hard_wired_value = None
        self.data = [0] * 32

    @staticmethod
    def _coerce(value):
        # Normalise any lane value to a uint32 bit pattern.
        if isinstance(value, float):
            return conv_to_uint32(value)
        return int(value) & 0xFFFFFFFF

    def __setitem__(self, key, value):
        assert not self.read_only
        self.data[key] = self._coerce(value) if self.blackhole else value

    def __getitem__(self, key):
        if self.hard_wired_value is not None:
            return self.hard_wired_value
        else:
            return self.data[key]

    def setReadOnly(self, hard_wired_value=None):
        self.read_only = True
        self.setHardwiredValue(hard_wired_value)

    def setHardwiredValue(self, value):
        if self.blackhole and value is not None:
            value = self._coerce(value)
        self.hard_wired_value = value
