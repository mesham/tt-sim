from enum import IntEnum

import numpy as np


class DstRegister:
    def __init__(self):
        self.dstBits = np.zeros([1024, 16], dtype=np.uint32)

    def getDst16b(self, idx0, idx1):
        return int(self.dstBits[idx0][idx1])

    def setDst16b(self, idx0, idx1, value):
        self.dstBits[idx0][idx1] = value

    def to_32b_row(self, r_16b):
        br = ((r_16b & 0x1F8) << 1) | (r_16b & 0x207)
        return br, br + 8

    def getDst32b(self, idx0, idx1):
        r1, r2 = self.to_32b_row(idx0)
        v1 = self.dstBits[r1][idx1]
        v2 = self.dstBits[r2][idx1]
        return int((v1 << 16) | (v2 & 0xFFFF))

    def setDst32b(self, idx0, idx1, value):
        r1, r2 = self.to_32b_row(idx0)
        self.dstBits[r1][idx1] = value >> 16
        self.dstBits[r2][idx1] = value & 0xFFFF

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
        self.data = np.empty([64, 16], dtype=np.uint32)

    def flipAllowedClient(self):
        if self.allowedClient == SrcRegister.SrcClient.Unpackers:
            self.allowedClient = SrcRegister.SrcClient.MatrixUnit
        else:
            self.allowedClient = SrcRegister.SrcClient.Unpackers

    def getAllowedClient(self):
        return self.allowedClient

    def setAllowedClient(self, c):
        self.allowedClient = c

    def __getitem__(self, key):
        x, y = key
        return int(self.data[x][y])

    def __setitem__(self, key, value):
        x, y = key
        self.data[x][y] = value


class LReg:
    def __init__(self):
        self.read_only = False
        self.hard_wired_value = None
        self.data = [0] * 32

    def __setitem__(self, key, value):
        assert not self.read_only
        self.data[key] = value

    def __getitem__(self, key):
        if self.hard_wired_value is not None:
            return self.hard_wired_value
        else:
            return self.data[key]

    def setReadOnly(self, hard_wired_value=None):
        self.read_only = True
        self.setHardwiredValue(hard_wired_value)

    def setHardwiredValue(self, value):
        self.hard_wired_value = value
