from enum import IntEnum


class DstRegister:
    def __init__(self):
        self.dstBits = [[0 for _ in range(16)] for _ in range(1024)]
        self.undefined_rows = []

    def getDst16b(self, idx0, idx1):
        if idx0 in self.undefined_rows:
            return None
        return self.dstBits[idx0][idx1]

    def setDst16b(self, idx0, idx1, value):
        if idx0 in self.undefined_rows:
            self.undefined_rows.remove(idx0)
        self.dstBits[idx0][idx1] = value

    def to_32b_row(self, r_16b):
        br = ((r_16b & 0x1F8) << 1) | (r_16b & 0x207)
        return br, br + 8

    def getDst32b(self, idx0, idx1):
        r1, r2 = self.to_32b_row(idx0)
        if r1 in self.undefined_rows or r2 in self.undefined_rows:
            return None

        v1 = self.dstBits[r1][idx1]
        v2 = self.dstBits[r2][idx1]

        return (v1 << 16) | (v2 & 0x00FF)

    def setDst32b(self, idx0, idx1, value):
        r1, r2 = self.to_32b_row(idx0)
        if r1 in self.undefined_rows:
            self.undefined_rows.remove(r1)
        if r2 in self.undefined_rows:
            self.undefined_rows.remove(r2)

        v1 = value >> 16
        v2 = value & 0x00FF

        self.dstBits[r1][idx1] = v1
        self.dstBits[r2][idx1] = v2

    def setUndefinedRow(self, row, isDst32=False):
        if isDst32:
            if row * 2 not in self.undefined_rows:
                self.undefined_rows.append(row * 2)
            if (row * 2) + 1 not in self.undefined_rows:
                self.undefined_rows.append((row * 2) + 1)
        else:
            if row not in self.undefined_rows:
                self.undefined_rows.append(row)


class SrcRegister:
    class SrcClient(IntEnum):
        MatrixUnit = 0
        Unpackers = 1

    def __init__(self):
        self.allowedClient = SrcRegister.SrcClient.Unpackers
        self.data = [[0 for _ in range(16)] for _ in range(64)]

    def flipAllowedClient(self):
        if self.allowedClient == SrcRegister.SrcClient.Unpackers:
            self.allowedClient = SrcRegister.SrcClient.MatrixUnit
        else:
            self.allowedClient = SrcRegister.SrcClient.Unpackers

    def getAllowedClient(self):
        return self.allowedClient

    def __getitem__(self, key):
        x, y = key
        return self.data[x][y]

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
