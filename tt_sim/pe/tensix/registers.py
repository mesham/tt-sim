from enum import IntEnum

import numpy as np


class DstRegister:
    def __init__(self):
        buf = bytearray(32768)
        self.dst16 = np.ndarray([1024, 16], dtype=np.uint16, buffer=buf)
        self.dst32 = np.ndarray([512, 16], dtype=np.uint32, buffer=buf)
        self.undefined_rows = []

    def getDst16b(self, idx0, idx1):
        if idx0 in self.undefined_rows:
            return None
        return self.dst16[idx0, idx1]

    def getDst32b(self, idx0, idx1):
        if idx0 * 2 in self.undefined_rows or (idx0 * 2) + 1 in self.undefined_rows:
            return None
        return self.dst32[idx0, idx1]

    def setDst16b(self, idx0, idx1, value):
        if idx0 in self.undefined_rows:
            self.undefined_rows.remove(idx0)
        self.dst16[idx0, idx1] = value

    def setDst32b(self, idx0, idx1, value):
        if idx0 * 2 in self.undefined_rows:
            self.undefined_rows.remove(idx0 * 2)
        if (idx0 * 2) + 1 in self.undefined_rows:
            self.undefined_rows.remove((idx0 * 2) + 1)
        self.dst32[idx0, idx1] = value

    def setUndefinedRow(self, row, isDst32=False):
        if isDst32:
            self.undefined_rows.append(row * 2)
            self.undefined_rows.append((row * 2) + 1)
        else:
            self.undefined_rows.append(row)


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

    def __getitem__(self, key):
        x, y = key
        return self.data[x, y]

    def __setitem__(self, key, value):
        x, y = key
        self.data[x, y] = value
