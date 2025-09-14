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
        return self.dst[idx0, idx1]

    def getDst32b(self, idx0, idx1):
        if idx0 * 2 in self.undefined_rows or (idx0 * 2) + 1 in self.undefined_rows:
            return None
        return self.dst[idx0, idx1]

    def setUndefinedRow(self, row, isDst32=False):
        if isDst32:
            self.undefined_rows.append(row * 2)
            self.undefined_rows.append((row * 2) + 1)
        else:
            self.undefined_rows.append(row)
