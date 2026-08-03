from enum import IntEnum

import numpy as np

from tt_sim.util.conversion import conv_to_uint32


class DstRegister:
    def __init__(self):
        self.dstBits = np.zeros([1024, 16], dtype=np.uint32)
        # Per-row validity, the hardware's Dst "zero flags" (ttsim
        # ``dst_row_valid``). Reset asserts them all, ZEROACC *clears the flags*
        # rather than the data, and any write re-asserts the flag for the row it
        # touched. A read of a flag-cleared row does not return the underlying
        # bits: every consumer sees zero except the max-pool path (GMPOOL),
        # which sees all-ones — the most negative datum in its comparison
        # order, i.e. minus infinity. That asymmetry is the whole point of the
        # flags: it is how a plain `reduce(max)` over negative data works, and
        # it is why the GMPOOL reads below pass ``isGmpool=True`` and nothing
        # else does (ttsim's ``read_dst16b``/``read_dst32b`` take the same
        # ``is_gmpool`` template flag, set only by ``TENSIX_EXECUTE_GMPOOL``).
        # tt-sim additionally keeps zeroing the backing data on a clear, so the
        # zero every other consumer sees comes out of the array either way.
        self.dstRowValid = np.ones(1024, dtype=bool)
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

    def getDst16b(self, idx0, idx1, isGmpool=False):
        row = self.adj16(idx0)
        if not self.dstRowValid[row]:
            return 0xFFFF if isGmpool else 0
        return int(self.dstBits[row][idx1])

    def setDst16b(self, idx0, idx1, value, validOnLastColumn=False):
        row = self.adj16(idx0)
        self.dstBits[row][idx1] = value
        self._setRowValid(row, idx1, validOnLastColumn)

    def getDst32b(self, idx0, idx1, isGmpool=False):
        br = self.adj32(idx0)
        if not self.dstRowValid[br]:
            return 0xFFFFFFFF if isGmpool else 0
        v1 = self.dstBits[br][idx1]
        v2 = self.dstBits[br + 8][idx1]
        return int((v1 << 16) | (v2 & 0xFFFF))

    def setDst32b(self, idx0, idx1, value, validOnLastColumn=False):
        br = self.adj32(idx0)
        self.dstBits[br][idx1] = value >> 16
        self.dstBits[br + 8][idx1] = value & 0xFFFF
        self._setRowValid(br, idx1, validOnLastColumn)

    def _setRowValid(self, row, column, validOnLastColumn):
        """Re-assert a row's zero flag on write.

        ``validOnLastColumn`` mirrors ttsim's ``set_valid_on_last_column_only``
        template flag: the accumulating writers hold the flag off until the last
        column of the row lands, so a read-modify-write loop over the 16 columns
        of a flag-cleared row keeps seeing it as cleared for the whole pass.
        That only matters where reading a cleared row differs from reading its
        (zeroed) data, i.e. GMPOOL.
        """
        if not validOnLastColumn or column == 15:
            self.dstRowValid[row] = True

    # -- Whole-row accessors -------------------------------------------------
    #
    # The four methods below are the per-datum accessors above applied to a
    # whole row (or a block of rows) at once, so a consumer that touches every
    # column of a row -- the matrix unit reads and writes an entire Dst
    # rectangle per MVMUL -- pays numpy's overhead once instead of 16 times per
    # row. ``rows`` is an array of *unadjusted* row indices, exactly what the
    # scalar ``idx0`` is; the values are int64 blocks of shape
    # ``(len(rows), 16)``. None of them take ``isGmpool`` (GMPOOL walks a column
    # at a time) or ``validOnLastColumn`` (a whole row lands at once, so the
    # flag would be re-asserted either way).

    def getDst16bRows(self, rows):
        rows = self.adj16(np.asarray(rows))
        valid = self.dstRowValid[rows][:, np.newaxis]
        return np.where(valid, self.dstBits[rows], 0).astype(np.int64)

    def setDst16bRows(self, rows, values):
        rows = self.adj16(np.asarray(rows))
        self.dstBits[rows] = values
        self.dstRowValid[rows] = True

    def getDst32bRows(self, rows):
        br = self.adj32(np.asarray(rows))
        hi = self.dstBits[br].astype(np.int64)
        lo = self.dstBits[br + 8].astype(np.int64)
        valid = self.dstRowValid[br][:, np.newaxis]
        return np.where(valid, (hi << 16) | (lo & 0xFFFF), 0)

    def setDst32bRows(self, rows, values):
        br = self.adj32(np.asarray(rows))
        self.dstBits[br] = values >> 16
        self.dstBits[br + 8] = values & 0xFFFF
        self.dstRowValid[br] = True

    def setUndefinedRow(self, row, isDst32=False):
        # ZEROACC clears a row's zero flag, it does not touch the data. Every
        # consumer but GMPOOL then reads the row as zero, so tt-sim also zeroes
        # the backing store: that keeps the array agreeing with the flag and
        # keeps the paths that rely on reading a cleared row as +0 (e.g. the
        # FP32 copy_tile's zeroed SrcB operand) working unchanged. The 32-bit
        # clear therefore names the pair ``getDst32b`` actually reads --
        # ``Adj32(row)`` and ``+8``, ttsim's ``dst32b_adjust_row`` -- rather than
        # ``row*2``/``row*2+1``, which covered the same 32 backing rows for the
        # aligned 16-row block ZEROACC uses but the wrong two for a single row.
        if isDst32:
            br = self.adj32(row)
            self.dstRowValid[br] = False
            self.dstBits[br, :] = 0
            self.dstBits[br + 8, :] = 0
        else:
            row = self.adj16(row)
            self.dstRowValid[row] = False
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

    def readRows(self, row, numRows):
        """The ``numRows`` x 16 block starting at ``row``, as int64.

        The conversions out of the Src storage layout
        (``DataFormatConversions``' ``*InSrcTo*`` family) are pure bit
        arithmetic, so the matrix unit converts a whole block in one numpy call
        rather than one datum at a time; int64 is what keeps the shifts they do
        from wrapping (Src holds 19 bits, FP32 takes 32). Out-of-range rows
        raise, as indexing a row past the end of a bank does -- the wrap
        hardware applies to a 6-bit row index is not modelled here either way.
        """
        block = self.data[row : row + numRows]
        if block.shape[0] != numRows:
            raise IndexError(
                f"Src rows [{row}, {row + numRows}) exceed the 64-row bank"
            )
        return block.astype(np.int64)

    def writeDatums(self, rows, columns, values):
        """The per-datum ``__setitem__`` applied to a whole block of datums.

        ``rows`` and ``columns`` are index arrays broadcasting together to the
        shape of ``values`` -- each broadcast pair is exactly the ``[x, y]`` key
        the scalar setter takes -- so a caller expresses whatever row/column map
        it has (the unpacker's haloize transpose included) and pays numpy's
        overhead once per block instead of once per datum.
        Indexing follows numpy's rules just as the scalar setter's does: a
        negative column wraps to the end of the row -- silently clobbering the
        far end of the row rather than dropping the datum, which is why the
        unpacker rejects the one mode that would produce one (ColShift) rather
        than computing it -- and a row past the end of the bank raises. The
        caller owns injectivity of the map: with a repeated (row, column) pair
        this writes one of the values rather than the scalar loop's explicit
        last-one-wins.
        """
        self.data[rows, columns] = values


class LReg:
    """One SFPU LReg (32 lanes).

    Every lane holds a **uint32 bit pattern**, matching ttsim's ``l_regs`` — a
    float value is stored as its FP32 bits and integer/bitwise results as their
    bits, and each op reads with the right accessor (``conv_to_float`` for the
    float value, the raw int for bits). This is what keeps the SFPU pipeline
    bit-exact with the reference on both architectures.

    tt-sim used to keep a *mixed* model on Wormhole, where a lane could hold a
    Python float instead. That silently destroyed information the SFPU relies
    on: NaN payloads (the accurate FP32 exp builds ``2**n`` through an integer
    ``0x7ffffffd``-shaped lane, which becomes a payload-less ``nan`` the moment
    it is stored as a Python float), and it made the *integer* ops (SFPIADD,
    SFPSHFT) do float arithmetic whenever their operand happened to have been
    written by a float op. It also carried full double precision between ops
    instead of rounding to FP32 each time. See docs/plans/blackhole-support.md.
    """

    def __init__(self, blackhole=False):
        # ``blackhole`` no longer selects a value model (both are uint32); it is
        # kept because callers construct LRegs per-arch and future per-arch
        # register behaviour would land here.
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
        self.data[key] = self._coerce(value)

    def __getitem__(self, key):
        if self.hard_wired_value is not None:
            return self.hard_wired_value
        else:
            return self.data[key]

    def setReadOnly(self, hard_wired_value=None):
        self.read_only = True
        self.setHardwiredValue(hard_wired_value)

    def setHardwiredValue(self, value):
        if value is not None:
            value = self._coerce(value)
        self.hard_wired_value = value
