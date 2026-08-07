"""The compiled kernel behind :mod:`tt_sim.pe.tensix.backends.fpu_jit`.

Imported lazily and only when numba is present -- importing this module *is*
the numba dependency, which is why it is a module of its own rather than a
guarded block inside ``fpu_jit``. See that module for why this exists, what it
costs and how it is tested.

The body is a line-by-line transliteration of ``MatrixUnit``'s
``_fpu_group_sums_batch`` and ``_fpu_accumulate_batch`` with the numpy
operations replaced by scalar arithmetic over the same indices, and it must
stay one: those two are the reference, they are themselves pinned against the
scalar port of ttsim's C, and ``fpu_accumulate_test.py`` fuzzes this kernel
against them. Every intermediate is int64, for the reasons and within the
bounds the two docstrings there record.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, fastmath=False)
def mvmul_fused(
    srcA, srcB, dstVals, fidelityPhase, expProdAdj, useDst32b, negOneRenormBug
):
    """srcA (16, C) lanes x columns; srcB (16, RB) lanes x rows (RB == 1 or R);
    dstVals (R, C). Returns (R, C) FP32 bit patterns."""
    R, C = dstVals.shape
    RB = srcB.shape[1]
    out = np.empty((R, C), dtype=np.int64)
    zeroExp = expProdAdj if expProdAdj > 0 else 0

    gSignA = np.empty(2, dtype=np.int64)
    gExpA = np.empty(2, dtype=np.int64)
    gManA = np.empty(2, dtype=np.int64)
    manv = np.empty(2, dtype=np.int64)

    for r in range(R):
        rb = 0 if RB == 1 else r
        for c in range(C):
            # ---- group sums ------------------------------------------------
            for g in range(2):
                expSum = -(1 << 62)
                for k in range(8):
                    lane = g * 8 + k
                    a = srcA[lane, c]
                    b = srcB[lane, rb]
                    if (a & 0x7F800000) != 0 and (b & 0x7F800000) != 0:
                        e = ((a >> 23) & 0xFF) + ((b >> 23) & 0xFF) + expProdAdj
                    else:
                        e = zeroExp
                    if e > expSum:
                        expSum = e
                manSum = 0
                for k in range(8):
                    lane = g * 8 + k
                    a = srcA[lane, c]
                    b = srcB[lane, rb]
                    nz = (a & 0x7F800000) != 0 and (b & 0x7F800000) != 0
                    mA = ((a >> 13) & 0x3FF) | 0x400
                    mB = ((b >> 13) & 0x3FF) | 0x400
                    if fidelityPhase & 1:
                        mA = (mA >> 1) & 0x1F
                    else:
                        mA = mA >> 6
                    if fidelityPhase & 2:
                        mB = (mB & 0xF) << 3
                    else:
                        mB = mB >> 4
                    if nz:
                        sgn = (a >> 31) ^ (b >> 31)
                        e = ((a >> 23) & 0xFF) + ((b >> 23) & 0xFF) + expProdAdj
                        m = mA * mB
                    else:
                        sgn = 0
                        e = zeroExp
                        m = 0
                    sh = expSum - e
                    if sh > 30:
                        sh = 30
                    aligned = ((m << 1) + (1 << sh)) >> (sh + 1)
                    if sgn != 0:
                        manSum -= aligned
                    else:
                        manSum += aligned
                if expSum > 0:
                    gSignA[g] = 1 if manSum < 0 else 0
                    gExpA[g] = expSum
                    gManA[g] = (manSum if manSum >= 0 else -manSum) << 13
                else:
                    gSignA[g] = 0
                    gExpA[g] = 0
                    gManA[g] = 0

            # ---- accumulate -------------------------------------------------
            dv = dstVals[r, c]
            signDst = dv >> 31
            expDst = (dv >> 23) & 0xFF
            manDst = ((dv & 0x7FFFFF) | 0x800000) if expDst != 0 else 0

            exp = gExpA[0]
            if gExpA[1] > exp:
                exp = gExpA[1]
            if expDst > exp:
                exp = expDst

            for g in range(2):
                sh = exp - gExpA[g]
                ss = sh
                if ss < 1:
                    ss = 1
                if ss > 30:
                    ss = 30
                rounded = (gManA[g] + (1 << (ss - 1)) - gSignA[g]) >> ss
                if sh == 0:
                    m = gManA[g]
                elif sh < 31:
                    m = rounded
                else:
                    m = 0
                if not useDst32b:
                    m = (m + (1 << 12) - gSignA[g]) & ~0x1FFF
                manv[g] = m

            shD = exp - expDst
            ssD = shD
            if ssD < 1:
                ssD = 1
            if ssD > 30:
                ssD = 30
            roundedDst = (manDst + (1 << (ssD - 1))) >> ssD
            if shD == 0:
                pass
            elif shD < 31:
                manDst = roundedDst
            else:
                manDst = 0
            if not useDst32b:
                manDst = (manDst + (1 << 12)) & ~0x1FFF

            s0 = manv[0] if gSignA[0] == signDst else -manv[0]
            s1 = manv[1] if gSignA[1] == signDst else -manv[1]
            acc = manDst + s0 + s1
            sign = (signDst ^ 1) if acc < 0 else signDst
            man = acc if acc >= 0 else -acc

            bl = 0
            t = man
            while t:
                bl += 1
                t >>= 1
            sh = (32 - bl) - 8
            if negOneRenormBug and sign != 0 and man == 1:
                sh -= 27
            expOut = exp - sh
            if not useDst32b:
                sh -= 16
            rs = -sh
            if rs < 1:
                rs = 1
            if rs > 30:
                rs = 30
            if sh < 0:
                man = (man + (1 << (rs - 1))) >> rs
            else:
                man = man << sh
            if not useDst32b:
                man = man << 16
            if (man & 0x1000000) != 0:
                expOut += 1
            man = man & 0x7FFFFF

            if exp <= 0 or acc == 0 or expOut <= 0:
                res = 0
            elif expOut >= 255:
                res = (sign << 31) | (255 << 23)
            else:
                eo = expOut
                if eo < 0:
                    eo = 0
                if eo > 254:
                    eo = 254
                res = (sign << 31) | (eo << 23) | man
            out[r, c] = res
    return out
