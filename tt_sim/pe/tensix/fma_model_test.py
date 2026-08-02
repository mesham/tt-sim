"""Pin the SFPU float ALU against ttsim, on both architectures.

``fma_model_wh`` / ``fma_model_bh`` (``backends/vector.py``) are ports of
ttsim's ``src/fma.cpp``: the exact hardware ``x*y + z`` every SFPU float
multiply/add rounds through (denormal flushing, three guard bits,
round-to-nearest-even). They are *different silicon*, not one model with a
flag — ttsim picks between them with ``#define fma_model fma_model_{wh,bh}``
— and they disagree on ~10% of random inputs:

- NaN: Blackhole returns the canonical ``0x7fc00000`` immediately, Wormhole
  keeps computing with a ``0x7f800001`` "nan_result" so mantissa bits of the
  real result leak into what comes back.
- Underflow: an underflowing product returns ``+0`` on Wormhole where
  Blackhole returns ``z_sign & p_sign``, and a denormal result is flushed
  discarding the sign rather than renormalised.
- Rounding: the final sticky is ``r_m & 1`` on Wormhole against Blackhole's
  ``(r_m & (n | 1)) != 0``.

The vectors below come from compiling ttsim's C standalone and dumping both
models over the same inputs; that harness also fuzz-matched both ports on 200k
random triples (19934 of which exercised the Wormhole/Blackhole difference),
with zero mismatches.

Runs standalone (``python3 -m tt_sim.pe.tensix.fma_model_test``) or under
pytest.
"""

from tt_sim.pe.tensix.backends.vector import fma_model_bh, fma_model_wh

# (x, y, z, expected Wormhole, expected Blackhole).
FMA_VECTORS = [
    # --- the two models agree: ordinary values, overflow to Inf, denormal in
    (0x4FBC64B3, 0xBF604992, 0x40000000, 0xCFA50E41, 0xCFA50E41),
    (0xC25BEBF6, 0x961BC602, 0x007FFFFF, 0x1905D1F8, 0x1905D1F8),
    (0x4E7C7C81, 0x323EAEC4, 0x43BD5B97, 0x43C33C1D, 0x43C33C1D),
    (0xC366C8B9, 0x42755960, 0xF0B13873, 0xF0B13873, 0xF0B13873),
    (0xBFE05F91, 0xBF800000, 0xCDC86E11, 0xCDC86E11, 0xCDC86E11),
    (0x6D46C41E, 0x4B800000, 0x5A304DC9, 0x7946C41E, 0x7946C41E),
    (0x450B24C4, 0x7E800000, 0xDBE70C81, 0x7F800000, 0x7F800000),
    (0xF1AD093E, 0x5EC75388, 0xC09BBDEA, 0xFF800000, 0xFF800000),
    (0x4B800000, 0x3849C313, 0x4A47223E, 0x4A472EDA, 0x4A472EDA),
    (0x3F3B74A3, 0x44A328C7, 0xBE9A81A4, 0x446EDEEF, 0x446EDEEF),
    # --- NaN handling: Wormhole leaks result bits into its NaN, Blackhole
    #     returns the canonical quiet NaN.
    (0x3F800000, 0xFFC00000, 0x3F7FFFFF, 0xFF800001, 0x7FC00000),
    (0x7FC00000, 0x4EC9A5D3, 0x46CACEA3, 0x7F800001, 0x7FC00000),
    (0x3F3E69F0, 0x42EB17C2, 0x7FC00000, 0x7F800001, 0x7FC00000),
    (0x23D9C929, 0x7F800001, 0xCC37DC7D, 0x7FD9C92B, 0x7FC00000),
    (0xFF800000, 0xBF10BBE4, 0xFFC00000, 0xFFEF441D, 0x7FC00000),
    (0xFF7FFFFF, 0xFFC00000, 0xFF800000, 0x7F800001, 0x7FC00000),
    # --- underflow / denormal flush: Wormhole discards the sign, Blackhole
    #     keeps it (and keeps a denormal addend).
    (0xB776ED18, 0x00800000, 0x80400000, 0x00000000, 0x80000000),
    (0x42C58CBE, 0x80400000, 0x80000000, 0x00000000, 0x80000000),
    (0x00000000, 0xB3C26FC1, 0x804D9049, 0x00000000, 0x80000000),
    (0x3F7FFFFF, 0x00800000, 0x007FFFFF, 0x00000000, 0x00800000),
    (0xC19AC891, 0x00000001, 0x80000000, 0x00000000, 0x80000000),
    (0x007FFFFF, 0x80400000, 0x80000000, 0x00000000, 0x80000000),
]


def test_fma_models_match_ttsim():
    for x, y, z, wh, bh in FMA_VECTORS:
        assert fma_model_wh(x, y, z) == wh, (
            f"wh fma({x:#010x}, {y:#010x}, {z:#010x}) = "
            f"{fma_model_wh(x, y, z):#010x}, expected {wh:#010x}"
        )
        assert fma_model_bh(x, y, z) == bh, (
            f"bh fma({x:#010x}, {y:#010x}, {z:#010x}) = "
            f"{fma_model_bh(x, y, z):#010x}, expected {bh:#010x}"
        )


def test_the_two_models_really_differ():
    """Guard against one arch quietly being wired to the other's model."""
    differing = [v for v in FMA_VECTORS if v[3] != v[4]]
    assert len(differing) >= 10


def test_simple_identities():
    """Sanity: the models are a plain a*b+c on values with no rounding to do."""
    for fma in (fma_model_wh, fma_model_bh):
        assert fma(0x40000000, 0x40400000, 0x3F800000) == 0x40E00000  # 2*3+1 = 7
        assert fma(0x3F800000, 0x3F800000, 0x00000000) == 0x3F800000  # 1*1+0 = 1
        assert fma(0xBF800000, 0x3F800000, 0x40000000) == 0x3F800000  # -1*1+2 = 1


if __name__ == "__main__":
    test_fma_models_match_ttsim()
    test_the_two_models_really_differ()
    test_simple_identities()
    print("fma_model tests OK")
