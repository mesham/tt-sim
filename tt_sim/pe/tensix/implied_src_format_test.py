"""Tests for Blackhole's implied SrcA operand format.

Wormhole's matrix unit always reads the SrcA operand format from
ALU_FORMAT_SPEC_REG0_SrcA. Blackhole instead implies it from the format the
unpacker last wrote the bank in, latched when the bank was handed over, unless
DISABLE_IMPLIED_SRCA_FMT_Base says otherwise (ttsim src/tensix.cpp, guarded on
TT_ARCH_VERSION >= 1).

Runs standalone (``python3 -m tt_sim.pe.tensix.implied_src_format_test``) or
under pytest.
"""

from contextlib import contextmanager

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.backends.backend_base import DataFormat
from tt_sim.pe.tensix.backends.config import TensixConfigurationConstants
from tt_sim.pe.tensix.tensix import TensixCoProcessor


@contextmanager
def _matrix_unit(blackhole):
    """A matrix unit on the given arch.

    The config-register layout is a process-global selection, so restore the
    Wormhole layout on the way out — other tests in the same session (and the
    Wormhole replay guards) expect it.
    """
    try:
        yield (
            TensixCoProcessor(
                None,
                BLACKHOLE_PROFILE.tensix_cfg_state_size if blackhole else None,
                BLACKHOLE_PROFILE.tensix_thd_state_size if blackhole else None,
                blackhole=blackhole,
            )
            .getBackend()
            .matrix_unit
        )
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def test_wormhole_ignores_the_latched_format():
    with _matrix_unit(blackhole=False) as mu:
        mu.backend.getSrcA(mu.srcABank).setDataFormat(DataFormat.TF32)
        assert mu.implied_srcA_format(0, DataFormat.FP32) == DataFormat.FP32


def test_blackhole_takes_the_latched_format():
    with _matrix_unit(blackhole=True) as mu:
        mu.backend.getSrcA(mu.srcABank).setDataFormat(DataFormat.TF32)
        assert mu.implied_srcA_format(0, DataFormat.FP32) == DataFormat.TF32


def test_blackhole_falls_back_before_any_unpack():
    # Nothing has handed a bank over yet, so there is no format to imply.
    with _matrix_unit(blackhole=True) as mu:
        assert mu.implied_srcA_format(0, DataFormat.BF16) == DataFormat.BF16


def test_blackhole_reads_the_bank_the_matrix_unit_is_on():
    with _matrix_unit(blackhole=True) as mu:
        mu.backend.getSrcA(0).setDataFormat(DataFormat.TF32)
        mu.backend.getSrcA(1).setDataFormat(DataFormat.BF16)
        mu.srcABank = 0
        assert mu.implied_srcA_format(0, DataFormat.FP32) == DataFormat.TF32
        mu.srcABank = 1
        assert mu.implied_srcA_format(0, DataFormat.FP32) == DataFormat.BF16


def test_disable_implied_srca_fmt_restores_the_configured_format():
    with _matrix_unit(blackhole=True) as mu:
        mu.backend.getSrcA(mu.srcABank).setDataFormat(DataFormat.TF32)
        addr32 = TensixConfigurationConstants.get_addr32(
            "DISABLE_IMPLIED_SRCA_FMT_Base"
        )
        mu.backend.getConfigUnit().threadConfig[0][addr32] = 1
        assert mu.implied_srcA_format(0, DataFormat.FP32) == DataFormat.FP32


def test_fp32_operand_style_is_tf32_not_bf16_on_blackhole():
    """The bug this models: an FP32 copy keeps 10 mantissa bits, not 7.

    tt-metal leaves ALU_FORMAT_SPEC_REG0_SrcA at FP32 while the unpacker
    converts to TF32 on the way into Src, so reading the configured register
    picks the BF16 operand style and silently narrows every copied datum.
    """
    with _matrix_unit(blackhole=True) as mu:
        mu.backend.getSrcA(mu.srcABank).setDataFormat(DataFormat.TF32)
        srcAStyle, _ = mu.get_dataformat_and_useDst(0, 0)
        assert srcAStyle == DataFormat.TF32

    with _matrix_unit(blackhole=False) as wh:
        srcAStyle, _ = wh.get_dataformat_and_useDst(0, 0)
        assert srcAStyle == DataFormat.BF16


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"{test.__name__} OK")
    print(f"All {len(tests)} implied-src-format tests passed")
