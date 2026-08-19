"""Tests for how wide the packer reads Dst: ``PCK_DEST_RD_CTRL_Read_32b_data``.

The width of a Dst read is decided by that one config register, and *not* by the
pack source format. tt-metal's ``_llk_pack_hw_configure_`` /
``reconfig_packer_data_format`` set it to
``is_32b_format || is_fp32_dest_acc_en``, so a 32-bit ``In_data_format`` implies
it, but ``ComputeConfig{.fp32_dest_acc_en = true}`` over **Float16_b** circular
buffers -- the ordinary bf16-storage/fp32-accumulate GEMM -- sets it with a
16-bit pack source format. Inferring the width from the format instead read the
high half of every other Dst row and left most of the tile unwritten: a
consumer's ``gemm_bf16_check`` came back ``errors=4096 of 4096`` on tt-sim while
both cards passed it.

The reads below all come from Dst row 8, deliberately: ``Adj32`` folds a 32-bit
Dst row ``r`` onto 16-bit rows ``2r`` and ``2r + 8``, so for rows 0..7 a 16-bit
read happens to land on the fp32 datum's high half -- which is bit-identical to
the bf16 encoding -- and the bug is invisible. Row 8 is the first row where the
two disagree.

Runs standalone (``python3 -m tt_sim.pe.tensix.pack_dest_rd_ctrl_test``) or
under pytest.
"""

from contextlib import contextmanager

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import DataFormatConversions, TensixConfigurationConstants

BF16 = 5  # DataFormat.BF16
FP32 = 0  # DataFormat.FP32
#: L1_Dest_addr, in 16-byte units, and the byte address it names.
L1_DEST_ADDR = 0x100
OUT_BYTE_ADDR = L1_DEST_ADDR << 4
#: Datums in one Dst row, and the row the pack reads (see the module docstring).
ROW = 16
SRC_ROW = 8

#: ``(fp32 bit pattern, the bf16 the packer must emit for it)``. The packer
#: rounds to nearest on the way out of a 32-bit Dst -- ttsim's
#: ``(value + 0x8000) >> 16`` -- saturates NaN to infinity and flushes a result
#: that has come out denormal to +0, so several of these differ from a plain
#: truncation of the top 16 bits.
PATTERNS = (
    (0x3F800000, 0x3F80),  # 1.0, exact in bf16
    (0xC0200000, 0xC020),  # -2.5, exact in bf16
    (0x3F80C000, 0x3F81),  # rounds up (truncating would give 0x3F80)
    (0x3F817FFF, 0x3F81),  # rounds down
    (0xBF80C000, 0xBF81),  # negative, rounds away from zero
    (0x40490FDB, 0x4049),  # pi
    (0x00000000, 0x0000),  # +0
    (0x80000000, 0x0000),  # -0 flushes to +0 (the sign is dropped)
    (0x3E800000, 0x3E80),  # 0.25
    (0x41200000, 0x4120),  # 10.0
    (0xC1200000, 0xC120),  # -10.0
    (0x7F7FFFFF, 0x7F80),  # max normal, rounds up to infinity
    (0x7FC00000, 0x7F80),  # NaN saturates to infinity
    (0xFF800000, 0xFF80),  # -infinity
    (0x00004000, 0x0000),  # fp32 denormal, flushed to +0
    (0x80004000, 0x0000),  # negative denormal, flushed to +0 (sign dropped)
)
assert len(PATTERNS) == ROW


class _L1:
    """Addressable memory that remembers every byte written."""

    def __init__(self, size=1 << 20):
        self.data = bytearray(size)
        self.written = set()

    def read(self, addr, size):
        return bytes(self.data[addr : addr + size])

    def write(self, addr, payload):
        self.data[addr : addr + len(payload)] = payload
        self.written.update(range(addr, addr + len(payload)))


def _set_config(backend, key, value, state_id=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


@contextmanager
def _packer(blackhole, in_format, out_format, read_32b):
    """A backend set up to pack Dst row ``SRC_ROW`` to L1, one row per PACR.

    The config-register layout is a process-global selection, so restore the
    Wormhole layout on the way out.
    """
    try:
        coprocessor = TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size if blackhole else None,
            BLACKHOLE_PROFILE.tensix_thd_state_size if blackhole else None,
            blackhole=blackhole,
        )
        backend = coprocessor.getBackend()
        memory = _L1()
        backend.setAddressableMemory(memory)

        _set_config(backend, "THCON_SEC0_REG1_L1_Dest_addr", L1_DEST_ADDR)
        _set_config(backend, "THCON_SEC0_REG1_Sub_l1_tile_header_size", 1)
        _set_config(backend, "THCON_SEC0_REG1_In_data_format", in_format)
        _set_config(backend, "THCON_SEC0_REG1_Out_data_format", out_format)
        # Inverted field: 1 means zero compression is *disabled*, as tt-metal
        # sets it for every packer it configures.
        _set_config(backend, "THCON_SEC0_REG1_Disable_zero_compress", 1)
        _set_config(backend, "PCK_EDGE_OFFSET_SEC0_mask", 0xFFFF)
        _set_config(backend, "PACK_COUNTERS_SEC0_pack_reads_per_xy_plane", ROW)
        # The register under test. Left at its reset zero when read_32b is
        # false, which is what every 16-bit-Dst program runs with.
        if read_32b:
            _set_config(backend, "PCK_DEST_RD_CTRL_Read_32b_data", 1)
        # Read from Dst row SRC_ROW: the offset is in units of 16 datums.
        _set_config(backend, "DEST_TARGET_REG_CFG_PACK_SEC0_Offset", SRC_ROW)

        # ADC channel 1 X is the last input datum index (inclusive): one row.
        backend.getADC(0).Packers.Channel[1].X = ROW - 1

        yield backend, memory
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _pacr(backend):
    """One PACR over a single Dst read interface, so both architectures pack
    exactly the 16 datums of ``SRC_ROW`` (``PackSel``/``read_intf_sel`` = 1)."""
    backend.packer_unit.handle_pacr(
        {"raw_instruction": 0x41 << 24},
        0,
        {
            "Last": 1,
            "Flush": 0,
            "OvrdThreadId": 0,
            "PackSel": 1,
            "ZeroWrite": 0,
            "AddrMode": 0,
        },
    )


def _packed(memory, count, width=2):
    return [
        int.from_bytes(memory.read(OUT_BYTE_ADDR + width * i, width), "little")
        for i in range(count)
    ]


@pytest.mark.parametrize("blackhole", [False, True])
def test_32b_dst_read_with_a_16b_pack_format(blackhole):
    """fp32_dest_acc_en over bf16 CBs: 32-bit Dst reads, 16-bit output.

    Every datum has to survive, narrowed to bf16. Reading Dst 16 bits at a time
    here -- what inferring the width from ``In_data_format`` did -- returns the
    *low* half of the fp32 datums of row 0 for this row, which is zero, so the
    whole row packs as zeroes.
    """
    with _packer(blackhole, BF16, BF16, read_32b=True) as (backend, memory):
        for col, (fp32, _) in enumerate(PATTERNS):
            backend.getDst().setDst32b(
                SRC_ROW, col, DataFormatConversions.FP32ToDstFormatFP32(fp32)
            )
        _pacr(backend)
        assert _packed(memory, ROW) == [bf16 for _, bf16 in PATTERNS]
        assert memory.written == set(range(OUT_BYTE_ADDR, OUT_BYTE_ADDR + 2 * ROW))


@pytest.mark.parametrize("blackhole", [False, True])
def test_16b_dst_read_when_the_register_is_clear(blackhole):
    """The default: register at its reset zero, so Dst is read 16 bits wide.

    This is every program that does not set ``fp32_dest_acc_en``, and it must
    be untouched by the width fix -- including that a *cleared* register reads
    16 bits whatever else is configured.
    """
    with _packer(blackhole, BF16, BF16, read_32b=False) as (backend, memory):
        expected = []
        for col, (fp32, _) in enumerate(PATTERNS):
            bf16 = fp32 >> 16
            backend.getDst().setDst16b(
                SRC_ROW, col, DataFormatConversions.BF16ToDstFormatBF16(bf16)
            )
            expected.append(bf16)
        _pacr(backend)
        assert _packed(memory, ROW) == expected


@pytest.mark.parametrize("blackhole", [False, True])
def test_32b_dst_read_with_a_32b_pack_format(blackhole):
    """Float32 circular buffers: unchanged, and now for the documented reason.

    tt-metal sets the register for a 32-bit ``In_data_format`` too
    (``is_32b_format || is_fp32_dest_acc_en``), so reading the width off the
    register rather than off the format leaves this path packing exactly the
    fp32 datums it always did.
    """
    with _packer(blackhole, FP32, FP32, read_32b=True) as (backend, memory):
        for col, (fp32, _) in enumerate(PATTERNS):
            backend.getDst().setDst32b(
                SRC_ROW, col, DataFormatConversions.FP32ToDstFormatFP32(fp32)
            )
        _pacr(backend)
        assert _packed(memory, ROW, width=4) == [fp32 for fp32, _ in PATTERNS]


def test_unmodelled_32b_read_format_raises():
    """A 32-bit read under a source format whose narrowing is not modelled --
    fp16, which needs the Blackhole-only ``Round_10b_mant`` path -- is refused
    rather than silently packing a bf16-shaped datum."""
    with _packer(blackhole=False, in_format=1, out_format=1, read_32b=True) as (
        backend,
        _,
    ):
        with pytest.raises(NotImplementedError, match="32-bit Dst read"):
            _pacr(backend)


def test_fp32_in_dst_to_bf16_rounds():
    """The narrowing itself, independent of the packer."""
    for fp32, bf16 in PATTERNS:
        dst = DataFormatConversions.FP32ToDstFormatFP32(fp32)
        assert DataFormatConversions.FP32InDstToBF16(dst) == bf16, hex(fp32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
