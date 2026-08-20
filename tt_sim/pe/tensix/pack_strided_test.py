"""Tests for Blackhole PACR's ``dst_access_mode`` (``DST_ACCESS_STRIDED_MODE``).

Blackhole's single packer gathers 16 datums per selected Dst *read interface*.
In the default ``DST_ACCESS_NORMAL_MODE`` interface ``k`` reads Dst row
``base + k`` -- successive rows of one 16-row face. In
``DST_ACCESS_STRIDED_MODE`` it reads row ``base + 16*k`` instead -- the *same*
row of successive faces -- which is what turns a row-major
``pack_untilize_dest`` into a plain pack: each PACR emits one 32-datum output
row spanning a face pair, and the MOP steps ``base`` down the 16 rows of the
face.

Two decode gaps stood between tt-sim and that behaviour, and both are pinned
here:

* ``dst_access_mode`` (raw bit 17) is absent from the shared, Wormhole-layout
  PACR argument table, which stops at ``AddrMode``;
* the same table gives ``AddrMode`` every bit up to the opcode (23:15), because
  it is the last argument. Blackhole packs ``dst_access_mode``,
  ``row_pad_zero`` and ``cfg_context`` into that span, so a strided PACR's
  ``ADDR_MOD_1`` decoded as section 5 -- one no LLK programs -- and the pack Y
  counter silently stopped advancing.

The stride is applied to *logical* Dst rows: tt-sim (like hardware, unlike
ttsim) puts every Dst access through ``Adj16``/``Adj32``, whose gates this now
drives from ``DEST_ACCESS_CFG``. ``registers_test.py`` pins the transforms
themselves; ``test_dest_access_cfg_write_sets_the_dst_gates`` below pins the
plumbing.

Runs standalone (``python3 -m tt_sim.pe.tensix.pack_strided_test``) or under
pytest.
"""

from contextlib import contextmanager

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import (
    DataFormatConversions,
    TensixConfigurationConstants,
    TensixInstructionDecoder,
)
from tt_sim.util.bits import get_nth_bit

BF16 = 5  # DataFormat.BF16
#: L1_Dest_addr, in 16-byte units, and the byte address it names.
L1_DEST_ADDR = 0x100
OUT_BYTE_ADDR = L1_DEST_ADDR << 4
#: Datums one Dst read interface supplies per PACR.
ROW = 16
#: Dst rows a strided read steps between interfaces.
STRIDE = 16
#: Enough Dst rows for two 16-row faces plus a couple of base offsets.
DST_ROWS = 64


class _L1:
    def __init__(self, size=1 << 20):
        self.data = bytearray(size)

    def read(self, addr, size):
        return bytes(self.data[addr : addr + size])

    def write(self, addr, payload):
        self.data[addr : addr + len(payload)] = payload


def _set_config(backend, key, value, state_id=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


def _dst_value(row, col):
    """A distinct bf16 bit pattern per (row, column), as Dst stores it."""
    return ((row * ROW + col + 1) << 8) | 0x3F


def _pacr_word(
    addr_mode=0,
    dst_access_mode=0,
    read_intf_sel=0,
    last=1,
    ctxt_ctrl=0,
    addr_cnt_context=0,
    row_pad_zero=0,
    cfg_context=0,
):
    """A Blackhole PACR, encoded exactly as ``TT_OP_PACR`` does.

    ``ckernel_ops.h``: ``cfg_context << 21 | row_pad_zero << 18 |
    dst_access_mode << 17 | addr_mode << 15 | addr_cnt_context << 13 |
    zero_write << 12 | read_intf_sel << 8 | ovrd_thread_id << 7 | concat << 4 |
    ctxt_ctrl << 2 | flush << 1 | last``.
    """
    return (
        (0x41 << 24)
        | (cfg_context << 21)
        | (row_pad_zero << 18)
        | (dst_access_mode << 17)
        | (addr_mode << 15)
        | (addr_cnt_context << 13)
        | (read_intf_sel << 8)
        | (ctxt_ctrl << 2)
        | last
    )


@contextmanager
def _packer(blackhole=True, remap=True):
    """A backend set up for a BF16 -> BF16 Dst -> L1 pack.

    ``remap`` drives the real ``DEST_ACCESS_CFG`` register rather than poking
    ``DstRegister``, so the config plumbing is on the path under test. The
    config-register layout is a process-global selection, so restore the
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
        _set_config(backend, "THCON_SEC0_REG1_In_data_format", BF16)
        _set_config(backend, "THCON_SEC0_REG1_Out_data_format", BF16)
        _set_config(backend, "THCON_SEC0_REG1_Disable_zero_compress", 1)
        _set_config(backend, "PCK_EDGE_OFFSET_SEC0_mask", 0xFFFF)
        _set_config(backend, "PACK_COUNTERS_SEC0_pack_reads_per_xy_plane", ROW)
        if blackhole and remap:
            # What tt-metal's _llk_math_reconfig_remap_ writes before a
            # pack_untilize: both bits, together.
            _set_config(backend, "DEST_ACCESS_CFG_remap_addrs", 1)
            _set_config(backend, "DEST_ACCESS_CFG_swizzle_32b", 1)

        # ADC channel 1 X is the last input datum index (inclusive): one row.
        backend.getADC(0).Packers.Channel[1].X = ROW - 1

        for row in range(DST_ROWS):
            for col in range(ROW):
                backend.getDst().setDst16b(row, col, _dst_value(row, col))

        yield backend, memory
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _decode(word):
    """The arguments the shared (Wormhole-layout) table produces for ``word``.

    The real decoder, not a hand-rolled stand-in: each argument is handed every
    bit up to the next one's ``start_bit``, so ``Flush`` runs 3:1, ``ZeroWrite``
    12:14 and ``AddrMode`` -- being last -- 23:15. Blackhole's extra fields all
    land inside those spans, which is exactly what these tests are about.
    """
    return TensixInstructionDecoder.getInstructionInfo(word)["instr_args"]


def _pacr(backend, **kwargs):
    word = _pacr_word(**kwargs)
    backend.packer_unit.handle_pacr({"raw_instruction": word}, 0, _decode(word))


def _packed(memory, count):
    return [
        int.from_bytes(memory.read(OUT_BYTE_ADDR + 2 * i, 2), "little")
        for i in range(count)
    ]


def _expected(rows):
    return [
        DataFormatConversions.BF16InDstToBF16(_dst_value(row, col))
        for row in rows
        for col in range(ROW)
    ]


@pytest.mark.parametrize(
    "read_intf_sel, rows",
    [
        (0x1, [0]),  # a single interface: unchanged by the mode
        (0x3, [0, STRIDE]),  # pack_untilize's pair: one row of two faces
        (0x0, [0, STRIDE, 2 * STRIDE, 3 * STRIDE]),  # all four interfaces
    ],
)
def test_strided_pacr_reads_rows_a_face_apart(read_intf_sel, rows):
    with _packer() as (backend, memory):
        _pacr(backend, dst_access_mode=1, read_intf_sel=read_intf_sel)
        assert _packed(memory, ROW * len(rows)) == _expected(rows)


@pytest.mark.parametrize(
    "read_intf_sel, rows",
    [(0x1, [0]), (0x3, [0, 1]), (0x0, [0, 1, 2, 3])],
)
def test_normal_mode_still_reads_contiguous_rows(read_intf_sel, rows):
    """The mode is opt-in: with the bit clear the interfaces step by one row."""
    with _packer() as (backend, memory):
        _pacr(backend, dst_access_mode=0, read_intf_sel=read_intf_sel)
        assert _packed(memory, ROW * len(rows)) == _expected(rows)


def test_wormhole_ignores_bit_17():
    """Wormhole has no ``dst_access_mode``: bit 17 is not part of any field.

    The same encoded word packs contiguous rows here, which is also what pins
    the ``AddrMode`` re-read as Blackhole-only -- on Wormhole ``AddrMode`` keeps
    the shared table's 23:15 span.
    """
    with _packer(blackhole=False) as (backend, memory):
        _pacr(backend, dst_access_mode=1, read_intf_sel=0x1)
        assert _packed(memory, ROW) == _expected([0])


def test_strided_pacr_uses_the_two_bit_addr_mode():
    """``ADDR_MOD_1`` on a strided PACR must select section 1, not section 5.

    Bit 17 falls inside the shared table's ``AddrMode`` span, so before the
    re-read a strided ``ADDR_MOD_1`` resolved to ``ADDR_MOD_PACK_SEC5`` -- which
    no LLK programs, and which therefore left the pack Y counter parked.
    ``pack_untilize_dest``'s MOP issues 16 such PACRs, one per face row, so a
    single output row was replicated down the whole tile.
    """
    with _packer() as (backend, _):
        backend.config_unit.setThreadConfig(
            0,
            TensixConfigurationConstants.get_addr32("ADDR_MOD_PACK_SEC1_YsrcIncr"),
            1 << TensixConfigurationConstants.get_shamt("ADDR_MOD_PACK_SEC1_YsrcIncr"),
        )
        adc = backend.getADC(0).Packers.Channel[0]
        assert adc.Y == 0
        _pacr(backend, addr_mode=1, dst_access_mode=1, read_intf_sel=0x3)
        assert adc.Y == 1


def test_dest_access_cfg_write_sets_the_dst_gates():
    """A ``DEST_ACCESS_CFG`` write reaches ``DstRegister``'s Adj16/Adj32 gates."""
    with _packer(remap=False) as (backend, _):
        dst = backend.getDst()
        gates = lambda: (dst.dest_remap_addrs, dst.dest_swizzle_32b)  # noqa: E731
        assert gates() == (False, False)
        _set_config(backend, "DEST_ACCESS_CFG_remap_addrs", 1)
        assert gates() == (True, False)
        _set_config(backend, "DEST_ACCESS_CFG_swizzle_32b", 1)
        assert gates() == (True, True)
        _set_config(backend, "DEST_ACCESS_CFG_remap_addrs", 0)
        assert gates() == (False, True)


def test_strided_without_the_dest_access_cfg_gates_raises():
    """The refusal is narrowed, not deleted.

    ttsim ties strided mode to both ``DEST_ACCESS_CFG`` bits and refuses
    otherwise; without the remap the stride is not 16 rows and there is no
    reference for what it would be.

    The message is pinned as well as the exception. The only way a real kernel
    reaches this is a ``pack_untilize_dest_init`` issued *after* its math --
    ``MATH(_llk_math_reconfig_remap_)`` then spins on the MATH_PACK semaphore
    behind the very pack it is meant to configure, so the two bits are still
    clear when PACK issues the PACR. Stating only the condition sends the
    reader looking for a missing tt-sim mode instead of a missing config write,
    which is what happened the first time (`optests/packuntilizeinit` on
    Blackhole); the fix belongs in the kernel.
    """
    with _packer(remap=False) as (backend, _):
        with pytest.raises(NotImplementedError, match="DEST_ACCESS_CFG") as excinfo:
            _pacr(backend, dst_access_mode=1, read_intf_sel=0x3)
        message = str(excinfo.value)
        assert "pack_untilize_dest_init AFTER its math" in message
        assert "_llk_math_reconfig_remap_" in message
        assert "MATH_PACK" in message


def test_strided_with_a_non_contiguous_interface_mask_raises():
    with _packer() as (backend, _):
        with pytest.raises(NotImplementedError, match="read_intf_sel"):
            _pacr(backend, dst_access_mode=1, read_intf_sel=0x5)


# --- The four Blackhole PACR fields tt-sim does not decode at all ------------
#
# ``ctxt_ctrl`` (3:2), ``addr_cnt_context`` (14:13), ``row_pad_zero`` (20:18)
# and ``cfg_context`` (22:21) have no entry in the shared Wormhole argument
# table and nothing in ``PackerUnit`` reads them. None of them is *mis*-decoded
# (the test below pins that), but a kernel setting one would have it silently
# ignored -- so, as ttsim does in ``TENSIX_EXECUTE_PACR``, PACR refuses.

#: ``(kwarg, all-ones value, bit span as it appears in the error)``.
BH_ONLY_FIELDS = [
    ("ctxt_ctrl", 0b11, "3:2"),
    ("addr_cnt_context", 0b11, "14:13"),
    ("row_pad_zero", 0b111, "20:18"),
    ("cfg_context", 0b11, "22:21"),
]


@pytest.mark.parametrize(
    "field, value, span", BH_ONLY_FIELDS, ids=[f[0] for f in BH_ONLY_FIELDS]
)
def test_unmodelled_blackhole_pacr_field_raises(field, value, span):
    with _packer() as (backend, _):
        with pytest.raises(NotImplementedError, match=f"{field}={value}.*{span}"):
            _pacr(backend, read_intf_sel=0x1, **{field: value})


@pytest.mark.parametrize(
    "field, value, span", BH_ONLY_FIELDS, ids=[f[0] for f in BH_ONLY_FIELDS]
)
def test_unmodelled_field_never_leaked_into_an_argument_that_is_read(
    field, value, span
):
    """Why nothing was silently *wrong* before the refusal, only ignored.

    Each field lands inside a decoded argument's span -- ``ctxt_ctrl`` inside
    ``Flush`` (3:1), ``addr_cnt_context`` inside ``ZeroWrite`` (14:12),
    ``row_pad_zero`` and ``cfg_context`` inside ``AddrMode`` (23:15) -- but
    every consumer takes only the bit or bits Wormhole defines: ``Flush`` and
    ``ZeroWrite`` via ``get_nth_bit(..., 0)``, ``AddrMode`` (on Blackhole) via
    a raw 16:15 re-read. ``Last`` is one bit, ``Concat`` is never read and
    ``read_intf_sel`` occupies Wormhole's ``PackSel`` span exactly.
    """
    baseline = _decode(_pacr_word(read_intf_sel=0x1))
    polluted = _decode(_pacr_word(read_intf_sel=0x1, **{field: value}))

    # The raw argument really is polluted for two of the four -- this is not a
    # vacuous test.
    assert polluted != baseline

    assert polluted["Last"] == baseline["Last"]
    assert polluted["PackSel"] == baseline["PackSel"]
    assert polluted["OvrdThreadId"] == baseline["OvrdThreadId"]
    assert get_nth_bit(polluted["Flush"], 0) == get_nth_bit(baseline["Flush"], 0)
    assert get_nth_bit(polluted["ZeroWrite"], 0) == get_nth_bit(
        baseline["ZeroWrite"], 0
    )
    # Blackhole's AddrMode is the raw 2-bit field, not the decoded 23:15 span.
    assert (_pacr_word(read_intf_sel=0x1, **{field: value}) >> 15) & 3 == 0


@pytest.mark.parametrize(
    "field, value, span", BH_ONLY_FIELDS, ids=[f[0] for f in BH_ONLY_FIELDS]
)
def test_wormhole_pacr_is_unaffected(field, value, span):
    """Those bit positions mean something else on Wormhole, whose PACR is
    correct today -- so the refusal must be Blackhole-only."""
    with _packer(blackhole=False) as (backend, memory):
        _pacr(backend, read_intf_sel=0x1, **{field: value})
        assert _packed(memory, ROW) == _expected([0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
