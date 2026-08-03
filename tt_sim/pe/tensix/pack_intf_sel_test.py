"""Tests for which packer a PACR drives, and where it reads Dst from.

The PACR field in instruction bits 11:8 means two different things:

* on **Wormhole** it is ``PackSel``, a mask over four independent packers, each
  reading its own configuration section (``THCON_SEC0_REG1`` / ``SEC0_REG8`` /
  ``SEC1_REG1`` / ``SEC1_REG8``);
* on **Blackhole** there is a single packer and the field is ``read_intf_sel``,
  a mask over that packer's four *Dst read interfaces*. Interface ``k`` reads
  Dst row ``base + k``, and the whole configuration comes from
  ``THCON_SEC0_REG1``.

Reading the Blackhole field as a packer selection is not a harmless
over-approximation: the tilize pack MOP issues ``0b0101`` / ``0b1010``, so it
packed the tile twice *and* consulted ``THCON_SEC1_REG1``, which tt-metal never
writes on this architecture. Its zero ``Disable_zero_compress`` bit -- inverted
naming: 0 means "compression is not disabled" -- then read as "this packer is
compressing" and refused the program with ``NotImplementedError``. So these
pin all three consequences:

* the mask selects one packer on Blackhole and up to four on Wormhole;
* the interfaces it names fix both the datum count and the Dst rows;
* an unwritten section is never consulted, while a packer that this PACR *does*
  drive and that genuinely asks for zero compression still raises.

Runs standalone (``python3 -m tt_sim.pe.tensix.pack_intf_sel_test``) or under
pytest.
"""

from contextlib import contextmanager

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import DataFormatConversions, TensixConfigurationConstants

BF16 = 5  # DataFormat.BF16
#: L1_Dest_addr, in 16-byte units, and the byte address it names.
L1_DEST_ADDR = 0x100
OUT_BYTE_ADDR = L1_DEST_ADDR << 4
#: Datums one Dst read interface supplies per PACR.
ROW = 16


class _L1:
    """Addressable memory that remembers every byte written, and by whom."""

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


def _dst_value(row, col):
    """A distinct bf16 bit pattern per (row, column), as Dst stores it.

    Dst holds bf16 as sign, mantissa(7), exponent(8); the packer's
    ``BF16InDstToBF16`` rearranges that to the IEEE order, so the expected
    output is that function of what is written here.
    """
    return ((row * ROW + col + 1) << 8) | 0x3F


@contextmanager
def _packer(blackhole, compress=False):
    """A backend configured for a BF16 -> BF16 Dst -> L1 pack of one row.

    Only ``THCON_SEC0_REG1`` is written -- packers 1..3's sections are left at
    zero, exactly as tt-metal leaves them on Blackhole.

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
        _set_config(backend, "THCON_SEC0_REG1_In_data_format", BF16)
        _set_config(backend, "THCON_SEC0_REG1_Out_data_format", BF16)
        # The field is inverted: 1 means zero compression is *disabled*, which
        # is what tt-metal sets for every packer it configures.
        _set_config(
            backend, "THCON_SEC0_REG1_Disable_zero_compress", 0 if compress else 1
        )
        # Pass every datum through the edge masks (the reduce LLK is what
        # drives these; see PackerUnit.edge_masks_for_pacr).
        _set_config(backend, "PCK_EDGE_OFFSET_SEC0_mask", 0xFFFF)
        _set_config(backend, "PACK_COUNTERS_SEC0_pack_reads_per_xy_plane", ROW)

        # ADC channel 1 X is the last input datum index (inclusive): one row.
        backend.getADC(0).Packers.Channel[1].X = ROW - 1

        for row in range(4):
            for col in range(ROW):
                backend.getDst().setDst16b(row, col, _dst_value(row, col))

        yield backend, memory
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _pacr(backend, packSel):
    backend.packer_unit.handle_pacr(
        None,
        0,
        {
            "Last": 1,
            "Flush": 0,
            "OvrdThreadId": 0,
            "PackSel": packSel,
            "ZeroWrite": 0,
            "AddrMode": 0,
        },
    )


def _packed(memory, count):
    return [
        int.from_bytes(memory.read(OUT_BYTE_ADDR + 2 * i, 2), "little")
        for i in range(count)
    ]


def _expected(rows):
    """What ``rows`` of Dst look like once packed, in emission order."""
    return [
        DataFormatConversions.BF16InDstToBF16(_dst_value(row, col))
        for row in rows
        for col in range(ROW)
    ]


@pytest.mark.parametrize(
    "packSel, wormhole",
    [(0x0, (0,)), (0x1, (0,)), (0x3, (0, 1)), (0x5, (0, 2)), (0xF, (0, 1, 2, 3))],
)
def test_participating_packers(packSel, wormhole):
    """The mask selects packers on Wormhole and always the one packer on Blackhole."""
    with _packer(blackhole=False) as (backend, _):
        assert backend.packer_unit.participating_packers(packSel) == wormhole
    with _packer(blackhole=True) as (backend, _):
        assert backend.packer_unit.participating_packers(packSel) == (0,)


@pytest.mark.parametrize(
    "packSel, interfaces",
    [(0x0, [0, 1, 2, 3]), (0x1, [0]), (0x3, [0, 1]), (0x5, [0, 2]), (0xA, [1, 3])],
)
def test_dst_read_interfaces(packSel, interfaces):
    """``read_intf_sel`` names the interfaces, and 0 means all four."""
    with _packer(blackhole=True) as (backend, _):
        assert backend.packer_unit.dst_read_interfaces(packSel) == interfaces


@pytest.mark.parametrize(
    "packSel, rows",
    [
        (0x0, [0, 1, 2, 3]),  # all four interfaces: four contiguous Dst rows
        (0x1, [0]),
        (0x3, [0, 1]),
        (0x5, [0, 2]),  # tilize's first PACR: the even rows of two pairs
        (0xA, [1, 3]),  # tilize's second PACR: the odd ones
    ],
)
def test_blackhole_pacr_reads_selected_interfaces(packSel, rows):
    """A Blackhole PACR packs 16 datums per selected interface, from row base + k.

    None of these may consult ``THCON_SEC1_*``: those sections are unwritten
    here, so before the mask was read as an interface selection ``0b0101`` and
    ``0b1010`` raised ``NotImplementedError`` out of the zero-compression check
    rather than packing anything.
    """
    with _packer(blackhole=True) as (backend, memory):
        _pacr(backend, packSel)
        assert _packed(memory, ROW * len(rows)) == _expected(rows)
        # Exactly the datums packed, and nothing at the address an unwritten
        # section would have pointed packers 2/3 at.
        assert memory.written == set(
            range(OUT_BYTE_ADDR, OUT_BYTE_ADDR + 2 * ROW * len(rows))
        )


def test_wormhole_pacr_packs_every_selected_packer():
    """Wormhole is unmoved: ``PackSel`` still runs one packer per set bit.

    Packer 1's section (``THCON_SEC0_REG8``) is unconfigured here, so its copy
    lands at the bottom of L1 -- which is the point: on this architecture the
    mask *does* reach a second packer, where on Blackhole the same bits must
    not. tt-metal configures every packer it selects, so a real Wormhole
    program never packs from an unwritten section.
    """
    with _packer(blackhole=False) as (backend, memory):
        _pacr(backend, 0x3)
        assert _packed(memory, ROW) == _expected([0])
        # L1_Dest_addr 0 + the tile header skip -> byte address 16.
        assert 16 in memory.written


def test_participating_packer_asking_for_compression_still_raises():
    """The refusal is scoped, not deleted."""
    with _packer(blackhole=True, compress=True) as (backend, _):
        with pytest.raises(NotImplementedError, match="zero compression"):
            _pacr(backend, 0x5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
