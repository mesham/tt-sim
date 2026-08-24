"""Tests for the PACR output address generator's channel-1 (Y/Z/W) offset.

``PackerUnit.generate_output_address`` folds the packer's channel-1 address
counters into the L1 destination address::

    YZW_Addr = PCK0_ADDR_BASE_REG_1_Base + ADC_Out.Y * ADDR_CTRL_XY_REG_1_Ystride
                                         + ADC_Out.Z * ADDR_CTRL_ZW_REG_1_Zstride
                                         + ADC_Out.W * ADDR_CTRL_ZW_REG_1_Wstride

but the two architectures disagree on the *units* of that sum, which is the
whole point of this file. ``Addr`` counts in 16-byte units (it is shifted left
by four to become a byte address). On **Wormhole** ``YZW_Addr`` is already in
those units and ``OutputAddressGenerator.md`` folds it in as ``&= ~15``; on
**Blackhole** the registers hold **bytes** and the doc's branch is ``>>= 4``.
Applying Wormhole's mask on Blackhole therefore does not merely round
differently -- it lands a nonzero offset **16x too far out**.

The sanity check on the Blackhole reading is tt-metal's own numbers:
``set_packer_strides`` on Blackhole writes exactly one channel-1 register,
``z_stride_ch1 = FACE_R_DIM * y_stride``, which for bf16 is 512 -- one whole
face of output. ``>> 4`` turns that into the 32 16-byte units that a face
occupies; ``& ~15`` would step a tilizing pack 8 KB per face. ttsim agrees
(``addr += yzw_addr >> 4`` under ``TT_ARCH_VERSION == 1``).

This path is **unreached by every program in the example set**: outside
``llk_pack_tilize`` tt-metal leaves the base and all three channel-1 strides at
zero, and the Wormhole LLK never writes them at all, so every PACR the replays
issue computes ``YZW_Addr == 0`` and lands in the same place either way. These
tests are the only thing that exercises the line, on either architecture.

Runs standalone (``python3 -m tt_sim.pe.tensix.pack_yzw_addr_test``) or under
pytest.
"""

from contextlib import contextmanager

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import DataFormatConversions, TensixConfigurationConstants

BF16 = 5  # DataFormat.BF16
#: ``THCON_SEC0_REG1_L1_Dest_addr``, in 16-byte units, and the byte address it
#: names once ``Sub_l1_tile_header_size`` has suppressed the header skip.
L1_DEST_ADDR = 0x100
BASE_BYTE_ADDR = L1_DEST_ADDR << 4
#: Datums one Dst read interface supplies per PACR.
ROW = 16

#: ``FACE_R_DIM * FACE_C_DIM * sizeof(bf16)``: what tt-metal's Blackhole
#: ``set_packer_strides`` puts in ``PCK0_ADDR_CTRL_ZW_REG_1_Zstride`` for a
#: tilizing pack, and the one channel-1 value a real program ever writes.
FACE_BYTES = 16 * 16 * 2


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


def _dst_value(col):
    """A distinct bf16 bit pattern per column, as Dst stores it."""
    return ((col + 1) << 8) | 0x3F


@contextmanager
def _packer(blackhole):
    """A backend configured for a BF16 -> BF16 Dst -> L1 pack of one row.

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
        # Inverted field: 1 means zero compression is disabled, as tt-metal
        # sets for every packer it configures.
        _set_config(backend, "THCON_SEC0_REG1_Disable_zero_compress", 1)
        _set_config(backend, "PCK_EDGE_OFFSET_SEC0_mask", 0xFFFF)
        _set_config(backend, "PACK_COUNTERS_SEC0_pack_reads_per_xy_plane", ROW)

        # ADC channel 1 X is the last input datum index (inclusive): one row.
        backend.getADC(0).Packers.Channel[1].X = ROW - 1

        for col in range(ROW):
            backend.getDst().setDst16b(0, col, _dst_value(col))

        yield backend, memory
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _set_channel1(backend, base=0, y=0, ystride=0, z=0, zstride=0, w=0, wstride=0):
    """Program the channel-1 address registers and their counters."""
    _set_config(backend, "PCK0_ADDR_BASE_REG_1_Base", base)
    _set_config(backend, "PCK0_ADDR_CTRL_XY_REG_1_Ystride", ystride)
    _set_config(backend, "PCK0_ADDR_CTRL_ZW_REG_1_Zstride", zstride)
    _set_config(backend, "PCK0_ADDR_CTRL_ZW_REG_1_Wstride", wstride)
    adc = backend.getADC(0).Packers.Channel[1]
    adc.Y, adc.Z, adc.W = y, z, w


def _pacr(backend):
    """One PACR driving packer 0 / read interface 0, in the default addr mode."""
    backend.packer_unit.handle_pacr(
        {"raw_instruction": 0x41 << 24},
        0,
        {
            "Last": 1,
            "Flush": 0,
            "OvrdThreadId": 0,
            "PackSel": 0x1,
            "ZeroWrite": 0,
            "AddrMode": 0,
        },
    )


def _expected_row():
    """The 16 bf16 datums a PACR emits, in order."""
    return [
        DataFormatConversions.BF16InDstToBF16(_dst_value(col)) for col in range(ROW)
    ]


def _packed_at(memory, byte_addr):
    return [
        int.from_bytes(memory.read(byte_addr + 2 * i, 2), "little") for i in range(ROW)
    ]


def _pack_and_find(backend, memory):
    """Issue the PACR and return the byte address the row actually landed at."""
    _pacr(backend)
    assert memory.written, "the PACR wrote nothing"
    start = min(memory.written)
    assert memory.written == set(range(start, start + 2 * ROW))
    assert _packed_at(memory, start) == _expected_row()
    return start


#: ``(kwargs, wormhole_offset_bytes, blackhole_offset_bytes)``.
#:
#: Wormhole keeps ``YZW_Addr`` in 16-byte units (``& ~15`` -> ``* 16`` bytes);
#: Blackhole reads it as bytes (``>> 4`` then ``<< 4`` -> the value itself,
#: truncated to a 16-byte boundary).
CASES = [
    # tt-metal's Blackhole tilizing pack: one face per Z step. 512 bytes is
    # right; 8192 is what the Wormhole mask would have produced.
    (dict(z=1, zstride=FACE_BYTES), FACE_BYTES * 16, FACE_BYTES),
    (dict(z=3, zstride=FACE_BYTES), 3 * FACE_BYTES * 16, 3 * FACE_BYTES),
    # A base with no counters at all.
    (dict(base=256), 256 * 16, 256),
    # Y and W strides, and all four terms summing.
    (dict(y=2, ystride=32), 64 * 16, 64),
    (dict(w=1, wstride=2048), 2048 * 16, 2048),
    (
        dict(base=16, y=1, ystride=32, z=1, zstride=FACE_BYTES, w=1, wstride=1024),
        (16 + 32 + FACE_BYTES + 1024) * 16,
        16 + 32 + FACE_BYTES + 1024,
    ),
    # Not a multiple of 16: both architectures discard the low four bits, but
    # of different units -- Wormhole drops 15 *16-byte units* (240 bytes) and
    # Blackhole drops 8 bytes.
    (dict(base=280), 272 * 16, 272),
]


@pytest.mark.parametrize("kwargs, wh_offset, bh_offset", CASES)
def test_channel1_offset_units_are_architecture_specific(kwargs, wh_offset, bh_offset):
    """The same register values move the pack 16x further on Wormhole than Blackhole.

    Before the fix both architectures took the Wormhole branch, so every
    Blackhole expectation here was off by exactly that factor.
    """
    with _packer(blackhole=False) as (backend, memory):
        _set_channel1(backend, **kwargs)
        assert _pack_and_find(backend, memory) == BASE_BYTE_ADDR + wh_offset

    with _packer(blackhole=True) as (backend, memory):
        _set_channel1(backend, **kwargs)
        assert _pack_and_find(backend, memory) == BASE_BYTE_ADDR + bh_offset


@pytest.mark.parametrize("blackhole", [False, True])
def test_zero_channel1_config_is_the_shared_case(blackhole):
    """With the registers unwritten -- what every current tt-metal pack does --
    both architectures land at the plain ``L1_Dest_addr``.

    This is why the divergence stayed latent: no example program in the tree
    reaches a nonzero ``YZW_Addr``.
    """
    with _packer(blackhole=blackhole) as (backend, memory):
        assert _pack_and_find(backend, memory) == BASE_BYTE_ADDR


def test_blackhole_face_stride_is_one_face():
    """The Blackhole reading is self-consistent: N Z-steps of tt-metal's
    ``z_stride_ch1`` land N faces apart, contiguously tiling the output."""
    starts = []
    with _packer(blackhole=True) as (backend, memory):
        for z in range(4):
            memory.written.clear()
            _set_channel1(backend, z=z, zstride=FACE_BYTES)
            starts.append(_pack_and_find(backend, memory))
    assert starts == [BASE_BYTE_ADDR + z * FACE_BYTES for z in range(4)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
