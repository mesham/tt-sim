"""Tests for the UNPACR input-walk modes the unpacker declines to model.

``UNPACR_Regular.md`` lets the input rows be *discontiguous*: with
``Tileize_mode`` set, the input address advances by ``RowStride`` bytes every
``UnpackRowWidth`` datums instead of abutting, and ``Upsample_rate`` spreads
each input datum over several output positions. ``perform_unpack`` models
neither -- it walks the input flat and emits one output datum per input datum --
so a configuration using either used to read the wrong L1 addresses (or write
the wrong Src/Dst rows) with no error at all. These pin the boundary:

* the *contiguous* RowStride is ``DatumSizeBytes * UnpackRowWidth``, and
  ``UnpackRowWidth`` is 16 on Wormhole but 32 on Blackhole for anything wider
  than a byte per datum -- so the same Tileize_mode configuration is contiguous
  on one architecture and strided on the other, and the check has to know which;
* ``RowStride`` is assembled from three 4-bit ``Shift_amount_cntx`` fields at
  shifts 4, 8 and 12 (max 65520 bytes);
* ``Upsample_and_interleave`` on its own, at ``Upsample_rate == 0``, is a no-op
  -- the doc's inner loop runs exactly once -- so it is accepted, not rejected.

Every case drives a real backend through ``read_unpack_state`` /
``perform_unpack_state``, so the config decode, the checks and (for the accepted
cases) the datum walk itself are all exercised.

Runs standalone (``python3 -m tt_sim.pe.tensix.unpack_stride_test``) or under
pytest.
"""

import struct
from contextlib import contextmanager
from itertools import product

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import DataFormatConversions, TensixConfigurationConstants

BF16 = 5  # DataFormat.BF16, as it appears in a tile descriptor / REG2

#: Where the tile lives in L1, and the ``REG3_Base_address`` that names it
#: (``InAddr = (Base + Offset + 1) * 16``).
IN_ADDR = 0x1000
BASE_ADDRESS = IN_ADDR // 16 - 1

#: Rows of 16 datums to unpack. 4 is enough for a stride to be visible and
#: leaves SrcA rows 4..63 untouched.
NUM_ROWS = 4

#: The UNPACR argument bits, all of them off: no row search, no context
#: counting, no zero-write, no dvalid flip, single-context mode.
UNPACR_ARGS = {
    "RowSearch": 0,
    "AutoIncContextID": 0,
    "ZeroWrite2": 0,
    "SetDatValid": 0,
    "OvrdThreadId": 0,
    "AddrCntContextId": 0,
    "CfgContextId": 0,
    "AddrMode": 0,
}


class _L1:
    """Just enough addressable memory for the unpacker to read a tile out of."""

    def __init__(self, size=1 << 16):
        self.data = bytearray(size)

    def read(self, addr, size):
        return bytes(self.data[addr - IN_ADDR : addr - IN_ADDR + size])

    def write(self, addr, payload):
        self.data[addr - IN_ADDR : addr - IN_ADDR + len(payload)] = payload


def _set_config(backend, key, value, state_id=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


def _set_thread_config(backend, key, value, thread=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    cfg = backend.config_unit.threadConfig[thread]
    cfg[addr32] = (cfg[addr32] & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)


@contextmanager
def _unpacker(blackhole=False, tileize=0, shifts=(0, 0, 0), upsample=(0, 0)):
    """A backend configured for a BF16 -> BF16 unpack of ``NUM_ROWS`` rows into SrcA.

    The config-register layout is a process-global selection, so restore the
    Wormhole layout on the way out -- other tests in the same session (and the
    Wormhole replay guards) expect it.
    """
    try:
        backend = TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size if blackhole else None,
            BLACKHOLE_PROFILE.tensix_thd_state_size if blackhole else None,
            blackhole=blackhole,
        ).getBackend()
        memory = _L1()
        backend.setAddressableMemory(memory)

        # Tile descriptor: BF16 datums, X=16, Y=Z=W=1, uncompressed.
        descriptor = TensixConfigurationConstants.get_addr32(
            "THCON_SEC0_REG0_TileDescriptor"
        )
        backend.config_unit.setConfig(0, descriptor + 0, (16 << 16) | BF16)
        backend.config_unit.setConfig(0, descriptor + 1, (1 << 16) | 1)
        backend.config_unit.setConfig(0, descriptor + 2, 1)
        backend.config_unit.setConfig(0, descriptor + 3, 0)

        _set_config(backend, "THCON_SEC0_REG3_Base_address", BASE_ADDRESS)
        _set_config(backend, "THCON_SEC0_REG2_Out_data_format", BF16)
        # Byte address 128 -> OutAddr 64 -> SrcA start row 0 (unpacker 0's OutAddr
        # counts from row 4).
        _set_config(backend, "UNP0_ADDR_BASE_REG_1_Base", 128)
        _set_thread_config(backend, "SRCA_SET_SetOvrdWithAddr", 1)

        _set_config(backend, "THCON_SEC0_REG2_Tileize_mode", tileize)
        for index, shift in enumerate(shifts):
            _set_config(backend, f"THCON_SEC0_REG2_Shift_amount_cntx{index}", shift)
        _set_config(backend, "THCON_SEC0_REG2_Upsample_rate", upsample[0])
        _set_config(backend, "THCON_SEC0_REG2_Upsample_and_interleave", upsample[1])

        # ADC channel 1 X is the last datum index to read (inclusive).
        backend.getADC(0).Unpacker[0].Channel[1].X = NUM_ROWS * 16 - 1
        yield backend, memory
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _bf16_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def _datum(row, col):
    """A distinct value per (row, column), so a misread address is visible."""
    return float(row * 16 + col + 1)


def _fill(memory, stride):
    for row in range(NUM_ROWS):
        for col in range(16):
            memory.write(
                IN_ADDR + row * stride + col * 2,
                struct.pack("<H", _bf16_bits(_datum(row, col))),
            )


def _unpack(backend):
    unpacker = backend.unpacker_units[0]
    unpacker.perform_unpack_state(0, unpacker.read_unpack_state(0, UNPACR_ARGS))
    srcA = backend.getSrcA(0)
    return [[srcA[row, col] for col in range(16)] for row in range(NUM_ROWS)]


def _expected():
    return [
        [
            DataFormatConversions.BF16ToSrcBF16(_bf16_bits(_datum(row, col)))
            for col in range(16)
        ]
        for row in range(NUM_ROWS)
    ]


def _contiguous_stride(blackhole):
    # BF16 is 2 bytes, and UnpackRowWidth is 32 on Blackhole above 1 byte/datum.
    return 2 * (32 if blackhole else 16)


@pytest.mark.parametrize("blackhole", [False, True])
def test_contiguous_unpack_moves_the_datums(blackhole):
    """The baseline the checks are calibrated against: rows abutting in L1."""
    with _unpacker(blackhole) as (backend, memory):
        _fill(memory, stride=32)
        assert _unpack(backend) == _expected()


@pytest.mark.parametrize("blackhole", [False, True])
def test_tileize_with_a_contiguous_row_stride_is_accepted(blackhole):
    """Tileize_mode is fine as long as its RowStride is the abutting one.

    The check is "the stride equals the contiguous value", not "the stride is
    zero" -- and the contiguous value is arch-dependent, so the *same* stride
    that is accepted here would be rejected on the other architecture.
    """
    stride = _contiguous_stride(blackhole)
    with _unpacker(blackhole, tileize=1, shifts=(stride >> 4, 0, 0)) as (
        backend,
        memory,
    ):
        _fill(memory, stride=32)
        assert _unpack(backend) == _expected()


@pytest.mark.parametrize("blackhole", [False, True])
def test_tileize_with_a_strided_row_stride_raises(blackhole):
    """A non-abutting stride fails loudly instead of reading the wrong addresses."""
    contiguous = _contiguous_stride(blackhole)
    # 64 bytes is *the* contiguous stride on Blackhole and a strided one on
    # Wormhole, which is exactly the case a stride-agnostic check would miss.
    for stride in (16, 32, 64, 128, 65520):
        if stride == contiguous:
            continue
        with _unpacker(
            blackhole,
            tileize=1,
            shifts=((stride >> 4) & 0xF, (stride >> 8) & 0xF, (stride >> 12) & 0xF),
        ) as (backend, memory):
            _fill(memory, stride=stride)
            with pytest.raises(NotImplementedError) as excinfo:
                _unpack(backend)
            message = str(excinfo.value)
            assert f"RowStride of {stride} bytes" in message
            assert f"only correct at {contiguous} bytes" in message


@pytest.mark.parametrize(
    "shifts,stride",
    [
        ((1, 0, 0), 16),
        ((0, 1, 0), 256),
        ((0, 0, 1), 4096),
        ((0xF, 0xF, 0xF), 65520),
    ],
)
def test_row_stride_is_assembled_from_the_three_shift_fields(shifts, stride):
    """RowStride = cntx0 << 4 | cntx1 << 8 | cntx2 << 12 (UNPACR_Regular.md).

    Read back out of the rejection message, which is the only place the decoded
    stride is now observable -- and the reason it has to be reported at all.
    """
    with _unpacker(tileize=1, shifts=shifts) as (backend, memory):
        _fill(memory, stride=32)
        with pytest.raises(NotImplementedError, match=f"RowStride of {stride} bytes"):
            _unpack(backend)


@pytest.mark.parametrize("rate", [1, 2, 3])
def test_upsample_rate_raises(rate):
    """Upsampling emits several output datums per input datum; the walk emits one."""
    with _unpacker(upsample=(rate, 0)) as (backend, memory):
        _fill(memory, stride=32)
        with pytest.raises(NotImplementedError) as excinfo:
            _unpack(backend)
        message = str(excinfo.value)
        assert f"Upsample_rate={rate}" in message
        assert f"UpsampleZeroes={(1 << rate) - 1}" in message


def test_upsample_interleave_alone_is_accepted():
    """At Upsample_rate 0 the doc's upsample loop runs once, so interleave is a no-op.

    Rejecting it would be over-strict: it changes nothing about where the datums
    land.
    """
    with _unpacker(upsample=(0, 1)) as (backend, memory):
        _fill(memory, stride=32)
        assert _unpack(backend) == _expected()


def main():
    """Run every test without pytest, expanding ``parametrize`` as pytest would."""
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        grids = []
        for mark in getattr(fn, "pytestmark", []):
            if mark.name != "parametrize":
                continue
            argnames = [n.strip() for n in mark.args[0].split(",")]
            grids.append(
                [
                    dict(zip(argnames, values if len(argnames) > 1 else (values,)))
                    for values in mark.args[1]
                ]
            )
        for combination in product(*grids):
            kwargs = {}
            for case in combination:
                kwargs.update(case)
            fn(**kwargs)
    print("unpack_stride_test OK: unmodelled unpack walks raise, contiguous ones run")


if __name__ == "__main__":
    main()
