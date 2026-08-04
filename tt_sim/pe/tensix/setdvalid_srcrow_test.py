"""``SETDVALID`` must not replace the unpacker's per-thread ``srcRow`` list.

A regression test for a crash the Tensix cycle-cost benchmark
(``perfbench/tensixbench``, see ``docs/plans/tensix-cost-benchmark.md``) found
on Blackhole. ``UnPackerUnit.srcRow`` is ``[0] * 3`` -- one entry per Tensix
thread -- and the unpacker's own bank flip writes ``srcRow[issue_thread]``. The
Miscellaneous Unit's ``SETDVALID`` handler, which performs the same bank flip
from the other side, assigned the bare attribute instead, replacing the list
with a scalar. Every later ``UNPACR`` then died with::

    TypeError: 'int' object does not support item assignment

It survived because in-tree workloads do not reach it: the ISA docs' `SETDVALID`
is a Wormhole NoOp and Blackhole kernels mark dvalid inline on the regular
`UNPACR` (see ``UnPackerUnit.flip_src_banks``). Nothing issued a bare
``SETDVALID`` before an unpack until a benchmark did it deliberately, to hold
the SrcA/SrcB valid bits across an unrolled ``MVMUL`` burst.

Run standalone (``python3 -m tt_sim.pe.tensix.setdvalid_srcrow_test``) or under
pytest.
"""

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor

#: ``SETDVALID`` from ``tensix_instructions.yaml``: op_binary 87, opcode in bits
#: 24-31, ``setvalid`` at bit 0 (bit 0 = SrcA, bit 1 = SrcB).
SETDVALID = 87 << 24


def _backend():
    return TensixCoProcessor(
        None,
        WORMHOLE_PROFILE.tensix_cfg_state_size,
        WORMHOLE_PROFILE.tensix_thd_state_size,
    ).getBackend()


def _misc(backend):
    return backend.misc_unit


def test_setdvalid_keeps_srcrow_indexable_per_thread():
    backend = _backend()
    misc = _misc(backend)

    before = [list(u.srcRow) for u in backend.unpacker_units]
    assert all(len(rows) == 3 for rows in before)

    misc.handle_setdvalid(None, 1, {"setvalid": 0x3})

    for unit in backend.unpacker_units:
        # Before the fix this was an int, and the next UNPACR raised TypeError.
        assert isinstance(unit.srcRow, list)
        assert len(unit.srcRow) == 3


def test_setdvalid_only_touches_the_issuing_thread():
    backend = _backend()
    misc = _misc(backend)
    for unit in backend.unpacker_units:
        unit.srcRow = [11, 22, 33]

    misc.handle_setdvalid(None, 2, {"setvalid": 0x3})

    for unit in backend.unpacker_units:
        assert unit.srcRow[0] == 11
        assert unit.srcRow[1] == 22


def test_setdvalid_flips_only_the_selected_source_bank():
    backend = _backend()
    misc = _misc(backend)
    banks = [u.srcBank for u in backend.unpacker_units]

    misc.handle_setdvalid(None, 0, {"setvalid": 0x1})  # SrcA only

    assert backend.unpacker_units[0].srcBank == banks[0] ^ 1
    assert backend.unpacker_units[1].srcBank == banks[1]


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
