"""``SETDVALID`` must not replace the unpacker's per-thread ``srcRow`` list.

A regression test for a crash the Tensix cycle-cost benchmark
(``perfbench/tensixbench``, see ``docs/plans/tensix-cost-benchmark.md``) found
on Blackhole. ``UnPackerUnit.srcRow`` is ``[0] * 3`` -- one entry per Tensix
thread -- and the unpacker's own bank flip writes ``srcRow[issue_thread]``. The
Miscellaneous Unit's ``SETDVALID`` handler, which performs the same bank flip
from the other side, assigned the bare attribute instead, replacing the list
with a scalar. Every later ``UNPACR`` then died with::

    TypeError: 'int' object does not support item assignment

It survived because the in-tree workloads do not reach it -- but NOT for the
reason this file used to give. `SETDVALID.md` makes the instruction **fully
functional on Wormhole** and marks it `UnsupportedFunctionality` **on
Blackhole** ("its interaction with implied data formats is ill-specified"),
which is the exact opposite of the "Wormhole NoOp" this docstring claimed. The
behavioural conclusion was right and the stated reason was inverted: what
actually keeps the handler cold is that the tt-metal kernels these examples run
mark dvalid inline on the regular ``UNPACR`` (see
``UnPackerUnit.flip_src_banks``), so nothing issued a bare ``SETDVALID`` before
an unpack until a benchmark did it deliberately, to hold the SrcA/SrcB valid
bits across an unrolled ``MVMUL`` burst.

Blackhole LLKs *do* issue it in paths no in-tree example exercises --
``TTI_SETDVALID(0b10)`` in ``llk_math_eltwise_unary_datacopy.h``'s 32-bit
unpack-to-dest broadcasts -- which is why tt-sim models the instruction on both
architectures rather than refusing it on Blackhole the way the vendor reference
simulator does. That decision, and what it leaves unmodelled, is argued in
``MiscellaneousUnit.handle_setdvalid``'s docstring and pinned by
``test_setdvalid_is_modelled_on_blackhole_too`` below.

Run standalone (``python3 -m tt_sim.pe.tensix.setdvalid_srcrow_test``) or under
pytest.
"""

from contextlib import contextmanager

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.backends.backend_base import DataFormat
from tt_sim.pe.tensix.backends.config import TensixConfigurationConstants
from tt_sim.pe.tensix.registers import SrcRegister
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


@contextmanager
def _blackhole_backend():
    """A Blackhole backend, restoring the process-global config layout after.

    The register layout is selected per process, and the Wormhole replay guards
    in the same session expect the Wormhole one.
    """
    try:
        yield TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size,
            BLACKHOLE_PROFILE.tensix_thd_state_size,
            blackhole=True,
        ).getBackend()
    finally:
        TensixConfigurationConstants.use_blackhole(False)


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


def test_setdvalid_is_modelled_on_blackhole_too():
    """The arch decision, pinned so it is a choice rather than an oversight.

    The ISA docs open Blackhole's functional model with
    ``UnsupportedFunctionality()`` and the vendor reference simulator raises
    there, so refusing would have been defensible and is what tt-sim does for
    ``MOVDBGB2D``/``RESOURCEDECL``/``STREAMWAIT``/``STREAMWRCFG``. It is not
    what tt-sim does here, because real Blackhole LLK code issues
    ``TTI_SETDVALID(0b10)`` (``llk_math_eltwise_unary_datacopy.h``'s 32-bit
    unpack-to-dest broadcasts) and refusing would break that path in exchange
    for no check -- the vendor sim cannot be diffed against on an instruction it
    declines. See ``handle_setdvalid``'s docstring for the full argument.
    """
    with _blackhole_backend() as backend:
        before = backend.unpacker_units[1].srcBank
        backend.misc_unit.handle_setdvalid(None, 1, {"setvalid": 0b10})
        assert backend.getSrcB(before).allowedClient is SrcRegister.SrcClient.MatrixUnit
        assert backend.unpacker_units[1].srcBank == before ^ 1


def test_blackhole_implied_src_format_survives_setdvalid():
    """The documented in-practice behaviour, which is what tt-sim gives.

    ``SETDVALID.md`` says Blackhole sets ``ImpliedSrc{A,B}Fmt`` to
    ``UnpredictableValue()``, and in the same breath that the hardware "records
    a stale/held copy of a previous unpack's output format". tt-sim's implied
    format IS that stale copy -- it is the format latched on the Src bank by the
    last unpack, and ``SETDVALID`` does not disturb it. Unpredictability is not
    a value, so this is the closest a deterministic model can get, and the test
    exists so that anyone who later decides to poison the format on purpose
    has to come here and say so.
    """
    with _blackhole_backend() as backend:
        matrix = backend.matrix_unit
        bank = matrix.srcABank
        backend.getSrcA(bank).setDataFormat(DataFormat.TF32)
        backend.misc_unit.handle_setdvalid(None, 1, {"setvalid": 0b1})
        assert matrix.implied_srcA_format(1, DataFormat.FP32) == DataFormat.TF32


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
