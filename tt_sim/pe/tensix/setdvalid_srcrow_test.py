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
from tt_sim.pe.tensix.util import TensixInstructionDecoder

#: ``SETDVALID`` from ``tensix_instructions.yaml``: op_binary 87, opcode in bits
#: 24-31, ``setvalid`` at bit 0 (bit 0 = SrcA, bit 1 = SrcB).
SETDVALID = 87 << 24


#: ``UNPACR_NOP``, opcode 0x43. The two architectures lay the low 24 bits out
#: differently, so the benchmark's kernel emits a different word on each and both
#: are pinned here. These are exactly what
#: ``perfbench/tensixbench/src/kernels/compute/raw_probes.cpp`` builds via
#: ``TTI_UNPACR_NOP`` under ``TTBENCH_DVALID_UNPACR_NOP``.
#:
#: Blackhole: ``unpacker_select`` at bit 23, ``set_dvalid`` at bit 8,
#: ``unpack_pop`` at bits 1:0 (``p_unpacr_nop::UNP_ZEROSRC`` = 1).
#: Wormhole: ``unpacker_select`` at bit 23, ``NoOp`` mode select in the low bits
#: (``p_unpacr_nop::UNP_SET_DVALID`` = 0b111).
def _unpacr_nop_setdvalid(unpacker, blackhole):
    word = (0x43 << 24) | (unpacker << 23)
    return word | ((1 << 8) | 1 if blackhole else 0x7)


def _backend():
    return TensixCoProcessor(
        None,
        WORMHOLE_PROFILE.tensix_cfg_state_size,
        WORMHOLE_PROFILE.tensix_thd_state_size,
    ).getBackend()


@contextmanager
def _wormhole_backend():
    """A Wormhole backend, as a context manager so both arches read alike.

    Also pins the process-global config layout back to Wormhole on the way in,
    which matters when a Blackhole case ran first in the same session.
    """
    TensixConfigurationConstants.use_blackhole(False)
    yield _backend()


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


def _set_config(backend, key, value):
    """Write one config field, the way ``RMWCIB`` would, for the current state."""
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    unit = backend.getConfigUnit()
    old = unit.get_config_entry(0, addr32)
    unit.setConfig(0, addr32, (old & ~mask) | ((value << shamt) & mask))


def test_unpacr_nop_setdvalid_takes_a_defined_format_from_config():
    """Experiment X2's whole premise, pinned on both architectures.

    ``SETDVALID`` leaves ``ImpliedSrc{A,B}Fmt`` an ``UnpredictableValue()`` on
    Blackhole (see ``test_blackhole_implied_src_format_survives_setdvalid``: what
    tt-sim gives instead is whatever the last unpack happened to latch). The
    Blackhole-sanctioned replacement, ``UNPACR_NOP`` carrying ``set_dvalid``,
    does not have that problem -- ``UNPACR_NOP_SETDVALID.md`` has it copy
    ``THCON_SEC{0,1}_REG2_Out_data_format`` into the bank it hands over.

    That is what makes a source-format axis possible at all, and it is what
    ``perfbench/tensixbench --dvalid-unpacr-nop --src-format`` relies on: the
    benchmark writes the format into exactly this register and then issues
    exactly these instruction words. Anything that broke the link would make the
    format axis silently measure nothing, so it is pinned rather than assumed.
    """
    for blackhole in (False, True):
        maker = _blackhole_backend if blackhole else _wormhole_backend
        with maker() as backend:
            for fmt in (
                DataFormat.BF16,
                DataFormat.FP32,
                DataFormat.TF32,
                DataFormat.FP16,
            ):
                for unpacker_id, key, get_src in (
                    (0, "THCON_SEC0_REG2_Out_data_format", backend.getSrcA),
                    (1, "THCON_SEC1_REG2_Out_data_format", backend.getSrcB),
                ):
                    unpacker = backend.unpacker_units[unpacker_id]
                    bank = unpacker.srcBank
                    # Both banks back to the unpackers first. Blackhole's
                    # handler stalls (and does nothing) if the Matrix Unit still
                    # owns the bank it is about to reuse, which is the whole
                    # point of the ZEROSRC form -- but it means a previous
                    # iteration's hand-over would silently swallow this one.
                    for b in (0, 1):
                        get_src(b).setAllowedClient(SrcRegister.SrcClient.Unpackers)
                    _set_config(backend, key, int(fmt))

                    word = _unpacr_nop_setdvalid(unpacker_id, blackhole)
                    unpacker.handle_unpacr_nop(
                        TensixInstructionDecoder.getInstructionInfo(word),
                        1,
                        TensixInstructionDecoder.getInstructionInfo(word)["instr_args"],
                    )

                    src = get_src(bank)
                    assert src.getAllowedClient() is SrcRegister.SrcClient.MatrixUnit
                    assert src.getDataFormat() == fmt, (
                        f"{'blackhole' if blackhole else 'wormhole'} unpacker "
                        f"{unpacker_id}: expected {fmt!r}, got {src.getDataFormat()!r}"
                    )
                    # The instruction flips the unpacker's bank pointer, so the
                    # next iteration legitimately works on the other bank.
                    assert unpacker.srcBank == bank ^ 1


def test_unpacr_nop_setdvalid_twice_deadlocks_the_unpacker():
    """The Blackhole silicon hang, reproduced in the model.

    ``UNPACR_NOP`` with ``set_dvalid`` and ``Unpack_Pop = UNP_ZEROSRC`` -- the
    form ``_llk_unpack_set_srcb_dummy_valid_`` uses and the form
    ``perfbench/tensixbench --dvalid-unpacr-nop`` copied -- is the ACQUIRE half
    of the unpacker's handshake. It waits until the bank named by
    ``MatrixUnit.SrcABank`` is no longer valid, then zeroes the unpacker's own
    bank and hands it over. Issued twice with nothing releasing the first bank
    in between, the second one waits for ever.

    That is exactly what the benchmark does: every MATH probe issues
    ``clear_dvalid = 0`` so the valid bits survive the whole burst, so a
    completed run leaves ``SrcA[0]`` owned by the Matrix Unit. On silicon the
    next execution of the setup -- the next launch in the same process, or the
    first launch of the next process on the same card -- wedges, and only
    ``tt-smi -r 0`` clears it. A bare ``SETDVALID`` has no wait half, which is
    why ``--dvalid-once`` re-runs on the same card indefinitely.

    Pinned here because the state machine is the finding: the two shots below
    are the whole mechanism, and the second's ``blocked`` is the hang.
    """
    with _blackhole_backend() as backend:
        unpacker = backend.unpacker_units[0]
        word = _unpacr_nop_setdvalid(0, True)
        info = TensixInstructionDecoder.getInstructionInfo(word)

        unpacker.handle_unpacr_nop(info, 1, info["instr_args"])
        assert not unpacker.blocked, "the first hand-over finds the bank free"
        assert backend.getSrcA(0).getAllowedClient() is SrcRegister.SrcClient.MatrixUnit
        assert unpacker.srcBank == 1, "the unpacker's bank pointer advanced"
        assert backend.matrix_unit.srcABank == 0, "the Matrix Unit's did not"

        # Nothing has cleared dvalid -- as in the benchmark, where every MVMUL
        # carries clear_dvalid = 0 -- so the bank the Matrix Unit is pointing at
        # is still valid, and the acquire cannot complete.
        unpacker.handle_unpacr_nop(info, 1, info["instr_args"])
        assert unpacker.blocked, "the second hand-over must wait, for ever"
        assert unpacker.blocked_on() == (1, "UNPACR_NOP", "SrcA", 0)
        # And the thread that issued it must not read as done: a unit holding an
        # instruction that can never retire is what wedges the core on silicon.
        assert unpacker.hasInflightInstructionsFromThread(1)
        assert not unpacker.hasInflightInstructionsFromThread(0)

        # Releasing the bank -- the half the benchmark omits -- unblocks it.
        backend.getSrcA(0).setAllowedClient(SrcRegister.SrcClient.Unpackers)
        unpacker.clock_tick(0)
        assert not unpacker.blocked
        assert not unpacker.hasInflightInstructionsFromThread(1)


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
