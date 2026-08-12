"""The Src ownership handshake, from the Matrix Unit's side.

Two banks of ``SrcA`` and ``SrcB`` are passed back and forth between the
unpackers and the Matrix Unit (FPU) by one field per bank, ``AllowedClient``.
The unpackers **acquire** a bank (``UNPACR`` with ``SetDvalid``, ``UNPACR_NOP``
carrying ``set_dvalid``, ``SETDVALID``), the FPU **releases** it (``clear_dvalid``
on a math op, ``clear_ab_vld`` on ``SETRWC``, or ``CLEARDVALID``). tt-sim models
the unpacker's side in ``UnPackerUnit`` (``blocked`` / ``blocked_wait_bank``);
this file pins the FPU's two halves.

**Waiting to acquire.** ``WaitGate.checkIfFPUInstructionShouldStall`` holds an
FPU instruction at the gate while the bank it consumes is still the unpackers'.
That check has been there all along, contrary to
``docs/plans/matrix-unit-thread-contention.md``'s claim that "tt-sim's Matrix
Unit does not model the Wait Gate at all" -- but it is driven by a hardcoded
list of opcode *strings* compared against the decoded instruction name, so a
misspelling removes an instruction from the gate silently. One had:
``TRNSPSRCB`` was spelled ``TRANSPSRCB``, so a transpose never waited for SrcB
where ttsim's ``TENSIX_EXECUTE_TRNSPSRCB`` stalls until the FPU owns the bank.
:func:`test_every_gated_opcode_exists_in_the_instruction_table` is the check
that would have caught it. The list also had the opposite error -- an
instruction gated that the documents say is not, ``MOVDBGA2D``, whose whole
purpose is to read ``SrcA`` *without* waiting; :func:`test_movdbga2d_is_not_gated`
pins that, so both directions of edit to the list are now deliberate.

**Releasing.** Handing a bank back that the FPU does not own is
``NonContractualBehavior`` -- ttsim's ``math_clear_src_valid`` and
``TENSIX_EXECUTE_CLEARDVALID`` both ``TTSIM_VERIFY`` against it -- and on
silicon it desynchronises the bank pointers so a later acquire waits for a
release that never comes. tt-sim used to do it silently, in four separate
copies of the same five lines; it now raises
:class:`~tt_sim.pe.tensix.backends.matrix.SrcDvalidError` from one place.

Run standalone (``python3 -m tt_sim.pe.tensix.waitgate_allowed_client_test``)
or under pytest.
"""

import pytest
import yaml

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.pe.tensix.backends import matrix as matrix_mod
from tt_sim.pe.tensix.backends.matrix import SrcDvalidError
from tt_sim.pe.tensix.frontend import WaitGate
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixConfigurationConstants


def _coprocessor():
    return TensixCoProcessor(
        None,
        WORMHOLE_PROFILE.tensix_cfg_state_size,
        WORMHOLE_PROFILE.tensix_thd_state_size,
    )


def _op_binary(name):
    with open(
        "tt_sim/pe/tensix/tensix_instructions.yaml", encoding="utf-8"
    ) as instructions:
        return yaml.safe_load(instructions)[name]["op_binary"]


def _hand_banks_to_fpu(backend):
    for bank in (0, 1):
        backend.getSrcA(bank).allowedClient = SrcRegister.SrcClient.MatrixUnit
        backend.getSrcB(bank).allowedClient = SrcRegister.SrcClient.MatrixUnit


def _set_thread_config(backend, key, value, thread=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    cfg = backend.config_unit.threadConfig[thread]
    cfg[addr32] = (cfg[addr32] & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)


def _run(coprocessor, cycles):
    clocks = coprocessor.getClocks()
    for cycle in range(cycles):
        for clock in clocks:
            clock.clock_tick(cycle)


# --- the acquire side: waiting at the gate -----------------------------------


def test_every_gated_opcode_exists_in_the_instruction_table():
    """A misspelling here silently un-gates an instruction, so spell-check it.

    The gate compares decoded instruction *names* against these tuples, so a
    name the instruction table does not contain can never match and the
    instruction dispatches without ever testing ``AllowedClient``.
    """
    with open(
        "tt_sim/pe/tensix/tensix_instructions.yaml", encoding="utf-8"
    ) as instructions:
        known = set(yaml.safe_load(instructions))
    gated = set()
    for group in WaitGate.MATH_ALLOWED_CLIENT_INSTRUCTIONS:
        gated.update(group)
    assert gated - known == set()


@pytest.mark.parametrize(
    "opcode,which",
    [
        ("MVMUL", "both"),
        ("ELWADD", "both"),
        ("ELWSUB", "both"),
        ("ELWMUL", "both"),
        ("GAPOOL", "both"),
        ("GMPOOL", "both"),
        ("MOVA2D", "SrcA"),
        ("MOVB2A", "SrcB"),
        ("MOVB2D", "SrcB"),
        # The one the typo un-gated.
        ("TRNSPSRCB", "SrcB"),
    ],
)
def test_fpu_instruction_waits_for_the_bank_it_consumes(opcode, which):
    word = _op_binary(opcode) << 24
    if opcode == "GMPOOL":
        word |= 1 << 19  # instr_mod19: the only primitive either sim implements
    cp = _coprocessor()
    thread = cp.getThread(1)
    thread.push_wait_gate_instruction(word)
    _run(cp, 20)
    assert thread.wait_gate_instruction_fifo, (
        f"{opcode} dispatched with {which} still owned by the unpackers"
    )
    # ...and the thread is not reported done while it is stuck there, which is
    # what a kernel's tensix_sync() reads.
    assert cp.CoprocessorDoneCheck(1)

    cp = _coprocessor()
    _hand_banks_to_fpu(cp.getBackend())
    thread = cp.getThread(1)
    thread.push_wait_gate_instruction(word)
    _run(cp, 20)
    assert not thread.wait_gate_instruction_fifo


def test_movdbga2d_is_not_gated():
    """``MOVDBGA2D`` is the one SrcA-reading move that must *not* wait.

    It was in ``MATH_ALLOWED_CLIENT_INSTRUCTIONS[0]`` -- an over-charge, and
    over-charging is the direction the cost model's floor policy forbids.
    ``MOVDBGA2D.md``: "This instruction is identical to `MOVA2D`, except that
    it doesn't _automatically_ wait for `SrcA[MatrixUnit.SrcABank].AllowedClient
    == MatrixUnit`", and its functional model says to read ``MOVA2D``'s
    "ignoring the paragraph about the Wait Gate" -- the paragraph being the
    ``while (SrcA[...].AllowedClient != MatrixUnit) { wait; }`` loop
    :meth:`WaitGate.checkIfFPUInstructionShouldStall` implements. That is the
    instruction's entire reason to exist: it is the debug read that can look at
    a bank the unpackers still own.

    Asserted against the list rather than by issuing one, because
    ``matrix.py`` has no ``MOVDBGA2D`` handler -- an issued one raises
    ``NotImplementedError``, so the gate is the only place its behaviour is
    observable today. The companion assertion that ``MOVA2D`` *is* gated is
    what stops this being satisfiable by emptying the list.
    """
    srca_gated, srcb_gated = WaitGate.MATH_ALLOWED_CLIENT_INSTRUCTIONS
    assert "MOVDBGA2D" not in srca_gated
    assert "MOVDBGA2D" not in srcb_gated
    assert "MOVA2D" in srca_gated


def test_setrwc_does_not_wait_at_the_gate():
    """SETRWC reads no Src rows, so it is not a gated instruction.

    That is what makes it -- and CLEARDVALID -- able to reach the release check
    below with an unowned bank, where a gated instruction cannot.
    """
    cp = _coprocessor()
    thread = cp.getThread(1)
    thread.push_wait_gate_instruction(_op_binary("SETRWC") << 24)
    _run(cp, 20)
    assert not thread.wait_gate_instruction_fifo


# --- the release side: clearing dvalid on a bank the FPU does not own --------


def _issue_to_matrix_unit(backend, word, thread=0):
    assert backend.issueInstruction(word, thread)
    backend.matrix_unit.clock_tick(0)


def _cleardvalid(cleardvalid=0x3, reset=0):
    return (_op_binary("CLEARDVALID") << 24) | (cleardvalid << 22) | reset


def test_cleardvalid_on_an_unowned_bank_is_refused():
    backend = _coprocessor().getBackend()
    with pytest.raises(SrcDvalidError, match="SrcA bank 0"):
        _issue_to_matrix_unit(backend, _cleardvalid(cleardvalid=0x1))


def test_cleardvalid_twice_is_refused():
    """The shape that wedges a card: one acquire, two releases.

    ``perfbench/tensixbench``'s ``UNPACR_NOP``+``set_dvalid`` setup acquires and
    the trailing ``CLEARDVALID`` releases; a second release with no intervening
    acquire leaves the pointers a bank apart, and the next acquire waits for a
    hand-back that never comes (``docs/plans/matrix-unit-thread-contention.md``).
    """
    backend = _coprocessor().getBackend()
    _hand_banks_to_fpu(backend)
    _issue_to_matrix_unit(backend, _cleardvalid(cleardvalid=0x1))
    assert backend.matrix_unit.srcABank == 1
    _issue_to_matrix_unit(backend, _cleardvalid(cleardvalid=0x1))
    assert backend.matrix_unit.srcABank == 0
    # Bank 0 was released by the first CLEARDVALID and never re-acquired.
    with pytest.raises(SrcDvalidError, match="CLEARDVALID"):
        _issue_to_matrix_unit(backend, _cleardvalid(cleardvalid=0x1))


def test_a_legal_release_is_silent_and_flips_the_bank():
    backend = _coprocessor().getBackend()
    _hand_banks_to_fpu(backend)
    _issue_to_matrix_unit(backend, _cleardvalid(cleardvalid=0x3))
    assert backend.getSrcA(0).allowedClient is SrcRegister.SrcClient.Unpackers
    assert backend.getSrcB(0).allowedClient is SrcRegister.SrcClient.Unpackers
    assert backend.matrix_unit.srcABank == 1
    assert backend.matrix_unit.srcBBank == 1


def test_clr_dvalid_disable_suppresses_the_release_and_the_check():
    """``CLR_DVALID_Src{A,B}_Disable`` skips the hand-back but not the flip.

    The check lives inside the suppressed branch precisely because a thread
    that has disabled the release is not claiming to own anything.
    """
    backend = _coprocessor().getBackend()
    _set_thread_config(backend, "CLR_DVALID_SrcA_Disable", 1)
    word = (_op_binary("SETRWC") << 24) | (0x1 << 22)  # clear_ab_vld = SrcA
    _issue_to_matrix_unit(backend, word)
    assert backend.getSrcA(0).allowedClient is SrcRegister.SrcClient.Unpackers
    assert backend.matrix_unit.srcABank == 1


def test_the_check_can_be_turned_off():
    backend = _coprocessor().getBackend()
    matrix_mod.set_dvalid_checking_enabled(False)
    try:
        _issue_to_matrix_unit(backend, _cleardvalid(cleardvalid=0x1))
    finally:
        matrix_mod.set_dvalid_checking_enabled(True)
    assert backend.matrix_unit.srcABank == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
