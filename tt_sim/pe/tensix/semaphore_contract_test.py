"""A ``SEMPOST`` that loses information stops; one that does not stays silent.

Two shapes are refused, and the difference between them is the whole design
(the reasoning lives in :mod:`tt_sim.pe.tensix.semaphore_contract`):

* a post at ``Value == 15``, or a get at ``Value == 0``, where the operation is
  discarded by the hardware itself;
* a post at or above a ``Max`` some ``SEMINIT`` declared, where the increment
  happens but the producer has issued past its own ``SEMWAIT`` C1 gate.

The second is the one that bites: a compute kernel that hoists
``tile_regs_acquire()`` out of its output-tile loop removes the math thread's
back-pressure on ``MATH_PACK``, and once the packer falls behind, math wraps
onto a Dst bank the packer has not drained. tt-sim used to return the resulting
garbage; ttsim stopped with ``tensix_sempost: sem=2 sem_max=2``.

The silence half matters as much as the firing half, so the negative cases here
are not decoration: a ``Max`` rule applied to a semaphore no ``SEMINIT`` ever
configured, or to the memory-mapped RISC-V write path, fires on working
tt-metal kernels (377 times across the in-tree corpus, all on ``UNPACK_SYNC``),
and a guard that fires on working kernels gets switched off.

Run standalone (``python3 -m tt_sim.pe.tensix.semaphore_contract_test``) or
under pytest.
"""

import pytest

from tt_sim.arch import BLACKHOLE_PROFILE, WORMHOLE_PROFILE
from tt_sim.pe.tensix import semaphore_contract
from tt_sim.pe.tensix.semaphore_contract import SemaphoreContractError
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.util.conversion import conv_to_bytes

#: Opcodes from ``tensix_instructions.yaml``; ``sem_sel`` is a one-hot mask at
#: bit 2, and ``SEMINIT`` carries ``init_value`` at bit 16, ``max_value`` at 20.
SEMINIT_OP = 0xA3 << 24
SEMPOST_OP = 0xA4 << 24
SEMGET_OP = 0xA5 << 24


#: ``PC_BUF_SEMAPHORE_BASE`` offset within the Sync Unit's mapping: the unit
#: indexes ``SemaphoreAccess[i]`` at ``addr / 4``. Writing an odd value is a
#: SEMGET, an even one a SEMPOST (``SyncUnit.md``, "RISCV access to
#: semaphores").
def _mmio_post(sync, index):
    sync.write(index * 4, conv_to_bytes(0, 4))


def _mmio_get(sync, index):
    sync.write(index * 4, conv_to_bytes(1, 4))


def _seminit(sem_mask, init_value, max_value):
    return SEMINIT_OP | (max_value << 20) | (init_value << 16) | (sem_mask << 2)


def _sempost(sem_mask):
    return SEMPOST_OP | (sem_mask << 2)


def _semget(sem_mask):
    return SEMGET_OP | (sem_mask << 2)


def _sync_unit(profile=WORMHOLE_PROFILE):
    cop = TensixCoProcessor(
        None,
        profile.tensix_cfg_state_size,
        profile.tensix_thd_state_size,
        blackhole=profile is BLACKHOLE_PROFILE,
    )
    return cop, cop.getBackend().getSyncUnit()


def _issue(cop, thread_id, *instruction_words, cycles=12):
    """Push real instruction words through a thread's frontend and run."""
    thread = cop.getThread(thread_id)
    clocks = cop.getClocks()
    cycle = 0
    for word in instruction_words:
        thread.write(0, conv_to_bytes(word, 4))
        for _ in range(cycles):
            for clockable in clocks:
                clockable.clock_tick(cycle)
            cycle += 1


# --------------------------------------------------------------------------
# The acceptance shape: a post past a declared Max.
# --------------------------------------------------------------------------


def test_a_sempost_past_a_declared_max_stops_and_names_everything():
    """``SEMINIT(Max=2)``, post twice to fill it, and the third post refuses.

    This is ``optests/hoistacquire``'s ``stall`` mode at its smallest:
    ``MATH_PACK`` is semaphore 1, its ``Max`` is 2 under ``DstSync::SyncHalf``,
    and the math thread reaches ``Value == 2`` only by walking past the
    ``SEMWAIT`` on C1 that ``tile_regs_acquire()`` would have issued. tt-sim
    used to increment to 3 and carry on returning numbers.
    """
    cop, sync = _sync_unit()
    _issue(cop, 1, _seminit(1 << 1, 0, 2), _sempost(1 << 1), _sempost(1 << 1))
    assert sync.getSemaphore(1).value == 2

    with pytest.raises(SemaphoreContractError) as excinfo:
        _issue(cop, 1, _sempost(1 << 1))
    message = str(excinfo.value)

    # Everything a reader needs to act, per the house guard style: which
    # semaphore, its value, the bound it passed, who issued it, what the
    # hardware does, and how to turn the check off.
    assert "semaphore 1" in message
    assert "MATH_PACK" in message  # tt-metal's name for it, attributed as such
    assert "Value is 2" in message
    assert "Max 2" in message
    assert "Tensix thread 1" in message
    assert "no effect on SEMPOST" in message  # what hardware does
    assert "not cycle-accurate" in message  # why tt-sim will not guess
    assert semaphore_contract.DISABLE_ENV_VAR in message

    # And the refusal is clean: the atomic block did not half-apply.
    assert sync.getSemaphore(1).value == 2


def test_posting_up_to_max_is_silent():
    """The bound is ``>= Max``, so ``Max`` posts from empty are all fine.

    A guard that fired one post early would stop every correct producer on its
    last token, which is the most common shape there is.
    """
    cop, sync = _sync_unit()
    _issue(cop, 1, _seminit(1 << 1, 0, 4))
    for expected in range(1, 5):
        _issue(cop, 1, _sempost(1 << 1))
        assert sync.getSemaphore(1).value == expected


def test_a_get_makes_room_for_another_post():
    """The consumer draining a token re-opens the gate, as it does on silicon.

    Pins that the check reads live state rather than counting posts: this is
    the steady state of every correct producer/consumer pair, and it must be
    able to run indefinitely.
    """
    cop, sync = _sync_unit()
    _issue(cop, 1, _seminit(1 << 1, 0, 2), _sempost(1 << 1), _sempost(1 << 1))
    for _ in range(5):
        _issue(cop, 2, _semget(1 << 1))
        _issue(cop, 1, _sempost(1 << 1))
        assert sync.getSemaphore(1).value == 2


def test_a_seminit_reopens_the_bound():
    """``SEMINIT`` sets ``Value`` as well as ``Max``, so it is never stuck."""
    cop, sync = _sync_unit()
    _issue(cop, 1, _seminit(1 << 1, 2, 2))
    with pytest.raises(SemaphoreContractError):
        _issue(cop, 1, _sempost(1 << 1))
    _issue(cop, 1, _seminit(1 << 1, 0, 2), _sempost(1 << 1))
    assert sync.getSemaphore(1).value == 1


# --------------------------------------------------------------------------
# The silences that keep the guard usable.
# --------------------------------------------------------------------------


def test_a_semaphore_no_seminit_configured_is_not_bounded():
    """The false positive that would have disqualified the design.

    ``Max`` powers on at zero, and tt-metal's LLK ``SEMINIT``s exactly one of
    the eight semaphores. Every post to any of the others -- ``UNPACK_SYNC``,
    ``MATH_DONE``, ``UNPACK_TO_DEST`` -- is a post at ``Value >= Max`` on a
    working kernel. Measured on the in-tree corpus: 377 such posts, none of
    them a defect.
    """
    cop, sync = _sync_unit()
    assert sync.getSemaphore(5).max == 0
    for expected in range(1, 8):
        _issue(cop, 0, _sempost(1 << 5))
        assert sync.getSemaphore(5).value == expected


def test_the_memory_mapped_write_path_is_not_bounded_by_max():
    """A baby RISC-V has no ``SEMWAIT``, so ``Max`` is not its back-pressure.

    It polls with ``lw`` (``SyncUnit.md``), C1 is unavailable to it, and ttsim
    likewise checks only 15/0 on this path.
    """
    cop, sync = _sync_unit()
    _issue(cop, 1, _seminit(1 << 3, 0, 1))
    _mmio_post(sync, 3)
    _mmio_post(sync, 3)  # now past Max, and deliberately allowed
    assert sync.getSemaphore(3).value == 2


# --------------------------------------------------------------------------
# Saturation, on both paths and on both architectures.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [WORMHOLE_PROFILE, BLACKHOLE_PROFILE])
def test_a_sempost_at_fifteen_stops(profile):
    """``SEMPOST.md`` is shared between the arches, so this must be too."""
    cop, sync = _sync_unit(profile)
    sync.getSemaphore(6).value = 15
    with pytest.raises(SemaphoreContractError) as excinfo:
        _issue(cop, 2, _sempost(1 << 6))
    assert "remains at 15" in str(excinfo.value)
    assert sync.getSemaphore(6).value == 15


def test_a_semget_at_zero_stops():
    """The mirror image: a consumer took a token nobody posted."""
    cop, sync = _sync_unit()
    with pytest.raises(SemaphoreContractError) as excinfo:
        _issue(cop, 2, _semget(1 << 4))
    assert "remains at zero" in str(excinfo.value)


def test_the_memory_mapped_path_saturates_too():
    """tt-metal's own LLK asserts on both of these; so does ttsim."""
    _cop, sync = _sync_unit()
    sync.getSemaphore(2).value = 15
    with pytest.raises(SemaphoreContractError):
        _mmio_post(sync, 2)

    sync.getSemaphore(2).value = 0
    with pytest.raises(SemaphoreContractError):
        _mmio_get(sync, 2)


def test_a_multi_semaphore_post_checks_every_selected_semaphore():
    """``sem_sel`` is a mask, not an index. Every in-tree kernel uses one-hot,
    which is exactly why a walk that only looked at the first bit would go
    unnoticed."""
    cop, sync = _sync_unit()
    _issue(cop, 1, _seminit(0xFF, 0, 3))
    sync.getSemaphore(6).value = 3
    with pytest.raises(SemaphoreContractError) as excinfo:
        _issue(cop, 1, _sempost(0xFF))
    assert "semaphore 6" in str(excinfo.value)
    # Nothing moved: the check runs over the whole mask before the increments.
    assert [sync.getSemaphore(i).value for i in range(8)] == [0] * 6 + [3, 0]


# --------------------------------------------------------------------------
# The escape hatch.
# --------------------------------------------------------------------------


def test_the_environment_variable_restores_the_old_silence():
    """Same convention as ``TT_SIM_DISABLE_ALIGNMENT_CHECKS``."""
    cop, sync = _sync_unit()
    _issue(cop, 1, _seminit(1 << 1, 2, 2))
    semaphore_contract.set_checking_enabled(False)
    try:
        _issue(cop, 1, _sempost(1 << 1))
        assert sync.getSemaphore(1).value == 3
    finally:
        semaphore_contract.set_checking_enabled(True)


def test_the_environment_variable_is_read_from_the_environment(monkeypatch):
    """The cached flag and the variable that sets it stay in step."""
    monkeypatch.setenv(semaphore_contract.DISABLE_ENV_VAR, "yes")
    try:
        assert not semaphore_contract.refresh_from_env()
        assert not semaphore_contract.checking_enabled()
    finally:
        monkeypatch.delenv(semaphore_contract.DISABLE_ENV_VAR)
        assert semaphore_contract.refresh_from_env()


def main():
    class _Monkeypatch:
        def __init__(self):
            self._undo = []

        def setenv(self, name, value):
            import os

            self._undo.append((name, os.environ.get(name)))
            os.environ[name] = value

        def delenv(self, name):
            import os

            os.environ.pop(name, None)

    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        cases = [
            case
            for mark in marks
            if mark.name == "parametrize"
            for case in mark.args[1]
        ]
        if "monkeypatch" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            fn(_Monkeypatch())
            continue
        for case in cases or [None]:
            fn() if case is None else fn(case)
    print(
        "semaphore_contract_test OK: a SEMPOST past a declared Max or at 15 "
        "raises; posts within the bound, unconfigured semaphores and the "
        "memory-mapped path stay silent"
    )


if __name__ == "__main__":
    main()
