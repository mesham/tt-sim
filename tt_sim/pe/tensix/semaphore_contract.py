"""Tensix semaphore bound checking.

A Tensix semaphore is four bits of ``Value`` and four bits of ``Max``.
``SEMPOST`` increments ``Value``, ``SEMGET`` decrements it, ``SEMINIT`` sets
both, and ``SEMWAIT`` compares ``Value`` against ``0`` or against ``Max``. This
module refuses the two ways a program can drive that counter outside the
contract it wrote for itself.

What the ISA documentation actually says
----------------------------------------
It is worth being precise, because the answer is *not* that the hardware
faults. ``WormholeB0/TensixTile/TensixCoprocessor/SEMPOST.md`` (the Blackhole
page is a stub pointing at it, "The behavior is identical between Wormhole and
Blackhole") says:

    Various semaphores are selected using a bitmask, and the selected
    semaphores have their ``Value`` incremented (unless their ``Value`` is
    already 15, in which case it remains at 15).

with the functional model ``if (SemaphoreMask.Bit[i] && Semaphores[i].Value <
15) { Semaphores[i].Value += 1; }``. ``SEMGET.md`` is the mirror image at zero.
Neither page uses the words ``UndefinedBehavior`` or ``NonContractualBehavior``
that these documents use elsewhere and freely: **saturation is documented,
deliberate, defined behaviour**, and tt-sim models it exactly. It still does —
nothing in this module changes what ``SEMPOST`` computes.

``SEMINIT.md`` is equally explicit about the other bound:

    Note that ``Max`` is only subsequently used by ``SEMWAIT``; it has no
    effect on ``SEMPOST``.

So neither check below is reporting that tt-sim was about to compute the wrong
``Value``. Both are reporting that the *program* has lost information.

The two losses
--------------
**A post that saturates, or a get that underflows.** At ``Value == 15`` the
increment is dropped on the floor; at ``Value == 0`` so is the decrement. The
count no longer tracks the thing it was counting and no later operation can
recover it. tt-metal's own LLK asserts against precisely this, on the
memory-mapped path, in ``tt_llk_<arch>/common/inc/ckernel.h``::

    // The value is capped at 15; writing when already at 15 would silently
    // have no effect, so the assert guards against that misuse.
    inline void semaphore_post(const std::uint8_t index)
    {
        LLK_ASSERT(semaphore_read(index) < semaphore::SEMAPHORE_MAX_VALUE,
                   "Semaphore must not be already at max value.");

("misuse" is the vendor's word, and ``SEMAPHORE_MAX_VALUE`` is 15, not the
per-semaphore ``Max``.) The vendor reference simulator ttsim refuses it too, on
the same path — ``tensix_pc_buf_wr32`` in ``src/tile.cpp``::

    TTSIM_VERIFY(p_tile->tensix[tensix_id].sem[sem_index] < 15,
                 NonContractualBehavior, "sem%d overflow", sem_index);

**A post above a ``Max`` the program declared itself.** Here the increment
happens; what has been lost is the bound. ``Max`` has exactly one consumer —
``SEMWAIT``'s condition C1, "*Keep on waiting and blocking if... any of the
semaphores selected by ``SemaphoreMask`` have ``Value >= Max``*"
(``SyncUnit.md``) — so ``Max`` *is* the producer's own back-pressure threshold
and nothing else. A producer that follows the discipline waits at C1 until
``Value < Max`` and then posts, so ``Value`` can never exceed ``Max``. Reaching
``Max + 1`` is therefore proof that the producer issued past its own gate.

That is the check ttsim spells ``NonContractualBehavior: tensix_sempost:
sem=%d sem_max=%d`` (``TENSIX_EXECUTE_SEMPOST`` in ``src/tensix.cpp``), and its
``sem_max`` is this ``Max`` — the value the last ``SEMINIT`` wrote, not the
architectural 15. tt-sim has always modelled that field
(``TensixSyncUnit.TTSemaphore.max``, set by ``handle_seminit``, read by the
Wait Gate for C1); it simply never consulted it on the post side.

Why the second check earns a raise even though hardware carries on
------------------------------------------------------------------
Because what it detects is a *race*, and a race is the one class of question
tt-sim is structurally unable to answer. tt-sim is not cycle-accurate and does
not claim to be. While a program stays inside its own synchronisation, that
does not matter: the answer is decided by the handshakes, not by the schedule,
so tt-sim's schedule being wrong costs nothing. The moment a thread posts past
its declared ``Max`` it has stepped outside those handshakes, and from there the
computed values are decided by tt-sim's arbitrary interleaving instead. The
number tt-sim would go on to return is then not a prediction of what a card
does — it is an artefact, and it is indistinguishable, from the outside, from a
number that means something. Stopping is worth more than returning it.

This is not hypothetical. ``optests/hoistacquire`` hoists ``tile_regs_acquire``
out of a matmul's output-tile loop, which removes the math thread's only
back-pressure — that acquire is a ``SEMWAIT`` on C1 of ``MATH_PACK``, the
semaphore counting Dst banks the packer has yet to drain. With the packer
keeping up, nothing goes wrong and (measured, both architectures) nothing here
fires. With the writer stalled 50 iterations per tile, the math thread runs
ahead, wraps onto a Dst bank the packer has not drained, and every one of the
6144 elements of the hoisted half comes back wrong — and this check fires five
times, on ``MATH_PACK``, at ``Value`` 2, 2, 3, 3, 4 against ``Max`` 2. It is
exactly the point at which the two simulators diverged: ttsim stopped, tt-sim
returned the corruption.

Where the ``Max`` check applies, and where it deliberately does not
-------------------------------------------------------------------
The saturation check (post at 15, get at 0) applies to every path. The ``Max``
check applies **only to the Tensix ``SEMPOST`` instruction, and only to a
semaphore some ``SEMINIT`` has actually configured**. Both restrictions are
load-bearing, and both were measured rather than guessed:

* **Only after a ``SEMINIT``.** ``Max`` powers on at zero, and tt-metal's LLK
  initialises exactly one of the eight semaphores — ``MATH_PACK``, in
  ``_llk_math_pack_sync_init_``, to ``Max`` 1 (``DstSync::SyncFull``) or 2
  (``SyncHalf``). Every other semaphore it uses is posted against a ``Max``
  that is still its power-on zero, so ``Value >= Max`` is true on the very
  first post. Across the whole in-tree corpus that shape occurs 377 times, all
  of them on ``UNPACK_SYNC``, all in working kernels. A rule that did not
  require a ``SEMINIT`` would fire on every one of them.
* **Tensix ``SEMPOST`` only, not the memory-mapped RISC-V write.** A baby
  RISC-V has no ``SEMWAIT``; it polls with ``semaphore_read`` (``SyncUnit.md``:
  "*RISCV core can perform ``lw`` instructions in a polling loop*"). C1 is not
  available to it, so ``Max`` is not its back-pressure and passing it says
  nothing about that core's discipline. ttsim draws its two lines in exactly
  these places as well: its memory-mapped path checks 15 and 0, and only its
  ``SEMPOST`` instruction checks ``sem_max``.

What fires on the in-tree corpus
--------------------------------
Nothing. Measured by instrumenting both paths and running the full suite (1998
tests), the cost-model gate (every replay guard on both architectures, model on
and off), the fast-tier upstream sweep on both architectures, and
``optests/hoistacquire`` in its passing modes: zero posts at 15, zero gets at 0,
and zero posts above a configured ``Max``. The only ``Max`` events at all are
the 377 uninitialised-semaphore ones described above, which this module does
not check, and the five ``hoistacquire`` ones, which are the corruption.

Disabling
---------
Checking is on by default and is disabled by setting
``TT_SIM_DISABLE_SEMAPHORE_CHECKS`` to a truthy value (``1``/``true``/``yes``/
``on``, case-insensitive), following the same convention as
``TT_SIM_DISABLE_ALIGNMENT_CHECKS``.
"""

from __future__ import annotations

import os

#: Environment variable that turns semaphore bound checking off.
DISABLE_ENV_VAR = "TT_SIM_DISABLE_SEMAPHORE_CHECKS"

_TRUTHY = {"1", "true", "yes", "on"}

#: The architectural ceiling: ``Value`` is four bits and ``SEMPOST`` saturates
#: rather than wrapping (``SEMPOST.md``).
VALUE_MAX = 15

#: What tt-metal's LLK calls each semaphore index, from
#: ``tt_llk_<arch>/common/inc/ckernel_structs.h``. Quoted in the failure
#: message because "semaphore 1" is not something a kernel author can act on
#: whereas "MATH_PACK" is. Attributed, not asserted: the indices are tt-metal's
#: convention, not the ISA's -- the hardware has eight anonymous counters, and
#: a program that is not tt-metal may use them for anything.
LLK_SEMAPHORE_NAMES = {
    0: "FPU_SFPU",
    1: "MATH_PACK",
    2: "UNPACK_TO_DEST",
    3: "UNPACK_OPERAND_SYNC",
    4: "PACK_DONE",
    5: "UNPACK_SYNC",
    6: "UNPACK_MATH_DONE",
    7: "MATH_DONE",
}


class SemaphoreContractError(RuntimeError):
    """Raised when a Tensix semaphore operation is dropped by saturation, or
    when a ``SEMPOST`` carries a semaphore past the ``Max`` its own ``SEMINIT``
    declared. See the module docstring for which is which and why both stop."""


def _env_disabled() -> bool:
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY


#: Cached at import so the common path does not hit ``os.environ`` per
#: instruction. Call :func:`refresh_from_env` after mutating the environment.
_disabled = _env_disabled()


def refresh_from_env() -> bool:
    """Re-read :data:`DISABLE_ENV_VAR`. Returns whether checking is now enabled."""
    global _disabled
    _disabled = _env_disabled()
    return not _disabled


def checking_enabled() -> bool:
    """Whether semaphore bound checking is currently active."""
    return not _disabled


def set_checking_enabled(enabled: bool) -> None:
    """Force checking on or off, ignoring the environment (for tests)."""
    global _disabled
    _disabled = not enabled


def _name(index):
    llk = LLK_SEMAPHORE_NAMES.get(index)
    return (
        f"semaphore {index} (tt-metal's LLK calls it {llk})"
        if llk
        else f"semaphore {index}"
    )


def check_post(index, value, *, issuer, declared_max=None):
    """Raise :class:`SemaphoreContractError` if this ``SEMPOST`` is out of contract.

    ``value`` is the semaphore's value *before* the increment. ``issuer`` names
    who is posting, for the message. ``declared_max`` is the semaphore's
    ``Max`` when — and only when — the ``Max`` rule applies to this caller: a
    Tensix ``SEMPOST`` against a semaphore some ``SEMINIT`` has configured.
    Pass ``None`` (the default) otherwise, which leaves only the saturation
    check; the module docstring says why the memory-mapped path and the
    never-initialised semaphores are excluded.
    """
    if _disabled:
        return
    if value >= VALUE_MAX:
        raise SemaphoreContractError(
            f"SEMPOST from {issuer} on {_name(index)}, whose Value is already "
            f"{VALUE_MAX}. Per SEMPOST.md the increment is discarded -- 'unless "
            f"their Value is already 15, in which case it remains at 15' -- so a "
            f"token has been produced that no SEMGET can ever consume, and the "
            f"count has stopped tracking whatever it was counting. Hardware does "
            f"not fault here and neither does tt-sim's SEMPOST: the value stays "
            f"at {VALUE_MAX} and execution would carry on with a semaphore that "
            f"silently means nothing. tt-metal's own LLK asserts against this "
            f"case (ckernel.h, 'Semaphore must not be already at max value.'), as "
            f"does ttsim ('sem{index} overflow'). "
            f"Set {DISABLE_ENV_VAR}=1 to disable semaphore checking."
        )
    if declared_max is not None and value >= declared_max:
        raise SemaphoreContractError(
            f"SEMPOST from {issuer} on {_name(index)}, whose Value is {value} and "
            f"whose SEMINIT declared Max {declared_max}. Max has exactly one "
            f"consumer -- SEMWAIT's condition C1, 'keep on waiting and blocking "
            f"if ... Value >= Max' (SyncUnit.md) -- so it is this producer's own "
            f"back-pressure threshold, and a producer that honours it can never "
            f"post above it. Reaching Max means the thread issued past its own "
            f"gate: the resource the semaphore was counting is already fully "
            f"handed out, and whatever it protects is now being reused under its "
            f"holder. SEMINIT.md says Max 'has no effect on SEMPOST', so hardware "
            f"increments regardless and tt-sim still models that faithfully -- "
            f"the objection is not to the arithmetic but to what follows it. From "
            f"here the computed values are decided by the relative timing of the "
            f"threads, and tt-sim is not cycle-accurate, so any result it went on "
            f"to return would be an artefact of its own interleaving rather than "
            f"a prediction of what a card does. ttsim refuses the same "
            f"instruction ('tensix_sempost: sem={value} sem_max={declared_max}'). "
            f"The usual cause is back-pressure removed from a producer loop -- a "
            f"tile_regs_acquire() hoisted out of an output-tile loop is exactly "
            f"this, on MATH_PACK. "
            f"Set {DISABLE_ENV_VAR}=1 to disable semaphore checking."
        )


def check_get(index, value, *, issuer):
    """Raise :class:`SemaphoreContractError` if this ``SEMGET`` underflows.

    ``value`` is the semaphore's value *before* the decrement. The mirror of
    :func:`check_post`'s saturation arm: at zero the decrement is discarded, so
    a consumer has taken a token that was never produced.
    """
    if _disabled:
        return
    if value > 0:
        return
    raise SemaphoreContractError(
        f"SEMGET from {issuer} on {_name(index)}, whose Value is already 0. Per "
        f"SEMGET.md the decrement is discarded -- 'unless their Value is already "
        f"zero, in which case it remains at zero' -- so a consumer has taken a "
        f"token no producer ever posted, and the count has stopped tracking "
        f"whatever it was counting. Hardware does not fault here and neither "
        f"does tt-sim's SEMGET: the value stays at 0 and execution would carry "
        f"on. tt-metal's own LLK asserts against this case (ckernel.h, "
        f"'Semaphore must not be already at 0.'), as does ttsim "
        f"('sem{index} underflow'). "
        f"Set {DISABLE_ENV_VAR}=1 to disable semaphore checking."
    )
