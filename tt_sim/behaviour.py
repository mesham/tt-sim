"""What *this* tt-sim gets right — an assertion target for external suites.

Why this is not ``__version__``
-------------------------------
The compiler team ran their multicast suite against a tt-sim that delivered
correctly-encoded NoC 1 multicasts to nobody and reported success. Every run was
green because nothing was delivered, not because anything worked, and there was
no way to tell from the outside. Their question was "is there a version marker
we can assert on?", and the honest answer is that a version number is the wrong
shape for it: ``0.7.3`` tells you which build you have, never whether that build
has the fix, and a consumer who pins ``>= 0.7.3`` has encoded a fact about our
release history into their test suite rather than a fact about what they need.

So what is published here is **behaviour**: named guarantees an outside suite
may depend on. A name means the same thing forever; it is either present or it
is not; and asserting on it says what the suite actually depends on.

Most entries are of one shape — tt-sim used to get something silently wrong and
now refuses it — and for a long time that was the whole registry. It is
deliberately not the definition. ``dram-interleaved-bank-distinctness`` is a
**modelling property**: nothing raises, no environment variable switches it off,
and there is no exception class behind it. It is here because the test it
replaces was one an outside consumer was writing *by hand* — deriving, from our
internals and our shipped SoC descriptors, that DRAM banks are separate stores
at separate coordinates — in order to trust their own gate. That derivation is
precisely what this registry exists to absorb: the question a consumer needs
answered is "can a run against this build see the thing I am testing?", and a
substrate that is missing makes a suite vacuous just as surely as a guard that
does not fire. So the bar for an entry is that a consumer would otherwise have
to establish it themselves, not that something throws.

The difference does change how such an entry is *used*, and the guarantee text
says so: a property marker tells you the substrate is there, it does not tell
you when you have broken your own kernel.

Asserting on it
---------------
From Python, in a suite's session setup::

    from tt_sim.behaviour import require

    require("noc1-multicast-corner-order")

which raises :class:`UnsupportedBehaviour` — naming what is missing, what this
build does have, and where to read about it — rather than letting the suite
collect another set of vacuous green results. Against a tt-sim older than this
module the ``import`` fails instead, which is the same outcome at the same
moment; a consumer should not wrap it in ``except ImportError``, because a
checkout without this module is precisely the one whose results are suspect.

From a shell, a Makefile, or a C++ harness's test runner::

    python3 -m tt_sim.behaviour --require noc1-multicast-corner-order

exits 0 when every named behaviour is present and 1 when any is missing, with
the missing names on stderr. ``python3 -m tt_sim.behaviour`` with no arguments
lists what this build guarantees, one name per line, so a consumer can record
the set a run was made against.

:func:`supports` is the non-raising form, for a suite that would rather skip
than fail.

This module imports nothing from the rest of tt-sim and nothing outside the
standard library, so asserting on it costs no simulator construction and cannot
be broken by an import cycle.

Adding one, and why forgetting is caught
----------------------------------------
A registry nobody is forced to update decays into a lie, and this one is worth
more than the average: its whole purpose is to be trusted by someone who cannot
see our tree. Two guards hold it to the code.

* **Forward.** Every entry names the test that pins its guarantee, and
  ``behaviour_test.py`` imports that module and looks the function up. An entry
  whose test is renamed or deleted fails immediately, so no guarantee can
  outlive the thing that checks it.
* **Backward.** The registry is *not* the only place a new guard has to be
  mentioned, because that is the mention people forget. ``behaviour_test.py``
  parses every non-test module under ``tt_sim/`` for exception classes and
  requires each one to be either registered here or listed in
  ``_NOT_A_GUARANTEE`` with a reason. Adding a loudness guard to the simulator
  therefore turns the suite red until somebody has decided, in writing, whether
  an outside consumer would want to assert on it. That decision is the point;
  the registry is just where the "yes" answers land.

The dates are the day the guarantee first held in ``main``, recorded because it
is occasionally useful for bisection. They are not a version and nothing should
compare them.
"""

from __future__ import annotations

import argparse
import sys
from types import MappingProxyType
from typing import NamedTuple

#: Where the prose lives, quoted in the failure message so a consumer who hits
#: it has somewhere to go.
DOCS = "docs/running-tt-metal-on-the-simulator.md, section 7"


class Behaviour(NamedTuple):
    """One named, externally assertable guarantee about this simulator."""

    #: Stable identifier. Never renamed and never reused for another meaning —
    #: a consumer's test file outlives our refactors.
    name: str
    #: What a run against this build will do, in terms a consumer can observe
    #: from outside: what raises, what is delivered, what is counted.
    guarantee: str
    #: ``YYYY-MM-DD`` the guarantee first held in ``main``. Informational.
    since: str
    #: ``module:function`` of the test that pins it. Checked to exist.
    pinned_by: str


_BEHAVIOURS = (
    Behaviour(
        name="noc1-multicast-corner-order",
        guarantee=(
            "A NoC multicast whose rectangle corners are ordered for the wrong "
            "NoC raises tt_sim.network.multicast_order.MulticastOrderError at "
            "the point of issue. Before this, both wrong orderings completed "
            "green: low-corner-first on a translated NoC 1 was delivered to "
            "every destination (silicon delivers it to none and the sender's "
            "write barrier never retires), and the reverse enumerated an empty "
            "rectangle, so the barrier retired having sent nothing."
        ),
        since="2026-08-19",
        pinned_by=(
            "tt_sim.network.multicast_order_test:"
            "test_a_translated_noc1_multicast_written_low_first_raises"
        ),
    ),
    Behaviour(
        name="riscv-ebreak-halts",
        guarantee=(
            "A baby RISC-V executing ebreak raises "
            "tt_sim.pe.rv.breakpoint.RiscvBreakpoint naming the core and PC. "
            "ebreak is what a kernel ASSERT(), an LLK_ASSERT under "
            "TT_METAL_LLK_ASSERTS=1, and __builtin_trap() all lower to, and "
            "tt-sim used to decode it as a no-op — so a program whose own "
            "authors had declared a state impossible ran past it and reported "
            "success while the same binary stopped dead on silicon. Set "
            "TT_SIM_IGNORE_EBREAK=1 to restore the old skip-and-continue."
        ),
        since="2026-08-04",
        pinned_by="tt_sim.pe.rv.breakpoint_test:test_ebreak_names_the_core_and_pc",
    ),
    Behaviour(
        name="dram-interleaved-bank-distinctness",
        guarantee=(
            "Every DRAM bank tt-metal can interleave a buffer over is separate "
            "storage reachable at its own NoC 0 coordinate: 12 banks on "
            "Wormhole (6 channels x two 1 GiB dram_views), 8 on Blackhole (8 "
            "channels x one view), counted from the shipped "
            "driver/<arch>/soc_descriptor.yaml. A write to one bank is "
            "invisible in every other; each bank's descriptor coordinate is a "
            "distinct cell that resolves to its own channel's tile (including "
            "the second worker-visible endpoint of a Wormhole channel); and "
            "each channel is modelled whole, so every view's range is backed. "
            "A host scatter and a kernel gather that disagree about page size, "
            "bank count or a bank's coordinate therefore corrupt data, as they "
            "do on silicon, instead of aliasing onto one flat store and coming "
            "back right. NOT covered: which page lands in which bank — that "
            "arithmetic is tt-metal's on both sides (host "
            "WriteToDeviceInterleavedContiguous, device InterleavedAddrGen) and "
            "never tt-sim's — nor bank ordering, nor any claim about a "
            "descriptor you supply yourself. Unlike the other entries here, "
            "nothing raises and no environment variable disables it: this is a "
            "modelling property, so it says the substrate your gate needs is "
            "present, not that you will be told when your kernel is wrong."
        ),
        since="2026-08-19",
        pinned_by=(
            "tt_sim.device.dram_banks_test:"
            "test_every_bank_is_its_own_storage_at_its_own_coordinate"
        ),
    ),
    Behaviour(
        name="tensix-latched-wait-survives-stallwait",
        guarantee=(
            "A Tensix STALLWAIT issued while an earlier SEMWAIT or STALLWAIT "
            "is latched and unsatisfied waits at its thread's Wait Gate "
            "instead of replacing it, so a compute kernel's Dst handshake "
            "holds however the LLK init calls are ordered around "
            "tile_regs_wait(). STALLWAIT.md's block-mask table ticks "
            "STALLWAIT in all nine columns, on both architectures, and it is "
            "the only instruction whose row is; tt-sim used to catch it by "
            "its execution unit (Sync Unit, block bit B1) alone, so a "
            "STALLWAIT behind a SEMWAIT whose block mask named the TDMA units "
            "walked past and that semaphore wait was silently forgotten. "
            "Concretely: pack_untilize_dest_init called after "
            "tile_regs_wait() rather than before matmul_init dropped the "
            "packer's wait on MATH_PACK, so a K >= 2 matmul packed a Dst that "
            "was still mid-accumulation -- wrong values, no fault, and "
            "correct on silicon and on ttsim. Like "
            "dram-interleaved-bank-distinctness and unlike the raising "
            "entries here, nothing throws and no environment variable "
            "disables it: this is a modelling property, so it says the "
            "ordering your kernel depends on is honoured, not that you will "
            "be told when your kernel is wrong."
        ),
        since="2026-08-20",
        pinned_by=(
            "tt_sim.pe.tensix.waitgate_stallwait_blocked_test:"
            "test_a_stallwait_does_not_forget_an_unsatisfied_semwait"
        ),
    ),
    Behaviour(
        name="noc-transfer-alignment",
        guarantee=(
            "A NoC transfer whose source and destination addresses are not "
            "congruent in the low bits the path requires raises "
            "tt_sim.network.alignment.NoCAlignmentError naming the path and "
            "both addresses. Hardware neither faults nor completes such a "
            "transfer correctly — it skews or drops the payload — so before "
            "this a program with the defect produced plausible wrong numbers. "
            "Set TT_SIM_DISABLE_ALIGNMENT_CHECKS=1 to switch it off, which "
            "restores the old silence."
        ),
        since="2026-08-02",
        pinned_by=(
            "tt_sim.network.alignment_test:"
            "test_misaligned_transfer_raises_actionable_message"
        ),
    ),
)

#: name -> :class:`Behaviour`, for a consumer that wants to print or record the
#: whole set. Read-only: a suite must not be able to talk itself into a
#: guarantee by mutating the registry.
BEHAVIOURS = MappingProxyType({b.name: b for b in _BEHAVIOURS})


#: Exception classes in ``tt_sim/`` that are deliberately *not* published as
#: behaviours, and why. The backward guard in ``behaviour_test.py`` requires
#: every exception class in the tree to appear here or in the registry above,
#: so this is the file where "we thought about it and the answer is no" is
#: written down rather than left implicit.
_NOT_A_GUARANTEE = MappingProxyType(
    {
        "UnsupportedBehaviour": (
            "this registry's own failure mode, not a claim about the simulator"
        ),
        "NoC1ShadowError": (
            "warn-by-default and opt-in via TT_SIM_NOC1_SHADOW, so a build "
            "cannot promise it fires; it guards a device-construction mistake "
            "rather than a program's behaviour"
        ),
        "NoCCoordinateError": (
            "a request naming a coordinate outside the grid — a malformed "
            "request rather than a modelling gap that used to pass"
        ),
        "NoCResponseError": (
            "internal NoC bookkeeping; reaching it means tt-sim is wrong, not "
            "that the program under test is"
        ),
        "UnitWedgedError": (
            "the deadlock watchdog. Worth publishing once the false-positive "
            "rate is argued (v3 Strand B says so of every guard); until then a "
            "consumer asserting on it would be asserting on a heuristic"
        ),
        "UnmodelledTileRegisterError": (
            "names a register tt-sim does not model. Honest, but the set it "
            "covers moves with every release, so it is not a stable guarantee"
        ),
        "SrcDvalidError": (
            "a Tensix Src-bank handshake violation: an internal invariant of "
            "the coprocessor model, not something a kernel author can act on"
        ),
        "CSRError": "base class for the CSR errors below",
        "NoCSRsError": (
            "a core built without Zicsr meeting a CSR instruction — a "
            "simulator configuration error, not a program defect"
        ),
        "UnknownCSRError": "an encoding no CSR answers for; malformed input",
        "UnmodelledCSRError": (
            "a real CSR tt-sim does not implement. Same objection as "
            "UnmodelledTileRegisterError: the covered set is not stable"
        ),
        "PlanError": (
            "a malformed congestion-sweep plan; an offline analysis tool's "
            "input validation, nowhere near the simulated device"
        ),
    }
)


class UnsupportedBehaviour(RuntimeError):
    """This tt-sim does not guarantee something the caller depends on."""


def supports(*names: str) -> bool:
    """``True`` when every named behaviour is guaranteed by this build."""
    return all(name in BEHAVIOURS for name in names)


def require(*names: str) -> None:
    """Assert every named behaviour, or raise :class:`UnsupportedBehaviour`.

    The failure names what is missing *and* what this build does have, because
    the two most likely causes are an old checkout and a typo, and the reader
    cannot tell which from a bare "not supported".
    """
    missing = [name for name in names if name not in BEHAVIOURS]
    if not missing:
        return
    have = ", ".join(sorted(BEHAVIOURS)) or "(none)"
    raise UnsupportedBehaviour(
        f"this tt-sim does not guarantee {', '.join(missing)}. Results from a "
        f"suite that depends on it are not trustworthy — the failure mode these "
        f"markers exist for is a green run that exercised nothing. This build "
        f"guarantees: {have}. See {DOCS}."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m tt_sim.behaviour",
        description="What this tt-sim guarantees, for an external test suite.",
    )
    parser.add_argument(
        "--require",
        metavar="NAME",
        nargs="+",
        default=(),
        help="exit non-zero unless every NAME is guaranteed by this build",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print each behaviour's guarantee, not just its name",
    )
    args = parser.parse_args(argv)

    if args.require:
        try:
            require(*args.require)
        except UnsupportedBehaviour as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    for name in sorted(BEHAVIOURS):
        if args.verbose:
            behaviour = BEHAVIOURS[name]
            print(f"{name}  (since {behaviour.since})")
            print(f"    {behaviour.guarantee}")
        else:
            print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
