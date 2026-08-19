"""Kernel-ELF auto-discovery, so source-level attribution needs no
hand-holding.

The simulator never sees an ELF. tt-metal JIT-compiles a kernel on the
host and streams the *bytes* over the wire as ordinary WRITE messages
(`tt_sim/bridge/`), so nothing in a run says where the ELF that produced
them lives. This module recovers that link from the host's build cache.

**Layout, as of tt-metal 0.74** (read off a real checkout, not assumed)::

    <cache>/<build-hash>/firmware/<risc>/<risc>.elf
    <cache>/<build-hash>/kernels/<kernel-name>/<config-hash>/<risc>/<risc>.elf
                                                                  /<risc>.elf.xip.elf
    <TT_METAL_RUNTIME_ROOT>/tt_metal/pre-compiled/<hash>/<risc>/<risc>.elf

``<risc>`` is one of ``brisc ncrisc trisc0 trisc1 trisc2``, which is
also how an ELF maps to a simulated unit. ``<cache>`` defaults to
``~/.cache/tt-metal-cache`` and is overridden by ``TT_METAL_CACHE``.
Both **firmware and kernel** are wanted: a data-movement core spends most
of a run inside firmware's launch and barrier loops, so a report built
from kernel ELFs alone leaves its largest rows unnamed.

**Kernels are relocated; firmware is not.** Firmware links to a fixed L1
address and is resident there. A kernel ELF links at its own base (e.g.
``0x49a0`` for BRISC) but tt-metal places it at a runtime base chosen per
core — on a live Wormhole ``six`` the BRISC kernel executes around
``0x8b00``. Comparing the ELF's segment against its own ``p_vaddr``
therefore *fails for every kernel*, which is why discovery also searches
resident memory for the segment's bytes and, on a hit, records the load
bias. Without that step the kernel half of a run is unattributable.

**Selection**, in decreasing order of how much it proves:

1. *Bytes at the link address* — the ELF is resident where it says.
2. *Bytes at some other address* — the ELF is resident, relocated by a
   recovered bias. Still proof of identity.
3. *Recency* — the newest candidate in the cache, unconfirmed. A live
   run rebuilds (or at least re-stamps) its kernels as it starts, so this
   is a good guess and is labelled as one.
4. *Rejected* — memory was readable and no candidate is what ran. The
   unit is left unattributed rather than labelled from the wrong kernel.

Every :class:`Discovered` records which applied, and the ranked report
prints it, because "we guessed by mtime" and "we found these exact bytes
in L1" are very different claims about the same table.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

#: RISC name in the cache path -> simulated unit name in ``Unit``.
RISC_TO_UNIT = {
    "brisc": "BRISC",
    "ncrisc": "NCRISC",
    "trisc0": "TRISC0",
    "trisc1": "TRISC1",
    "trisc2": "TRISC2",
}

#: Candidates byte-verified per (unit, role) before falling back to recency.
#: A build cache accumulates one firmware set per tt-metal build ever used on
#: the machine, and the one that ran is not necessarily the newest — on this
#: box the matching firmware ranked 17th of 19 by mtime. Verification is a
#: couple of reads per candidate, so the limit exists only to bound a
#: pathological cache.
VERIFY_LIMIT = 64
#: Candidates kept per (unit, role) before verification, newest first.
CANDIDATE_LIMIT = 128
#: A segment shorter than this is too weak a fingerprint to relocate on —
#: short runs of common bytes appear all over L1.
MIN_SEARCH_BYTES = 64

DEFAULT_CACHE = "~/.cache/tt-metal-cache"

#: ``how`` value for a unit whose resident code matched no candidate. The
#: chosen path is still recorded so a reader can see what was rejected, but
#: the profile must not load it.
REJECTED = "no match (resident code differs from every candidate)"


@dataclass
class Discovered:
    """One ELF chosen for one unit, plus how it was chosen."""

    unit: str
    path: Path
    how: str  # "verified" | "relocated" | "recent" | "explicit" | REJECTED
    candidates: int = 0
    role: str = "kernel"  # "kernel" | "firmware"
    bias: int = 0  # runtime address - ELF vaddr

    @property
    def usable(self) -> bool:
        return self.how != REJECTED


@dataclass
class Discovery:
    """The whole result, including why it found nothing."""

    elfs: list[Discovered] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    note: str = ""

    def as_pairs(self) -> list[tuple[str, str]]:
        return [(d.unit, str(d.path)) for d in self.elfs]


def cache_roots(env=None) -> list[Path]:
    """Directories that may hold firmware or kernel ELFs, most specific
    first."""
    env = os.environ if env is None else env
    roots: list[Path] = []
    explicit = env.get("TT_METAL_CACHE")
    if explicit:
        roots.append(Path(explicit).expanduser())
    roots.append(Path(DEFAULT_CACHE).expanduser())
    runtime = env.get("TT_METAL_RUNTIME_ROOT") or env.get("TT_METAL_HOME")
    if runtime:
        roots.append(Path(runtime).expanduser() / "tt_metal" / "pre-compiled")
    return [r for r in roots if r.is_dir()]


#: Filenames worth looking at, by the RISC that runs them.
_WANTED = {
    f"{risc}.elf{suffix}": risc for risc in RISC_TO_UNIT for suffix in ("", ".xip.elf")
}


def _candidates(roots: list[Path]) -> dict[tuple[str, str], list[Path]]:
    """All ``<risc>/<risc>.elf[.xip.elf]`` under ``roots``, by (unit, role).

    One ``os.walk`` per root rather than a recursive glob per (risc, role,
    pattern). The caches involved hold thousands of directories and this runs
    at process exit, where it is pure added latency on the user's run —
    fifteen ``**`` globs over the same tree is fifteen times the walk.
    """
    found: dict[tuple[str, str], list[Path]] = {}
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            parent = os.path.basename(dirpath)
            if parent not in RISC_TO_UNIT:
                continue
            # .../firmware/<risc>/ vs .../kernels/<name>/<cfg>/<risc>/. The
            # pre-compiled tree has neither component and is kernel-shaped.
            role = (
                "firmware"
                if os.path.basename(os.path.dirname(dirpath)) == "firmware"
                else "kernel"
            )
            for name in filenames:
                risc = _WANTED.get(name)
                if risc is None or risc != parent:
                    continue
                key = (RISC_TO_UNIT[risc], role)
                found.setdefault(key, []).append(Path(dirpath) / name)
    return found


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _rank(paths: list[Path], since: float | None) -> list[Path]:
    """Newest first; drop pre-session ELFs when that leaves anything, and
    prefer the plain ``.elf`` over its XIP twin at equal recency."""
    unique = sorted(set(paths))
    ranked = sorted(unique, key=lambda p: (-_mtime(p), p.name.endswith(".xip.elf")))
    if since is not None:
        fresh = [p for p in ranked if _mtime(p) >= since]
        if fresh:
            return fresh[:CANDIDATE_LIMIT]
    return ranked[:CANDIDATE_LIMIT]


def segments_of(path: Path) -> list[tuple[int, bytes]]:
    """``(vaddr, bytes)`` for every loadable segment with file content."""
    from elftools.elf.elffile import ELFFile

    with path.open("rb") as handle:
        elf = ELFFile(handle)
        out = []
        for seg in elf.iter_segments():
            if seg["p_type"] != "PT_LOAD" or seg["p_filesz"] == 0:
                continue
            out.append((seg["p_vaddr"], seg.data()))
        return out


def discover(
    env=None,
    since: float | None = None,
    verifier=None,
    searcher=None,
    roots: list[Path] | None = None,
    units: set[str] | None = None,
) -> Discovery:
    """Pick the firmware and kernel ELF for each baby core.

    ``verifier(unit, vaddr, size) -> bytes | None`` reads simulated memory
    at an address; ``None`` means "cannot read there", which abstains
    rather than failing the candidate. ``searcher(blob) -> int | None``
    finds a byte string anywhere in resident memory and is what recovers a
    relocated kernel's load bias. Both are optional.

    ``units`` restricts the search to those unit names. Identification parses
    up to ``VERIFY_LIMIT`` candidate ELFs *per unit and role*, so a caller
    that wants one core's ELF (``tt_sim.network.attribution``, naming the
    kernel behind a rejected transfer) pays a tenth of a whole-device scan.
    """
    env = os.environ if env is None else env

    explicit = env.get("TT_SIM_PROFILE_ELFS", "").strip()
    if explicit:
        return _from_explicit(explicit, units)

    search = roots if roots is not None else cache_roots(env)
    if not search:
        return Discovery(
            note=(
                "no tt-metal kernel cache found — set TT_METAL_CACHE or "
                "TT_SIM_PROFILE_ELFS=<unit>:<path>,... to attribute PCs"
            )
        )

    by_key = _candidates(search)
    result = Discovery(roots=[str(r) for r in search])
    for (unit, role), paths in sorted(by_key.items()):
        if units is not None and unit not in units:
            continue
        ranked = _rank(paths, since)
        if not ranked:
            continue
        chosen = None
        rejected = False
        if verifier is not None or searcher is not None:
            for path in ranked[:VERIFY_LIMIT]:
                # Only kernels move. Letting firmware relocate would let a
                # 64-byte prologue match some unrelated blob in L1.
                how, bias = _identify(
                    path, unit, verifier, searcher if role == "kernel" else None
                )
                if how is not None:
                    chosen = Discovered(unit, path, how, len(ranked), role, bias)
                    break
                if how is None and bias == -1:
                    rejected = True
        if chosen is None and rejected:
            result.elfs.append(Discovered(unit, ranked[0], REJECTED, len(ranked), role))
            continue
        if chosen is None:
            chosen = Discovered(unit, ranked[0], "recent", len(ranked), role)
        result.elfs.append(chosen)
    if not result.elfs:
        result.note = f"no kernel ELFs under {', '.join(result.roots)}"
    return result


def _from_explicit(spec: str, units: set[str] | None = None) -> Discovery:
    """``TT_SIM_PROFILE_ELFS=BRISC:/a.elf,TRISC2:/b.elf`` — or a bare
    comma-separated path list, in which case the unit comes from the
    filename. A trailing ``@0x1234`` sets an explicit load bias."""
    result = Discovery(note="TT_SIM_PROFILE_ELFS")
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        bias = 0
        if "@" in item:
            item, _, bias_str = item.rpartition("@")
            try:
                bias = int(bias_str, 0)
            except ValueError:
                bias = 0
        unit, _, path_str = item.rpartition(":")
        if not unit or not Path(path_str).exists():
            path_str = item
            unit = RISC_TO_UNIT.get(Path(item).name.split(".")[0], "")
        if not unit:
            continue
        if units is not None and unit not in units:
            continue
        result.elfs.append(Discovered(unit, Path(path_str), "explicit", bias=bias))
    return result


def _identify(path: Path, unit: str, verifier, searcher) -> tuple[str | None, int]:
    """``(how, bias)``. ``how`` is ``None`` when this is not the ELF;
    ``bias == -1`` alongside it means *definitely* not (memory was
    readable and differed), as opposed to no opinion."""
    try:
        segs = segments_of(path)
    except Exception:
        return None, 0
    if not segs:
        return None, 0

    decisive = False
    if verifier is not None:
        checked = 0
        mismatched = False
        for vaddr, data in segs:
            got = verifier(unit, vaddr, len(data))
            if got is None:
                continue
            checked += 1
            if bytes(got) != data:
                mismatched = True
                break
        if checked and not mismatched:
            return "verified", 0
        if mismatched:
            decisive = True

    # Search only after the link address was read and *contradicted*. An
    # abstention means the ELF's own vaddr could not be checked — for NCRISC
    # that is its private IRAM — and searching L1 anyway finds the staging
    # copy the host DMAs from, yielding a bias for an address the core never
    # executes. Recency with a zero bias is the right answer there.
    if searcher is not None and decisive:
        # Relocated kernel: find the largest loadable segment anywhere in
        # resident memory and take the difference as the load bias.
        vaddr, data = max(segs, key=lambda s: len(s[1]))
        if len(data) >= MIN_SEARCH_BYTES:
            at = searcher(data)
            if at is not None:
                bias = at - vaddr
                sign = "+" if bias >= 0 else "-"
                return (
                    "verified" if bias == 0 else f"relocated {sign}0x{abs(bias):x}"
                ), bias

    return None, (-1 if decisive else 0)


def session_start() -> float:
    """A conservative "before this run began" timestamp for ``since``.

    One second of slack absorbs filesystem timestamp granularity and the
    gap between process start and the profiler being wired up.
    """
    return time.time() - 1.0
