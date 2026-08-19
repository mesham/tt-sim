"""Which transfer failed — attributing a NoC request to the code that issued it.

:class:`~tt_sim.network.alignment.NoCAlignmentError` names two addresses and the
rule they broke. That is enough to know a program has a bug and not enough to
know *where*, which is the difference between "our compiler emits a misaligned
GEMM read somewhere" and "line 88 of the reader kernel does". This module
recovers the rest: the size and transaction id of the transfer, the baby core
that issued it, its PC, and — when the kernel's ELF can be identified — the
function and source line that PC belongs to.

Why a stack walk
----------------
A NoC transfer starts when a core stores to that NIU's ``NOC_CMD_CTRL``
register. ``NUI.write`` calls ``RequestInitiator.initiate`` **synchronously**,
so the whole of ``handle_read_transfer`` / ``handle_write_transfer`` — including
the alignment check — runs inside the issuing core's store instruction, on the
issuing core's Python stack. The issuer is therefore not something that has to
be threaded through :class:`~tt_sim.network.tt_noc.NUI.NoCDataRequest` and
carried to a later cycle: it is already right there, a few frames up.

That matters because alignment checking is on by default, so anything recorded
per transfer is paid for by every transfer in every run — and every one of
those payments is wasted, because the recording is only ever read when a
transfer is *rejected*. Discovering the issuer at raise time instead costs
exactly zero on the path where nothing is wrong. It is the same trade
:class:`~tt_sim.network.tt_noc.NoCCoordinateError` already makes, whose
description is likewise attached in an ``except`` clause rather than passed as
an argument.

What is recovered, and how much it proves
-----------------------------------------
* **The core and PC.** From the innermost RV32 core on the stack (live state:
  ``pc_register`` still holds the executing instruction's PC, ``nextpc`` is a
  separate register written back at the end of the tick), falling back to
  :attr:`~tt_sim.memory.memory.MemorySpace.caller_context` — the
  ``(unit_id, core_label, pc)`` tuple the interpreter already stamps on the
  memory space once per tick for exactly this purpose. Nothing is recovered for
  a transfer the *host* initiated over the wire bridge, which is correct: no
  core issued it.
* **The function and source line**, via
  :mod:`tt_sim.trace.elfdisc` + :class:`~tt_sim.trace.dwarf.DwarfIndex`, the
  same machinery the ranked profile report uses. Only *byte-verified* ELFs are
  used — an ELF discovery falls back to "newest in the build cache" when it
  cannot prove residency, and a confidently wrong function name on a fatal
  error is worse than none. Discovery is scoped to the issuing unit so this
  costs two ELF identifications, not twenty.
* **Not the buffer name.** tt-metal's allocator, which is what maps an address
  back to a named buffer, lives on the host and is never described to the
  simulator. The address *range* the transfer covers is reported instead.

Nothing here may raise. It runs while an exception is already being built, and
a failure to describe a fault must never replace it with a different one.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

#: How far up the stack to look for the issuing core. The interpreter is
#: about six frames above the alignment check; the limit only bounds the walk
#: on a stack where there is no core to find at all.
_MAX_FRAMES = 60

#: ELF selections trustworthy enough to name a function from. ``recent`` (the
#: mtime guess) and ``REJECTED`` are deliberately excluded.
_TRUSTED_HOW = ("verified", "relocated", "explicit")

#: Bytes of L1 searched when recovering a relocated kernel's load bias.
#: Matches ``tt_sim.trace.auto._L1_SEARCH_BYTES``.
_L1_SEARCH_BYTES = 1 << 20


class Issuer(NamedTuple):
    """The core that issued the transfer under inspection."""

    #: ``"BRISC"``, ``"TRISC1"``, ... — the name :mod:`tt_sim.trace.elfdisc`
    #: also uses for a unit, so it indexes an ELF directly.
    core: str
    #: PC of the store to ``NOC_CMD_CTRL`` that started the transfer.
    pc: int
    #: The issuing core's visible memory, used to prove which ELF is resident.
    memory: object


def _is_core(obj) -> bool:
    return (
        hasattr(obj, "core_label")
        and hasattr(obj, "pc_register")
        and hasattr(obj, "visible_memory")
    )


def find_issuer() -> Issuer | None:
    """The core currently executing the store that started this transfer.

    ``None`` when no core is on the stack — a host-initiated write over the
    bridge, or a test driving :class:`RequestInitiator` directly.
    """
    try:
        frame = sys._getframe(1)
    except (ValueError, AttributeError):  # pragma: no cover - no frame support
        return None
    fallback = None
    for _ in range(_MAX_FRAMES):
        if frame is None:
            break
        try:
            obj = frame.f_locals.get("self")
            if _is_core(obj):
                # Live state beats the stashed tuple: being on the stack is
                # itself proof this core is mid-instruction.
                return Issuer(
                    str(obj.core_label),
                    int(obj.pc_register.read_uint()),
                    obj.visible_memory,
                )
            if fallback is None:
                ctx = getattr(obj, "caller_context", None)
                if isinstance(ctx, tuple) and len(ctx) >= 3 and ctx[2] is not None:
                    fallback = Issuer(str(ctx[1]), int(ctx[2]), obj)
        except Exception:
            pass
        frame = frame.f_back
    return fallback


def _elf_probes(unit: str, memory):
    """``(verifier, searcher)`` over one core's view of its tile.

    The verifier abstains (``None``) for every other unit, so a scoped
    discovery never claims to have checked an ELF it could not read.
    """

    snapshot: list[bytes | None] = []

    def verifier(want_unit, vaddr, size):
        if want_unit != unit:
            return None
        try:
            return bytes(memory.read(vaddr, size))
        except Exception:
            return None

    def searcher(blob):
        if not snapshot:
            try:
                snapshot.append(bytes(memory.read(0, _L1_SEARCH_BYTES)))
            except Exception:
                snapshot.append(None)
        l1 = snapshot[0]
        if l1 is None:
            return None
        at = l1.find(blob)
        # Two copies of the same text mean the bias cannot be pinned down.
        if at < 0 or l1.find(blob, at + 1) >= 0:
            return None
        return at

    return verifier, searcher


def _kernel_name(path) -> str:
    """The tt-metal kernel name embedded in a cached kernel ELF's path.

    ``<cache>/<hash>/kernels/<name>/<config-hash>/<risc>/<risc>.elf`` — the
    component after ``kernels``. ``""`` for any other shape.
    """
    parts = list(getattr(path, "parts", ()))
    try:
        return parts[parts.index("kernels") + 1]
    except (ValueError, IndexError):
        return ""


class Source(NamedTuple):
    """What an ELF had to say about the issuing PC."""

    #: tt-metal kernel name from the cache path, or ``""``.
    kernel: str
    #: Named ranges covering the PC, outermost first.
    chain: list[str]
    #: ``"<file>:<line>"``, or ``""`` when the line program does not cover it.
    location: str
    #: The ELF the answer came from, and how it was chosen.
    elf: str
    how: str
    #: Why nothing was resolved, when nothing was — always safe to print.
    note: str = ""


def describe_source(issuer: Issuer) -> Source:
    """Resolve the issuing PC against the ELF that core is running.

    Only *byte-verified* selections are used. ELF discovery falls back to
    "newest in the build cache" when it cannot prove residency, and a
    confidently wrong function name attached to a fatal error is worse than no
    name at all — so an unproved candidate becomes a ``note``, not an answer.

    Never raises: an unreadable build cache, a missing pyelftools, or a
    stripped ELF each cost a line of the message, not the diagnosis.
    """
    tried: list[str] = []
    try:
        from tt_sim.trace.dwarf import DwarfIndex
        from tt_sim.trace.elfdisc import discover

        verifier, searcher = _elf_probes(issuer.core, issuer.memory)
        found = discover(verifier=verifier, searcher=searcher, units={issuer.core})
        # Kernel before firmware: a core sits in firmware's launch loop for
        # most of a run, but a transfer it *issued* is usually the kernel's.
        entries = sorted(
            (e for e in found.elfs if e.unit == issuer.core),
            key=lambda e: e.role != "kernel",
        )
        for entry in entries:
            if not any(entry.how.startswith(h) for h in _TRUSTED_HOW):
                continue
            index = DwarfIndex()
            try:
                if not index.load(entry.path, unit=issuer.core, bias=entry.bias):
                    tried.append(f"{entry.path} ({entry.how}, no DWARF)")
                    continue
            except Exception as exc:
                tried.append(f"{entry.path} (unreadable: {exc})")
                continue
            chain = index.inline_chain(issuer.pc, issuer.core)
            loc = index.nearest(issuer.pc, issuer.core)
            if not chain and loc is None:
                tried.append(f"{entry.path} ({entry.how}, does not cover the PC)")
                continue
            return Source(
                kernel=_kernel_name(entry.path),
                chain=chain,
                location=f"{loc.file}:{loc.line}" if loc is not None else "",
                elf=str(entry.path),
                how=entry.how,
            )
    except Exception as exc:
        return Source("", [], "", "", "", note=f"ELF lookup failed: {exc}")
    if tried:
        return Source(
            "", [], "", "", "", note="no ELF covers this PC: " + "; ".join(tried)
        )
    return Source(
        "",
        [],
        "",
        "",
        "",
        note=(
            f"no ELF proved resident for {issuer.core} — set "
            f"TT_SIM_PROFILE_ELFS={issuer.core}:<path to its .elf> to name the "
            f"function and line"
        ),
    )


def describe_transfer(request) -> str:
    """Size, transaction id and address spans, read off the NIU's registers.

    ``request`` is a :class:`~tt_sim.network.tt_noc.NUI.RequestInitiator`, duck
    typed rather than imported: this module is imported *by* the NoC, from the
    ``except`` clause that catches the alignment error, so importing it back
    would be a cycle. Every field is optional, so a partially-built initiator
    (a unit test, a half-programmed NIU) still gets whatever is there.
    """
    bits: list[str] = []
    try:
        size = int(getattr(request, "at_len_be", 0) or 0)
        if size:
            bits.append(f"{size} bytes")
        packet_tag = int(getattr(request, "packet_tag", 0) or 0)
        bits.append(f"transaction id {(packet_tag >> 10) & 0xF}")
        # NOC_CTRL bits 0-1: 0 = read, 1 = atomic, 2 = write. A read's remote
        # end is the target coord, a write's is the return coord.
        mode = int(getattr(request, "ctrl", 0) or 0) & 0x3
        nui = getattr(request, "nui", None)
        strategy = getattr(nui, "noc_coord_strategy", None)
        if strategy is not None:
            picker = strategy.target_coord if mode == 0 else strategy.ret_coord
            bits.append(f"remote tile {tuple(picker(request))}")
    except Exception:
        pass
    return ", ".join(bits)


def span(addr: int, size: int) -> str:
    """``0x1000..0x103f`` — the range a transfer of ``size`` bytes covers."""
    if size <= 0:
        return hex(addr)
    return f"{hex(addr)}..{hex(addr + size - 1)}"


def provenance(request, src_addr: int, dst_addr: int) -> str:
    """The lines appended to a :class:`NoCAlignmentError` message.

    ``""`` when nothing at all could be established, so a caller can append
    unconditionally. Runs only on the failing path.
    """
    lines: list[str] = []
    try:
        size = int(getattr(request, "at_len_be", 0) or 0)
    except Exception:
        size = 0
    detail = describe_transfer(request) if request is not None else ""
    if detail or size:
        lines.append(
            f"Transfer: {detail}; source {span(src_addr, size)}, "
            f"destination {span(dst_addr, size)}."
        )
    issuer = find_issuer()
    if issuer is not None:
        source = describe_source(issuer)
        issued = f"Issued by: {issuer.core} at PC={hex(issuer.pc)}"
        if source.kernel:
            issued += f', in kernel "{source.kernel}"'
        lines.append(issued + ".")
        if source.chain or source.location:
            site = " > ".join(source.chain)
            if source.location:
                site = f"{site} at {source.location}" if site else source.location
            lines.append(f"Call site: {site}.")
        if source.elf:
            lines.append(f"ELF: {source.elf} ({source.how}).")
        elif source.note:
            lines.append(f"ELF: {source.note}.")
    if not lines:
        return ""
    return "\n" + "\n".join("  " + line for line in lines)


def attach_provenance(exc, request, src_addr: int, dst_addr: int) -> None:
    """Append the provenance lines to an already-raised alignment error.

    The exception is enriched **in place**: same object, same type, same
    traceback, so re-raising it from the ``except`` clause that called this is
    indistinguishable from the original raise apart from the added text. That
    is what keeps "which programs raise" identical before and after — this
    function is only ever reached once something has already decided to raise.

    Never raises. A failure to describe a fault must not replace it.
    """
    try:
        extra = provenance(request, src_addr, dst_addr)
        if not extra:
            return
        args = exc.args or ("",)
        exc.args = (f"{args[0]}{extra}",) + tuple(args[1:])
    except Exception:
        pass
