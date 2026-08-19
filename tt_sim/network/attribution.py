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
* **The circular buffer the L1 end of the transfer lands in**, and with it the
  **page size in bytes** — the number that says whether a shard was split below
  tile granularity. This is *not* on the wire: the simulator is told addresses
  and payloads (``WRITE`` / ``READ`` / ``RESET_*`` / ``START`` / ``EXIT``, see
  :mod:`tt_sim.bridge.protocol`), never buffer layouts, and there is no Tensix
  configuration register that holds it either — circular buffers are a software
  construct. It is recovered instead from the ``cb_interface`` array in the
  issuing core's local memory, which the firmware fills in from L1 before the
  kernel runs and which tt-sim models like any other memory. Located by symbol
  (never by scanning), decoded only when the decode is self-consistent, and
  reported only for a buffer that actually contains the transfer's address —
  so a layout change in a future tt-metal release costs the line, not its
  truthfulness. When none of that lands, the message says so in one line
  rather than leaving the reader to wonder; see :func:`describe_page`.
* **Not the buffer name.** tt-metal's allocator, which is what maps an address
  back to a named buffer, lives on the host and is never described to the
  simulator. The address *range* the transfer covers is reported instead.

Nothing here may raise. It runs while an exception is already being built, and
a failure to describe a fault must never replace it with a different one.
"""

from __future__ import annotations

import struct
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

#: The tt-metal firmware global holding one ``CBInterface`` per circular
#: buffer. Defined (weakly) in both the firmware and the kernel ELFs, always in
#: the RISC's *local* data memory, and populated by
#: ``setup_local_cb_read_write_interfaces`` before the kernel's first
#: instruction — so during any transfer a kernel issues it is live.
_CB_SYMBOL = "cb_interface"

#: ``sizeof(CBInterface)``. Not read from the ELF: the union's other arms are
#: remote-CB shapes of the same width, and what the eight words *mean* is
#: verified against the values themselves (see :func:`_decode_cbs`) rather
#: than trusted.
_CB_ENTRY_BYTES = 32

#: The unit a ``LocalCBInterface``'s addresses are counted in, per issuing
#: core. Data-movement cores keep bytes; the TRISCs keep 16-byte units, because
#: ``circular_buffer_init.h`` shifts every address field right by
#: ``CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT`` (4) when ``COMPILE_FOR_TRISC``.
_CB_ADDR_UNIT = {"TRISC0": 16, "TRISC1": 16, "TRISC2": 16}

#: Upper bound on a plausible L1 byte offset, used only to reject a decode that
#: is obviously not a circular buffer. Wormhole and Blackhole L1 are both
#: 1464 KiB; the bound is deliberately loose because the real filter is
#: "does this buffer contain the address the transfer used".
_CB_MAX_L1 = 2 << 20

#: What the page-size line says when it has nothing. Fixed wording so a
#: consumer can grep for the absence as easily as for the presence.
_CB_ABSENT = "page size not visible to the simulator"


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
    #: ``(role, path)`` for every *trusted* ELF found for the issuing unit,
    #: whether or not it covered the PC. Carried so :func:`describe_page` can
    #: look up ``cb_interface`` without paying for a second discovery, and
    #: because the symbol lives in the **firmware** ELF — the one
    #: :func:`describe_source` deliberately looks at last.
    elfs: tuple[tuple[str, str], ...] = ()


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
    trusted: list[tuple[str, str]] = []
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
            trusted.append((str(entry.role), str(entry.path)))
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
                elfs=tuple(trusted),
            )
    except Exception as exc:
        return Source("", [], "", "", "", note=f"ELF lookup failed: {exc}")
    if tried:
        return Source(
            "",
            [],
            "",
            "",
            "",
            note="no ELF covers this PC: " + "; ".join(tried),
            elfs=tuple(trusted),
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
        elfs=tuple(trusted),
    )


class CircularBuffer(NamedTuple):
    """One decoded ``LocalCBInterface``, converted to bytes."""

    #: Index into ``cb_interface`` — the ``CBIndex::c_N`` the kernel names.
    index: int
    #: First and last byte of the buffer's L1 extent, inclusive.
    start: int
    end: int
    #: Bytes per page — the number that says whether a buffer was split below
    #: tile granularity, which the two hex addresses alone never showed.
    page_size: int
    #: ``fifo_size / fifo_page_size``. Derived rather than read from
    #: ``fifo_num_pages``, which the firmware only fills in on the writing
    #: side — a read-only interface leaves it zero.
    pages: int


def _cb_symbol(path) -> tuple[int, int] | None:
    """``(address, size)`` of ``cb_interface`` in an ELF, or ``None``.

    The symbol table, not DWARF: the array's *address* is all that is needed,
    and it is present even in a build with no debug info.
    """
    from elftools.elf.elffile import ELFFile

    with open(path, "rb") as handle:
        elf = ELFFile(handle)
        for section in (".symtab", ".dynsym"):
            table = elf.get_section_by_name(section)
            if table is None:
                continue
            for symbol in table.iter_symbols():
                if symbol.name == _CB_SYMBOL and symbol["st_size"]:
                    return int(symbol["st_value"]), int(symbol["st_size"])
    return None


def _decode_cbs(blob: bytes, unit: int) -> list[CircularBuffer]:
    """Decode ``cb_interface`` — every entry that proves it is one.

    ``LocalCBInterface`` is ``fifo_size, fifo_limit, fifo_page_size,
    fifo_num_pages, fifo_rd_ptr, fifo_wr_ptr, tiles_acked_received_init,
    fifo_wr_tile_ptr``. Rather than *trusting* that order — it belongs to
    tt-metal, which this repo deliberately does not pin a version of — each
    entry has to satisfy the arithmetic the fields hold to each other:
    ``fifo_limit == fifo_start + fifo_size``, ``fifo_size`` a whole number of
    pages, and both pointers either unset or inside ``[start, limit]``. A
    release that reorders the struct fails those and is silently skipped,
    which is the desired failure: no line beats a wrong line.
    """
    out: list[CircularBuffer] = []
    for offset in range(0, len(blob) - _CB_ENTRY_BYTES + 1, _CB_ENTRY_BYTES):
        size, limit, page, _num_pages, rd, wr = struct.unpack_from("<6I", blob, offset)
        if not size or not page or size % page or limit <= size:
            continue
        start = limit - size
        if not (rd or wr):
            continue
        if any(ptr and not start <= ptr <= limit for ptr in (rd, wr)):
            continue
        low, high = start * unit, limit * unit
        if low <= 0 or high > _CB_MAX_L1:
            continue
        out.append(
            CircularBuffer(
                index=offset // _CB_ENTRY_BYTES,
                start=low,
                end=high - 1,
                page_size=page * unit,
                pages=size // page,
            )
        )
    return out


def describe_page(issuer: Issuer, source: Source, addrs) -> str:
    """The page-size line: which circular buffer the transfer landed in.

    ``addrs`` is ``(label, address)`` pairs, tried in order; the first that
    falls inside a configured circular buffer wins. Only one of the two ends of
    a transfer is in L1, and which one depends on the direction, so both are
    offered and the label in the message records which one matched.

    Always returns a line. When the buffer cannot be found the line says so and
    why, because "we could not tell you" is itself the answer to a reader who
    would otherwise keep looking for a number that is not there.

    Never raises: it runs while an exception is being built.
    """
    try:
        return _describe_page(issuer, source, addrs)
    except Exception as exc:  # pragma: no cover - defensive
        return f"CB page: lookup failed ({exc}) — {_CB_ABSENT}."


def _describe_page(issuer: Issuer, source: Source, addrs) -> str:
    # Firmware first: ``cb_interface`` is a firmware-owned global, and a
    # kernel ELF that failed to prove residency may carry a stale address for
    # it. describe_source orders the other way round for exactly the opposite
    # reason (the PC is usually the kernel's), so the two do not share a list.
    paths = [
        path for _role, path in sorted(source.elfs, key=lambda rp: rp[0] != "firmware")
    ]
    if not paths:
        return (
            f"CB page: no ELF proved resident for {issuer.core}, so "
            f"{_CB_SYMBOL} could not be located — {_CB_ABSENT}."
        )
    for path in paths:
        try:
            symbol = _cb_symbol(path)
        except Exception:
            continue
        if symbol is None:
            continue
        addr, size = symbol
        try:
            blob = bytes(issuer.memory.read(addr, size))
        except Exception as exc:
            return (
                f"CB page: {_CB_SYMBOL} at {hex(addr)} is unreadable "
                f"({exc}) — {_CB_ABSENT}."
            )
        unit = _CB_ADDR_UNIT.get(issuer.core, 1)
        buffers = _decode_cbs(blob, unit)
        for label, address in addrs:
            for cb in buffers:
                if cb.start <= address <= cb.end:
                    return (
                        f"CB page: {label} {hex(address)} is in CB {cb.index} — "
                        f"{cb.pages} pages of {cb.page_size} bytes over "
                        f"{hex(cb.start)}..{hex(cb.end)} (read from "
                        f"{_CB_SYMBOL}[{cb.index}] in {issuer.core}'s local "
                        f"memory)."
                    )
        if buffers:
            return (
                f"CB page: none of the {len(buffers)} circular buffers "
                f"configured on {issuer.core} covers either address — "
                f"{_CB_ABSENT}."
            )
        return (
            f"CB page: {issuer.core} has no configured circular buffers — {_CB_ABSENT}."
        )
    return (
        f"CB page: no {_CB_SYMBOL} symbol in {issuer.core}'s resident ELFs "
        f"— {_CB_ABSENT}."
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
        # Beside the ranges, not after the attribution: the transfer size and
        # the page size are read together ("400 of 1600" is the diagnosis).
        # The issuer block below keeps its own order, so the call-site chain
        # is still the line directly under the PC it belongs to.
        lines.append(
            describe_page(
                issuer,
                source,
                (("destination", dst_addr), ("source", src_addr)),
            )
        )
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
