"""NoC address-alignment checking.

Real silicon places alignment requirements on the source and destination
addresses of a NoC transfer. They are *congruence* requirements rather than
absolute-alignment requirements: what matters is that the two addresses agree in
their low bits, because the NIU moves data through a fixed-width datapath and
only rotates the payload by the shared low-address bits. Violating them is
``UndefinedBehavior`` — the transfer is skewed or silently dropped, it does not
fault — so a simulator that ignores them quietly produces wrong answers.

Sources
-------
* ``WormholeB0/NoC/Alignment.md`` in https://github.com/tenstorrent/tt-isa-documentation
  — "violations of the alignment requirements result in UndefinedBehavior". Its
  length-mode table gives, for the source/destination pair, only *congruence*
  codes: ``C4`` / ``C16`` / ``C32``, defined there as
  ``(src % n) == (dst % n)``. Notably it contains **no** absolute source- or
  destination-alignment codes for the L1/DRAM cases, so an absolute-alignment
  check would be stricter than hardware.
* The vendor reference simulator ``ttsim`` (``src/tile.cpp``, ``noc_cmd_start``)
  enforces exactly these, and — importantly — distinguishes ``UndefinedBehavior``
  (hardware really rejects it) from ``UnimplementedFunctionality`` (that
  simulator merely does not handle it). Only the former are modelled here.

Rules modelled
--------------
======================  =========================================  ==============
Path                    Rule                                       ttsim citation
======================  =========================================  ==============
write (L1 -> anywhere)  ``C16``                                    tile.cpp:1555
read (DRAM -> L1)       ``C32`` Wormhole / ``C64`` Blackhole       tile.cpp:1573/1576
read (L1 -> L1)         ``C16``                                    tile.cpp:1583
read (MMIO -> L1)       ``C4``                                     tile.cpp:1581
======================  =========================================  ==============

Deliberately *not* modelled — see the module docstring of the test file for the
full reasoning — are the byte-enable-mode absolute alignments (byte-enable writes
are unimplemented in both tt-sim and ttsim), the atomic and inline-write address
rules (ttsim flags those ``UnimplementedFunctionality``, i.e. simulator limits
rather than hardware limits), and the maximum-packet-size bound (tt-sim
deliberately *splits* oversized bursts at the NIU instead of rejecting them).

Checking is on by default and is disabled by setting
``TT_SIM_DISABLE_ALIGNMENT_CHECKS`` to a truthy value (``1``/``true``/``yes``/
``on``, case-insensitive), following the ``TT_SIM_*`` convention used by the
``TT_SIM_DIAG_*`` diagnostics flags.
"""

from __future__ import annotations

import os

#: Environment variable that turns alignment checking off.
DISABLE_ENV_VAR = "TT_SIM_DISABLE_ALIGNMENT_CHECKS"

#: Addresses at or above this are MMIO rather than L1, per the "Address Type"
#: section of ``WormholeB0/NoC/Alignment.md``.
MMIO_BASE = 0xFF000000

#: Congruence modulus for a transfer whose remote end is L1 (Tensix or Ethernet
#: tile SRAM): ``C16``.
L1_CONGRUENCE = 16

#: Congruence modulus for a transfer whose remote end is MMIO: ``C4``.
MMIO_CONGRUENCE = 4

_TRUTHY = {"1", "true", "yes", "on"}


class NoCAlignmentError(RuntimeError):
    """Raised when a NoC transfer violates a hardware alignment requirement."""


def _env_disabled() -> bool:
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY


#: Cached at import so the common path does not hit ``os.environ`` per request.
#: Call :func:`refresh_from_env` after mutating the environment.
_disabled = _env_disabled()


def refresh_from_env() -> bool:
    """Re-read :data:`DISABLE_ENV_VAR`. Returns whether checking is now enabled."""
    global _disabled
    _disabled = _env_disabled()
    return not _disabled


def checking_enabled() -> bool:
    """Whether alignment checking is currently active."""
    return not _disabled


def set_checking_enabled(enabled: bool) -> None:
    """Force checking on or off, ignoring the environment (for tests)."""
    global _disabled
    _disabled = not enabled


def congruence_for_read(src_addr: int, src_is_dram: bool, dram_congruence: int) -> int:
    """The modulus a read must satisfy, given where it is reading *from*.

    ``dram_congruence`` is the architecture's
    :attr:`~tt_sim.arch.profile.ArchProfile.noc_dram_read_congruence` (32 on
    Wormhole, 64 on Blackhole).
    """
    if src_is_dram:
        return dram_congruence
    if src_addr >= MMIO_BASE:
        return MMIO_CONGRUENCE
    return L1_CONGRUENCE


def check_congruence(
    modulus: int,
    src_addr: int,
    dst_addr: int,
    *,
    path: str,
    noc_number: int | None = None,
    initiator: object = None,
) -> None:
    """Raise :class:`NoCAlignmentError` unless ``src_addr`` and ``dst_addr`` are
    congruent modulo ``modulus``.

    ``path`` names the access path (e.g. ``"DRAM -> L1 read"``) and appears in
    the message, along with both addresses and the required alignment, so the
    failure is actionable without re-deriving it from the docs.
    """
    if _disabled:
        return
    src_rem = src_addr % modulus
    dst_rem = dst_addr % modulus
    if src_rem == dst_rem:
        return
    where = "NoC" if noc_number is None else f"NoC{noc_number}"
    if initiator is not None:
        where = f"{where} {initiator}"
    raise NoCAlignmentError(
        f"{where} {path}: misaligned transfer. Source address {hex(src_addr)} "
        f"and destination address {hex(dst_addr)} must be congruent modulo "
        f"{modulus} (required alignment: matching low {modulus.bit_length() - 1} "
        f"address bits), but {hex(src_addr)} % {modulus} = {src_rem} and "
        f"{hex(dst_addr)} % {modulus} = {dst_rem}. Hardware would silently "
        f"corrupt or drop this transfer (see WormholeB0/NoC/Alignment.md). "
        f"Set {DISABLE_ENV_VAR}=1 to disable alignment checking."
    )
