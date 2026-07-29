"""NoC destination-coordinate encoding strategies (per architecture).

A NoC request names its destination tile by an (x, y) coordinate. *Where* that
coordinate lives in the NIU command registers differs by architecture, so the
:class:`~tt_sim.network.tt_noc.NUI.RequestInitiator` reads it through a strategy
the architecture profile selects:

- **Wormhole** packs the coordinate into the MID address register (X at bit 4,
  Y at bit 10); there is no dedicated coordinate register, so it steals the high
  bits of the 36-bit local address word.
- **Blackhole** holds the coordinate in a dedicated register
  (``NOC_TARG_ADDR_HI`` / ``NOC_RET_ADDR_HI``, offsets 0x8 / 0x14) encoded as
  ``(Y << 6) | X``; the MID register then carries address bits only.

Both take a request initiator and read its raw registers, so they stay decoupled
from the NUI internals. See ``docs/plans/blackhole-support.md``.
"""

from tt_sim.util.bits import extract_bits

_COORD_BITS = 6


class WormholeNocCoords:
    """Coordinate packed into the MID address register (X@4, Y@10)."""

    def target_coord(self, ri) -> tuple[int, int]:
        return (
            extract_bits(ri.target_addr_mid, _COORD_BITS, 4),
            extract_bits(ri.target_addr_mid, _COORD_BITS, 10),
        )

    def ret_coord(self, ri) -> tuple[int, int]:
        return (
            extract_bits(ri.ret_addr_mid, _COORD_BITS, 4),
            extract_bits(ri.ret_addr_mid, _COORD_BITS, 10),
        )

    def broadcast_coords(self, ri) -> tuple[int, int, int, int]:
        """``(x_start, y_start, x_end, y_end)`` from the MID register."""
        return (
            extract_bits(ri.ret_addr_mid, _COORD_BITS, 16),
            extract_bits(ri.ret_addr_mid, _COORD_BITS, 22),
            extract_bits(ri.ret_addr_mid, _COORD_BITS, 4),
            extract_bits(ri.ret_addr_mid, _COORD_BITS, 10),
        )


class BlackholeNocCoords:
    """Coordinate in the dedicated HI register, encoded ``(Y << 6) | X``.

    Multicast packs both corners into that register:
    ``x_end[0], y_end[6], x_start[12], y_start[18]``.
    """

    def target_coord(self, ri) -> tuple[int, int]:
        c = ri.target_addr_hi
        return (extract_bits(c, _COORD_BITS, 0), extract_bits(c, _COORD_BITS, 6))

    def ret_coord(self, ri) -> tuple[int, int]:
        c = ri.ret_addr_hi
        return (extract_bits(c, _COORD_BITS, 0), extract_bits(c, _COORD_BITS, 6))

    def broadcast_coords(self, ri) -> tuple[int, int, int, int]:
        """``(x_start, y_start, x_end, y_end)`` from the HI register."""
        c = ri.ret_addr_hi
        return (
            extract_bits(c, _COORD_BITS, 12),
            extract_bits(c, _COORD_BITS, 18),
            extract_bits(c, _COORD_BITS, 0),
            extract_bits(c, _COORD_BITS, 6),
        )
