"""Translated NoC coord -> tt-sim unified coord mapping.

This ships a hard-coded table covering the two coordinates the tt-metal
``one`` example exercises end-to-end:

    translated (1, 1)  -> unified (18, 18)   # the single Tensix
    translated (0, 11) -> unified (16, 16)   # the single DRAM channel

For all other translated coords the fabric falls back to ``NullCore`` (which
swallows writes and returns zeros) — proven sufficient to keep tt-metal's
device-init loop happy.

TODO(future): derive the full mapping from ``soc_descriptor.yaml`` so a
second Tensix or extra DRAM channels don't require code edits.

* ``functional_workers``, ``dram``, ``eth`` lists in the YAML are in
  non-translated NoC coords; the translation offset rule (TENSIX/DRAM
  routing through NoC translation tables, ranges 16-25 for unified
  coords) needs spelling out in code before this can land.
* Multi-Tensix support also needs ``Wormhole.__init__`` in
  ``tt_sim/device/tt_device.py`` extended — currently it hard-codes one
  ``TensixTile(18,18)`` and one ``DRAMTile(16,16)``.
"""

TENSIX_COORD_MAP = {(1, 1): (18, 18)}
DRAM_COORD_MAP = {(0, 11): (16, 16)}

# Wormhole NoC grid is 10 cols x 12 rows. NoC 1's origin is the bottom-right
# tile (data flows leftwards/upwards), so a NoC 0 logical coord (x, y) is the
# same physical tile as NoC 1 logical (NOC_GRID_X - 1 - x, NOC_GRID_Y - 1 - y).
# See tt-isa-documentation/WormholeB0/NoC/Coordinates.md.
NOC_GRID_X = 10
NOC_GRID_Y = 12


def noc1_mirror(noc0_coord):
    x, y = noc0_coord
    return (NOC_GRID_X - 1 - x, NOC_GRID_Y - 1 - y)
