"""Translated NoC coord -> tt-sim unified coord mapping.

Hand-rolled table covering the coordinates the supported tt-metal examples
hit end-to-end:

    translated (1, 1)  -> unified (18, 18)   # the single Tensix

DRAM: Wormhole B0 has 6 controllers, each reachable via 2 NoC sub-endpoints
serving the same physical memory (the two halves are distinguished by the
1 GB ``address_offset`` baked into the buffer address by the tt-metal
allocator — see ``wormhole_b0_80_arch.yaml`` ``dram_views:``). Both
sub-endpoints alias to the same simulator ``DRAMTile`` because that tile's
two banks (at ``0x0`` and ``0x4000_0000``) already model the same controller's
two halves.

For all other translated coords the fabric falls back to ``NullCore`` (which
swallows writes and returns zeros) — proven sufficient to keep tt-metal's
device-init loop happy.

TODO(future): derive the full mapping from ``soc_descriptor.yaml``. The
``functional_workers`` / ``dram`` / ``eth`` lists in the YAML are in
non-translated NoC coords; the translation rules (TENSIX/DRAM routing
through NoC translation tables, ranges 16-25 for unified coords) need
spelling out in code before this can land. Multi-Tensix support also needs
``Wormhole.__init__`` in ``tt_sim/device/tt_device.py`` extended — currently
it still hard-codes a single ``TensixTile(18,18)``.
"""

TENSIX_COORD_MAP = {(1, 1): (18, 18)}

# 12 DRAM banks (2 sub-endpoints per controller) -> 6 simulator DRAMTiles.
# Pairs sharing a unified coord are the two NoC paths into the same physical
# controller; the bank-offset that tt-metal bakes into the buffer address
# selects ``ddr_bank_0`` vs ``ddr_bank_1`` inside that DRAMTile.
DRAM_COORD_MAP = {
    (0, 11): (16, 16),  # channel 0, address_offset 0
    (0, 1): (16, 16),  # channel 0, address_offset 1 GB
    (0, 5): (17, 16),  # channel 1, address_offset 0
    (0, 7): (17, 16),  # channel 1, address_offset 1 GB
    (5, 1): (16, 17),  # channel 2, address_offset 0
    (5, 11): (16, 17),  # channel 2, address_offset 1 GB
    (5, 2): (17, 17),  # channel 3, address_offset 0
    (5, 9): (17, 17),  # channel 3, address_offset 1 GB
    (5, 8): (16, 18),  # channel 4, address_offset 0
    (5, 3): (16, 18),  # channel 4, address_offset 1 GB
    (5, 5): (17, 18),  # channel 5, address_offset 0
    (5, 7): (17, 18),  # channel 5, address_offset 1 GB
}

# Wormhole NoC grid is 10 cols x 12 rows. NoC 1's origin is the bottom-right
# tile (data flows leftwards/upwards), so a NoC 0 logical coord (x, y) is the
# same physical tile as NoC 1 logical (NOC_GRID_X - 1 - x, NOC_GRID_Y - 1 - y).
# See tt-isa-documentation/WormholeB0/NoC/Coordinates.md.
NOC_GRID_X = 10
NOC_GRID_Y = 12


def noc1_mirror(noc0_coord):
    x, y = noc0_coord
    return (NOC_GRID_X - 1 - x, NOC_GRID_Y - 1 - y)
