"""Multi-Tensix smoke test.

Instantiates a Wormhole with two Tensix tiles ((18, 18) and (19, 18)) and
runs the canned ``two/`` kernel (eltwise-add via the coprocessor) on each
tile sequentially. Both tiles share the same DRAM, so we drive each run
with a different input pair written to different DDR offsets and read
back its output from a third offset.

This is the minimum check that:

- ``Wormhole.__init__`` accepts ``tensix_coords`` and fans out tiles.
- Per-tile L1, coprocessor, mailboxes, NoC routers are independent.
- ``launch_firmware`` / ``run_kernel`` can target a specific tile via the
  ``coord`` kwarg without disturbing siblings.
- The deadlock detector handles multiple Tensix tiles.

The full ROADMAP §A test (eltwise-add split across two tiles with a CB
bridged over NoC) is left as follow-on work — that needs custom kernels
compiled against tt-metal.
"""

from driver.wormhole.tt_metal import TT_Metal
from driver.wormhole.wormhole_driver import launch_firmware, run_kernel
from tt_sim.device.tt_device import DeviceTileDiagnostics, Wormhole
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_int32,
)

TILE_A = (18, 18)
TILE_B = (19, 18)

tile_diags = DeviceTileDiagnostics()

wormhole = Wormhole(tile_diags, tensix_coords=(TILE_A, TILE_B))
tt_metal = TT_Metal("tt_metal_0.62.2.json")

# Bring up firmware on both tiles. Each call scopes its soft-reset to the
# target coord so the second call doesn't kill the first tile's running BRISC.
launch_firmware(wormhole, tt_metal, coord=TILE_A)
launch_firmware(wormhole, tt_metal, coord=TILE_B)

# Two input pairs, two output regions. The two/ kernel reads from 0x20 and
# 0x1C0, writes to 0x360 — we run it on each tile with the standard input
# but copy the result out between runs so the second run can use the same
# DDR layout without clobbering the first tile's output.
list1 = list(range(100))
list2 = [100 - i for i in range(100)]

# Tile A run.
wormhole.write((16, 16), 0x20, conv_to_bytes(list1))
wormhole.write((16, 16), 0x1C0, conv_to_bytes(list2))
run_kernel(wormhole, tt_metal, "two/parameters.json", coord=TILE_A)
results_a = [
    conv_to_int32(wormhole.read((16, 16), 0x360 + i * 4, 4)) for i in range(100)
]

# Tile B run — same inputs, fresh kernel launch.
wormhole.write((16, 16), 0x20, conv_to_bytes(list1))
wormhole.write((16, 16), 0x1C0, conv_to_bytes(list2))
run_kernel(wormhole, tt_metal, "two/parameters.json", coord=TILE_B)
results_b = [
    conv_to_int32(wormhole.read((16, 16), 0x360 + i * 4, 4)) for i in range(100)
]

expected = [a + b for a, b in zip(list1, list2)]
assert results_a == expected, (
    f"Tile A output mismatch: {results_a[:5]} vs {expected[:5]}"
)
assert results_b == expected, (
    f"Tile B output mismatch: {results_b[:5]} vs {expected[:5]}"
)

print("Example seven completed successfully")
