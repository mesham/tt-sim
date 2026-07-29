"""The Wormhole B0 architecture profile.

Values here are the single source of truth for the Wormhole constants the
device/tile classes used to hardcode. See ``tt_sim/device/tt_device.py`` for how
they are consumed and ``docs/plans/blackhole-support.md`` for the porting plan.
"""

from tt_sim.arch.profile import ArchProfile
from tt_sim.network.noc_coords import WormholeNocCoords

WORMHOLE_PROFILE = ArchProfile(
    name="wormhole",
    # NoC grid: 10 columns x 12 rows (NIU X in 0-9, Y in 0-11), per
    # WormholeB0/NoC/MemoryMap.md.
    noc_grid_x=10,
    noc_grid_y=12,
    # Max single NoC transfer: 8192 bytes, per WormholeB0/NoC/MemoryMap.md.
    noc_max_burst_size=8192,
    # Tensix L1 SRAM: TENSIX_SRAM_SIZE = 1464 * 1024, per
    # WormholeB0/TensixTile/L1.md.
    tensix_l1_size=1464 * 1024,
    # Baby-core local data memories (ttsim config.h, TT_ARCH_VERSION 0).
    brisc_local_mem_size=4 * 1024,
    ncrisc_local_mem_size=4 * 1024,
    trisc_local_mem_size=2 * 1024,
    # Wormhole keeps the reset-PC override in the Tensix backend config.
    baby_core_reset_pc_debug_regs=False,
    # Unified coords assigned to each of the 6 physical DRAM channels. Picked
    # from the (16-17, 16-18) band so they stay clear of the Tensix at (18, 18).
    dram_channel_unified_coords=(
        (16, 16),  # channel 0
        (17, 16),  # channel 1
        (16, 17),  # channel 2
        (17, 17),  # channel 3
        (16, 18),  # channel 4
        (17, 18),  # channel 5
    ),
    # The two worker-side SoC-physical NoC 0 coords per channel (from
    # ``dram_views`` in soc_descriptor.yaml). Element 0 is the primary; the rest
    # are aliases. Both sub-endpoints route to the same DRAM tile.
    dram_channel_physical_noc0_coords=(
        ((0, 11), (0, 1)),  # channel 0
        ((0, 5), (0, 7)),  # channel 1
        ((5, 1), (5, 11)),  # channel 2
        ((5, 2), (5, 9)),  # channel 3
        ((5, 8), (5, 3)),  # channel 4
        ((5, 5), (5, 7)),  # channel 5
    ),
    # Default: a single Tensix at the historical (18, 18) coord every single-tile
    # example bakes in.
    tensix_unified_coords=((18, 18),),
    # Wormhole packs the destination coord into the MID address register.
    noc_coord_strategy=WormholeNocCoords(),
    noc_blackhole_cmd_buf_layout=False,
)
