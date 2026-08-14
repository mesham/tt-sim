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
    # Each DRAM channel is a flat 2 GiB: soc_descriptor.yaml `dram_bank_size` =
    # 2147483648, matching ttsim config.h DRAM_CHANNEL_SIZE (TT_ARCH_VERSION 0).
    # 6 channels = 12 GiB. tt-metal banks each channel as two 1 GiB views
    # (`dram_view_size` = 1073741824 at address_offset 0 and 1 GiB), so the
    # second view is just the top half of this one range.
    dram_channel_size=0x8000_0000,
    # ...of which each *physical* GDDR6 channel serves 1 GiB. WormholeB0/
    # DRAMTile/README.md's "NoC to DRAM tile" map: "GDDR6 Channel 0 data"
    # 0x0_0000_0000-0x0_3FFF_FFFF, "GDDR6 Channel 1 data" 0x0_4000_0000-
    # 0x0_7FFF_FFFF. Two independent controllers behind one tile, which the
    # cost model's endpoint occupancy has to keep apart.
    dram_gddr_channel_size=0x4000_0000,
    # Default: a single Tensix at the historical (18, 18) coord every single-tile
    # example bakes in.
    tensix_unified_coords=((18, 18),),
    # Wormhole packs the destination coord into the MID address register.
    noc_coord_strategy=WormholeNocCoords(),
    noc_blackhole_cmd_buf_layout=False,
    # Tensix NoC 1 mirror aliases are REQUIRED here (unlike Blackhole, which
    # sets this False). tt-metal's `l1_bank_to_noc_xy` really does emit mirrored
    # worker coords on Wormhole's NoC 1 — measured off the wire, the NoC 1 half
    # of the table is `mirror(NoC 0)` — so dropping them breaks every L1-sharded
    # buffer program. See `ArchProfile.noc1_tensix_mirror_aliases`.
    noc1_tensix_mirror_aliases=True,
)
