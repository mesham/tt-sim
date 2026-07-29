"""The Blackhole A0 architecture profile.

Numbers are sourced in ``docs/plans/blackhole-reference.md`` (ttsim
``src/config.h`` + tt-metal's ``blackhole_140_arch.yaml``). Unlike Wormhole,
Blackhole tiles use their **physical NoC 0 coordinate** as the tile-directory
coord directly (there is no "unified" indirection — that exists only to serve
the Wormhole wire bridge). See ``docs/plans/blackhole-support.md``.

This is a *minimal bring-up* profile: one DRAM channel + one Tensix worker,
matching the scope decision. Expanding to the full 8 DRAM channels / 140 workers
is mechanical (extend the coord tuples), exactly as for Wormhole.
"""

from tt_sim.arch.profile import ArchProfile
from tt_sim.network.noc_coords import BlackholeNocCoords

BLACKHOLE_PROFILE = ArchProfile(
    name="blackhole",
    # NoC grid: 17 columns x 12 rows, per BlackholeA0/NoC/MemoryMap.md and the
    # SoC descriptor's grid.x_size/y_size.
    noc_grid_x=17,
    noc_grid_y=12,
    # Max single NoC transfer: 16384 bytes (config.h NOC_MAX_PACKET_SIZE).
    noc_max_burst_size=16384,
    # Tensix L1 SRAM: TENSIX_SRAM_SIZE = 1536 * 1024 = 1,572,864 bytes.
    tensix_l1_size=1536 * 1024,
    # Baby-core local data memories are doubled vs Wormhole (ttsim config.h,
    # TT_ARCH_VERSION 1) — the firmware stack lives at the top of this region.
    brisc_local_mem_size=8 * 1024,
    ncrisc_local_mem_size=8 * 1024,
    trisc_local_mem_size=4 * 1024,
    # Blackhole moves the reset-PC override to the RISCV_DEBUG_REG block.
    baby_core_reset_pc_debug_regs=True,
    # DRAM channel 0's worker-visible NoC 0 endpoint is physical (0, 11) (from
    # the descriptor's dram_views: channel 0 worker_endpoint noc0 index -> 0-11).
    # Blackhole uses physical coords as the tile coord, so unified == physical.
    dram_channel_unified_coords=((0, 11),),
    dram_channel_physical_noc0_coords=(((0, 11),),),
    # First functional worker: physical (1, 2).
    tensix_unified_coords=((1, 2),),
    # Blackhole holds the destination coord in the dedicated HI register.
    noc_coord_strategy=BlackholeNocCoords(),
    noc_blackhole_cmd_buf_layout=True,
)
