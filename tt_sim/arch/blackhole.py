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
    # DRAM-sourced NoC reads must be congruent modulo 64 (ttsim tile.cpp checks
    # `(src_addr & 63) == (dst_addr & 63)` under TT_ARCH_VERSION == 1), matching
    # Blackhole's 64 B NoC byte-enable span vs Wormhole's 32 B.
    noc_dram_read_congruence=64,
    # Tensix L1 SRAM: TENSIX_SRAM_SIZE = 1536 * 1024 = 1,572,864 bytes.
    tensix_l1_size=1536 * 1024,
    # Baby-core local data memories are doubled vs Wormhole (ttsim config.h,
    # TT_ARCH_VERSION 1) — the firmware stack lives at the top of this region.
    brisc_local_mem_size=8 * 1024,
    ncrisc_local_mem_size=8 * 1024,
    trisc_local_mem_size=4 * 1024,
    # Blackhole moves the reset-PC override to the RISCV_DEBUG_REG block.
    baby_core_reset_pc_debug_regs=True,
    # Blackhole's Tensix backend config state is larger than Wormhole's
    # (ttsim sim.h TT_ARCH_VERSION 1: 56 config / 68 thread-config registers).
    tensix_cfg_state_size=56,
    tensix_thd_state_size=68,
    # Use Blackhole Tensix semantics: BH config-register layout + STALLWAIT bits.
    tensix_blackhole=True,
    # Blackhole baby cores add, over RV32IM: Zba address-gen + Zbb basic
    # bit-manip (tt-metal's Blackhole kernels use them, e.g. zext.h / sh2add for
    # loop-bound and pointer arithmetic), Zaamo local-L1 atomics, and an F/Zfh
    # guard (floating-point is partially supported by hardware but not yet
    # modelled — the guard makes any use fail loudly). The V vector extension is
    # TRISC2-only, so its guard is attached there.
    baby_core_isa_extensions=("zba", "zbb", "zaamo", "zfh", "f_guard"),
    trisc2_isa_extensions=("v_guard",),
    # All 8 DRAM channels' worker-visible endpoints, in SoC-descriptor channel
    # order. tt-metal interleaves a buffer's tiles round-robin across every
    # channel (default InterleavedBufferConfig), so an example that reads more
    # than one tile — e.g. `six`'s 128^3 matmul, whose A/B tiles land on all 8
    # banks — reads zeros for any channel not modelled. Each channel's NoC 0 and
    # NoC 1 worker endpoints are *different* subchannels of its 3-endpoint DRAM
    # row: the coords below come from the descriptor's `dram` array indexed by
    # `dram_views[ch].worker_endpoint = [noc0_idx, noc1_idx]`:
    #   ch: dram row              worker_endpoint  -> NoC0     NoC1
    #   0:  [0-0, 0-1, 0-11]      [2, 1]           -> (0,11)   (0,1)
    #   1:  [0-2, 0-10, 0-3]      [0, 1]           -> (0,2)    (0,10)
    #   2:  [0-9, 0-4, 0-8]       [0, 1]           -> (0,9)    (0,4)
    #   3:  [0-5, 0-7, 0-6]       [0, 1]           -> (0,5)    (0,7)
    #   4:  [9-0, 9-1, 9-11]      [2, 1]           -> (9,11)   (9,1)
    #   5:  [9-2, 9-10, 9-3]      [2, 1]           -> (9,3)    (9,10)
    #   6:  [9-9, 9-4, 9-8]       [2, 1]           -> (9,8)    (9,4)
    #   7:  [9-5, 9-7, 9-6]       [2, 1]           -> (9,6)    (9,7)
    # Blackhole uses physical coords as the tile coord, so unified == NoC 0.
    dram_channel_unified_coords=(
        (0, 11),
        (0, 2),
        (0, 9),
        (0, 5),
        (9, 11),
        (9, 3),
        (9, 8),
        (9, 6),
    ),
    dram_channel_physical_noc0_coords=(
        ((0, 11),),
        ((0, 2),),
        ((0, 9),),
        ((0, 5),),
        ((9, 11),),
        ((9, 3),),
        ((9, 8),),
        ((9, 6),),
    ),
    # Each channel's NoC 1 worker endpoint (a different subchannel than NoC 0);
    # a NoC 1 read/write to the channel targets the grid-mirror of this coord.
    dram_channel_physical_noc1_coords=(
        (0, 1),
        (0, 10),
        (0, 4),
        (0, 7),
        (9, 1),
        (9, 10),
        (9, 4),
        (9, 7),
    ),
    # First functional worker: physical (1, 2).
    tensix_unified_coords=((1, 2),),
    # Blackhole holds the destination coord in the dedicated HI register.
    noc_coord_strategy=BlackholeNocCoords(),
    noc_blackhole_cmd_buf_layout=True,
    # Baby-core firmware bases (blackhole/dev_mem_map.h): with no IRAM
    # constraints every core boots from L1. Computed from the mailbox/zeros/LLK
    # chain: BRISC 0x39E0, then NCRISC 0x5BE0, TRISC0/1/2 0x65E0/0x6FE0/0x79E0.
    # (BRISC always boots from the 0x0 boot vector, so it needs no override.)
    ncrisc_firmware_base=0x5BE0,
    trisc0_firmware_base=0x65E0,
    trisc1_firmware_base=0x6FE0,
    trisc2_firmware_base=0x79E0,
)
