"""The architecture profile: per-arch hardware constants as plain data.

This is intentionally data-only. Per-architecture *behaviour* that cannot be
reduced to a constant (NoC address encoding, coordinate translation, the SFPU
instruction set) is modelled as strategies threaded in later phases of the
Blackhole port; see ``docs/plans/blackhole-support.md``. Fields are added here
as they are actually threaded through the device — an unused field is a promise
the code does not yet keep, so this grows phase by phase rather than all at once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchProfile:
    """Hardware constants that differ between Tenstorrent architectures."""

    #: Human-readable architecture name, e.g. ``"wormhole"``.
    name: str

    #: NoC torus dimensions (columns x rows) in SoC-physical NoC 0 space. NoC 1's
    #: origin is the opposite corner, so a tile's NoC 1 coord is the mirror
    #: ``(noc_grid_x - 1 - x, noc_grid_y - 1 - y)``.
    noc_grid_x: int
    noc_grid_y: int

    #: Largest single NoC transfer (L1<->L1 burst), in bytes. A request larger
    #: than this is split into this-sized flits by the NIU. 8 KiB on Wormhole,
    #: 16 KiB on Blackhole.
    noc_max_burst_size: int

    #: Tensix L1 SRAM size, in bytes.
    tensix_l1_size: int

    #: Per-baby-core private local data memory sizes, in bytes (at base
    #: ``0xFFB00000``). Blackhole doubles these vs Wormhole (ttsim ``config.h``):
    #: BRISC/NCRISC 8 KiB vs 4 KiB, TRISC 4 KiB vs 2 KiB. The firmware stack sits
    #: at the top of this region, so an undersized value faults on stack writes.
    brisc_local_mem_size: int
    ncrisc_local_mem_size: int
    trisc_local_mem_size: int

    #: Where the NCRISC/TRISC reset-PC override lives: the Tensix backend config
    #: (Wormhole, ``False``) or the RISCV_DEBUG_REG tile-control block
    #: (Blackhole, ``True``). See ``BabyRISCV.get_start_address``.
    baby_core_reset_pc_debug_regs: bool

    #: Unified coord assigned to each physical DRAM channel, one entry per
    #: channel, in the same order as the SoC descriptor's DRAM list.
    dram_channel_unified_coords: tuple[tuple[int, int], ...]

    #: Parallel to :attr:`dram_channel_unified_coords`. Each entry is the
    #: channel's worker-visible SoC-physical NoC 0 coords: element 0 is the
    #: primary (the NUI's id), the rest are aliases the NoC directory also
    #: registers so kernels addressing either sub-endpoint hit the same tile.
    dram_channel_physical_noc0_coords: tuple[tuple[tuple[int, int], ...], ...]

    #: Default Tensix worker tiles to instantiate, as unified coords.
    tensix_unified_coords: tuple[tuple[int, int], ...]

    #: Strategy that reads a NoC request's destination coordinate out of the NIU
    #: command registers (Wormhole packs it into MID; Blackhole uses a dedicated
    #: HI register). See :mod:`tt_sim.network.noc_coords`.
    noc_coord_strategy: object

    #: Whether the NIU command-buffer register layout is Blackhole's (buffers
    #: spaced 0x800 apart, and AT_DATA/CMD_CTRL/NODE_ID shifted by the inserted
    #: AT_LEN_BE_1 register) rather than Wormhole's (0x400 stride). The NUI
    #: normalises Blackhole addresses to the Wormhole-canonical layout.
    noc_blackhole_cmd_buf_layout: bool

    #: Bytes of DRAM one channel exposes to the NoC, i.e. the flat address space
    #: one ``DRAMTile`` answers for. This is the SoC descriptor's
    #: ``dram_bank_size``: 0x8000_0000 (2 GiB) x 6 channels = 12 GiB on
    #: Wormhole, 0xFF00_0000 x 8 = 31.9 GiB on Blackhole. Wormhole's matches the
    #: physical channel (``DRAM_CHANNEL_SIZE`` in the vendor reference
    #: simulator's ``src/config.h``); Blackhole's physical channel is
    #: 0x1_0000_0000 (4 GiB) but the top 16 MiB of a tile's NoC address space is
    #: the register aperture (the NIU block at 0xFFB20000 lives there), so the
    #: descriptor stops short of it and so do we — leaving a stray register
    #: access to a DRAM tile unmapped and loud rather than silently landing in
    #: data. tt-metal's ``dram_view_size`` banking (two 1 GiB views per Wormhole
    #: channel, one view per Blackhole channel) is an offset *inside* this
    #: range. Backed sparsely — see ``SparseAddressableMemory``.
    dram_channel_size: int

    #: RISC-V ISA extensions every baby core implements beyond RV32IM + the
    #: ``.ttinsn`` custom op, as lower-case names (mapped to ISA classes in
    #: ``BabyRISCV``). Wormhole baby cores are RV32IM-only, so this is empty;
    #: Blackhole adds Zba/Zbb bit-manip, Zaamo atomics, and an F/Zfh guard.
    #: See the per-arch ``BabyRISCV/InstructionSet.md`` ISA docs.
    baby_core_isa_extensions: tuple[str, ...] = ()

    #: Extra ISA extensions only the TRISC2 baby core has. The Blackhole ISA doc
    #: notes the V vector extension is "RISC-V T2 only", so its guard lands here
    #: rather than on every core.
    trisc2_isa_extensions: tuple[str, ...] = ()

    #: Default boot PC (firmware base) for NCRISC / TRISC0 / TRISC1 / TRISC2,
    #: used when tt-metal has not set a reset-PC override. ``None`` keeps the
    #: baked-in Wormhole defaults (NCRISC 0x12000, TRISC 0x6000/0xA000/0xE000).
    #: Blackhole has no IRAM constraints and boots every baby core from L1, so it
    #: sets these to its ``MEM_{NC,TRISC*}_FIRMWARE_BASE`` values.
    ncrisc_firmware_base: int | None = None
    trisc0_firmware_base: int | None = None
    trisc1_firmware_base: int | None = None
    trisc2_firmware_base: int | None = None

    #: Sizes of the Tensix coprocessor's backend config state (number of 128-bit
    #: config registers) and per-thread config state. Wormhole 47 / 57;
    #: Blackhole is larger at 56 / 68 (ttsim ``sim.h`` TT_ARCH_VERSION 1). An
    #: undersized value asserts when a kernel writes a high config register.
    tensix_cfg_state_size: int = 47
    tensix_thd_state_size: int = 57

    #: Whether the Tensix coprocessor uses Blackhole semantics: the backend
    #: config-register layout (``tensix_backend_cfg_blackhole.yaml``) and the
    #: STALLWAIT/SEMWAIT condition-mask bit assignments, both of which differ
    #: from Wormhole's. See ``TensixConfigurationConstants`` and ``WaitGate``.
    tensix_blackhole: bool = False

    #: Congruence modulus a NoC **read whose source is DRAM** must satisfy:
    #: ``(src_addr % m) == (dst_addr % m)``. Per ``WormholeB0/NoC/Alignment.md``
    #: the "Other -> L1" cell of the length-mode table is ``C32``, and the vendor
    #: reference simulator (``ttsim`` ``src/tile.cpp``, the ``src_tile_type ==
    #: 'D'`` branch of ``noc_cmd_start``) checks ``(src_addr & 31) == (dst_addr &
    #: 31)`` on Wormhole and ``& 63`` on Blackhole -- matching the 32 B / 64 B NoC
    #: byte-enable span. Violations are ``UndefinedBehavior`` on silicon: the
    #: transfer is skewed or dropped rather than faulting.
    noc_dram_read_congruence: int = 32

    #: Per-channel SoC-physical coordinate of the DRAM tile's **NoC 1** worker
    #: endpoint, parallel to :attr:`dram_channel_unified_coords`. A DRAM channel
    #: exposes several sub-endpoints and NoC 0 / NoC 1 use *different* ones (the
    #: SoC descriptor's ``worker_endpoint`` subchannel indices): Blackhole ch0 is
    #: NoC 0 (0, 11) but NoC 1 (0, 1). The device registers the tile's NoC 1
    #: routing entry at the mirror of this coord. ``None`` keeps the legacy
    #: behaviour of mirroring the NoC 0 coord (correct where both NoCs share the
    #: same physical endpoint, e.g. Wormhole's captured single-NoC flows).
    dram_channel_physical_noc1_coords: tuple[tuple[int, int], ...] | None = None

    #: Whether NoC 1's destination directory must carry a *mirror* alias
    #: ``(noc_grid_x-1-x, noc_grid_y-1-y)`` for each **Tensix** tile, in
    #: addition to its canonical coord. DRAM's mirror aliases are unconditional
    #: and are not covered by this flag.
    #:
    #: This exists because the two architectures genuinely differ, and the
    #: difference is not obvious from either one alone:
    #:
    #: * **Wormhole — ``True``, and load-bearing.** tt-metal builds
    #:   ``l1_bank_to_noc_xy`` through ``RiscFirmwareInitializer::
    #:   virtual_noc0_coordinate``, which mirrors worker coords into NoC 1's
    #:   half of the table whenever the coord is inside the grid and
    #:   translation is off — which is how tt-sim runs. Measured on the wire:
    #:   Wormhole's NoC 1 half of that table is ``mirror(NoC 0)``. Drop the
    #:   aliases here and every L1-sharded / interleaved-L1 buffer program
    #:   (``vecadd_sharding``, ``shard_data_rm``, ...) addresses an empty grid
    #:   cell.
    #: * **Blackhole — ``False``.** The same function early-outs on
    #:   ``|| cluster_.arch() == ARCH::BLACKHOLE``
    #:   (``risc_firmware_initializer.cpp:493-502``), unconditionally and
    #:   regardless of translation, so it has **never** emitted a mirrored
    #:   worker coordinate on Blackhole. Measured: Blackhole's
    #:   ``l1_bank_to_noc_xy`` NoC 1 half is byte-identical to its NoC 0 half,
    #:   and instrumented replays see only canonical worker coords on NoC 1.
    #:   The aliases were therefore dead weight that displaced 96 live workers
    #:   from their own canonical NoC 1 cells.
    #:
    #: **Do not "unify" this across architectures.** Copying Blackhole's value
    #: to Wormhole breaks the DRAM/L1 bank-table flow there;
    #: ``tt_sim/network/noc_routing_test.py`` asserts both values so that a
    #: change which generalises one arch's answer to the other fails loudly.
    noc1_tensix_mirror_aliases: bool = True

    #: ``cnt`` index of ``NOC_ID_LOGICAL`` within the ``NOC_CFG(cnt)`` register
    #: block at NIU offset ``0x100 + cnt*4``. **Architecture-specific and not
    #: interchangeable**: Wormhole numbers it ``0xE`` (offset ``0x138``) and
    #: Blackhole ``0x12`` (offset ``0x148``), because Blackhole's X/Y ID
    #: translation tables are six entries each rather than four
    #: (``hw/inc/internal/tt-1xx/{wormhole,blackhole}/noc/noc_parameters.h``).
    #: Getting it wrong is silent in both directions: the self-coordinate reads
    #: 0, *and* ``0x138`` — a Y translation table on Blackhole — aliases onto
    #: it.
    noc_id_logical_cfg_index: int = 0xE

    #: Whether ``NOC_ID_LOGICAL`` reports the **NoC 1 grid mirror** of the
    #: tile's coordinate on NoC 1, or the canonical (SoC-physical NoC 0) coord
    #: on both NoCs.
    #:
    #: This is the self-address counterpart of
    #: :attr:`noc1_tensix_mirror_aliases` and takes the same per-arch answer for
    #: the same reason. tt-metal firmware fills ``my_x[noc]`` / ``my_y[noc]``
    #: from this register (``risc_init``, ``risc_common.h:194-200``) and then
    #: compares them against the *bank table's* coordinate for that NoC in
    #: ``Noc::is_local_bank`` / ``TensorAccessor::is_local_bank``, and emits
    #: them as a destination in the single-argument ``get_noc_addr(addr)``
    #: (``dataflow_api_addrgen.h:281``). So the value has to agree with
    #: whatever convention that NoC's bank table uses:
    #:
    #: * **Wormhole — ``True``.** ``l1_bank_to_noc_xy``'s NoC 1 half is
    #:   ``mirror(NoC 0)``, and NoC 1's directory carries the matching Tensix
    #:   mirror aliases, so the mirror is both what the kernel compares against
    #:   and a key that resolves back to this tile.
    #: * **Blackhole — ``False``.** Its ``l1_bank_to_noc_xy`` NoC 1 half is
    #:   byte-identical to NoC 0 and nothing there registers a Tensix mirror,
    #:   so the mirror would be a coordinate no kernel expects and no directory
    #:   key resolves. This also matches silicon: Blackhole's *translated*
    #:   Tensix coordinate is a NOC0 Tensix coordinate
    #:   (``blackhole_coordinate_manager.cpp:100-148``), the same on both NoCs.
    #:
    #: ``NOC_NODE_ID`` is unaffected and stays the per-NoC physical node ID
    #: (mirrored on NoC 1) on both architectures — that is what the register is.
    noc_id_logical_mirrored_on_noc1: bool = True

    #: Whether this architecture's translated coordinates fall entirely
    #: *outside* the SoC-physical grid. One geometric fact with two
    #: consequences, both about NoC 1 under translation.
    #:
    #: * **Wormhole — ``True``.** Workers translate to ``(18..25, 18..27)`` and
    #:   ethernet to ``(18..25, {16,17})``, none of which is a cell of the
    #:   10x12 grid. So a physical coordinate still names exactly one tile and
    #:   the NoC 1 mirror keys stay: the ID translation table is the identity
    #:   everywhere the physical grid lives, which is the same reason Wormhole
    #:   DRAM — never translated at all — goes on being addressed by its mirror
    #:   under translation. A tile's ``NOC_NODE_ID`` self-coordinate is that
    #:   mirror, and it goes on resolving.
    #: * **Blackhole — ``False``.** A Blackhole core's translated coord *is* a
    #:   NoC 0 Tensix coord (``blackhole_coordinate_manager.cpp``), so the two
    #:   ranges are numerically the same one and every coordinate is read as
    #:   translated. There is no second space to keep, and a leftover mirror
    #:   key would not be merely dead — ``(GRID-1-x, GRID-1-y)`` of one worker
    #:   is the translated coordinate of another, so it would misroute live
    #:   traffic. Its mirror keys therefore go under translation.
    #:
    #: The second consequence is that a NoC 1 directory key is a grid cell
    #: where this is ``True`` and a translated identity — whose cell is the
    #: key's mirror — where it is ``False``; the null-route path needs the
    #: difference (:attr:`~tt_sim.network.tt_noc.NUI.keys_mirror_grid_cells`).
    #: Asserted against the built device rather than trusted, in
    #: ``tt_sim/device/noc_translation_test.py``.
    translated_coords_off_physical_grid: bool = True

    #: Per-channel **translated** coords of the DRAM tile's worker-visible
    #: endpoints — element 0 the NoC 0 sub-endpoint's, element 1 the NoC 1
    #: sub-endpoint's — parallel to :attr:`dram_channel_unified_coords`. Empty
    #: on an architecture that does not translate DRAM: Wormhole's virtualised
    #: core types are ``{TENSIX, ETH}`` only (``wh_hal.cpp``) and its
    #: coordinate manager says outright that "DRAM cores are not translated in
    #: Wormhole", so its DRAM keeps physical coords and the mirrored NoC 1
    #: alias in every mode. Blackhole translates DRAM
    #: (``bh_hal.cpp`` lists it) and the values come from
    #: ``BlackholeCoordinateManager::map_dram_banks``.
    dram_channel_translated_coords: tuple[tuple[tuple[int, int], ...], ...] = ()

    #: Bytes of a ``DRAMTile``'s address space served by one **physical** GDDR6
    #: channel, or ``None`` when the split is not published. This is not the
    #: same thing as :attr:`dram_channel_size`, which is the whole flat range a
    #: tile answers for: ``wh_dram`` says "DRAM tiles occur in groups of three,
    #: with two channels of GDDR6 present in each group", and its NoC address
    #: map names the halves -- "GDDR6 Channel 0 data" from ``0x0_0000_0000``
    #: and "GDDR6 Channel 1 data" from ``0x0_4000_0000``. Consumed only by the
    #: cost model, which queues requests per physical channel: two independent
    #: controllers must not be modelled as one queue, because that would invent
    #: serialisation the hardware does not have. ``None`` (Blackhole, which has
    #: no DRAM tile page in the ISA docs at all) means one queue per tile.
    #: That is the right shape rather than a fallback: the Blackhole part the
    #: descriptors describe fronts one bank per physical DRAM core, so one
    #: queue per tile is one queue per channel. It stopped being moot on
    #: 2026-08-12, when that arch gained a derived read rate and its reads
    #: began to queue.
    dram_gddr_channel_size: int | None = None

    @property
    def dram_gddr_channels_per_tile(self) -> int:
        """Physical GDDR6 channels behind one ``DRAMTile``; 1 when unpublished."""
        size = self.dram_gddr_channel_size
        if not size:
            return 1
        return max(1, self.dram_channel_size // size)

    @property
    def noc_kwargs(self) -> dict:
        """The per-arch NoC parameters a tile passes to each ``NUI`` it builds."""
        return {
            "arch": self.name,
            "noc_grid_x": self.noc_grid_x,
            "noc_grid_y": self.noc_grid_y,
            "noc_max_burst_size": self.noc_max_burst_size,
            "noc_coord_strategy": self.noc_coord_strategy,
            "noc_blackhole_cmd_buf_layout": self.noc_blackhole_cmd_buf_layout,
            "noc_dram_read_congruence": self.noc_dram_read_congruence,
            "noc_id_logical_cfg_index": self.noc_id_logical_cfg_index,
            "noc_id_logical_mirrored_on_noc1": self.noc_id_logical_mirrored_on_noc1,
        }
