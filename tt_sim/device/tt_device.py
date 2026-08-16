from abc import ABC

from tt_sim.device.clock import MultiTileClock, TileClock
from tt_sim.device.deadlock import (
    DeadlockDetector,
    deadlock_config_from_env,
    unit_stall_config_from_env,
)
from tt_sim.device.device import Device, DeviceTile
from tt_sim.device.reset import Reset
from tt_sim.network.noc_shadow import ShadowReporter
from tt_sim.network.tt_noc import AliasedEndpoint, NocLinkRegistry, resolved_nui
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.trace import enable_from_env
from tt_sim.util.bits import clear_bit, set_bit
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_uint32,
)


def _endpoint_at(nui, cell):
    """The directory entry for ``cell``: ``nui`` itself, or a view standing there.

    A tile answers to more than one NoC cell whenever its channel has several
    worker-visible endpoints (Wormhole's two-ended DRAM columns) or a different
    NoC 1 subchannel from its NoC 0 one (Blackhole DRAM). Its NUI stands on
    exactly one of them, so every other cell needs an
    :class:`~tt_sim.network.tt_noc.AliasedEndpoint` carrying that cell's own
    coord — otherwise the hop model charges the packet a flight to a different
    physical NIU, which on Wormhole was wrong for every one of the 480
    worker/DRAM-endpoint pairs on each NoC (mean 34 cycles, worst 90) and on
    Blackhole for every NoC 1 DRAM flight.
    """
    return nui if cell == (nui.x_coord, nui.y_coord) else AliasedEndpoint(nui, cell)


class TT_Device(Device):
    """Base device: the netlist assembly and every arch-agnostic facility.

    **Everything that is not a per-arch hardware fact belongs here, not in a
    concrete device's ``__init__``.** Wiring a device-level facility (tracing,
    the progress watchdog, diagnostics fan-out, clock/reset registration) in
    one architecture's constructor and not the other's has silently broken
    Blackhole twice: ``TT_SIM_TRACE_*`` was ignored outright, and a wedged
    kernel produced no ``[DEADLOCK]`` report. So the construction sequence is
    fixed here and the architectures fill in only the parts that genuinely
    differ:

    1. :meth:`_begin_construction` — profile, diagnostics, tracing bus, the
       default Tensix coord list. Called first, before any tile exists.
    2. The subclass builds its tiles, using :meth:`_build_dram_tiles` /
       :meth:`_build_tensix_tile` (arch hooks: ``dram_tile_class``,
       ``tensix_tile_class``, :meth:`_tensix_physical_coord`).
    3. ``TT_Device.__init__`` — directories, NoCs, clocks, resets, then the
       watchdog and the second tracing pass.

    ``tt_sim/device/parity_test.py`` fails if an architecture starts doing any
    of this itself.
    """

    #: Concrete tile classes this architecture instantiates. Set by the arch
    #: subclass — importing them here would be circular, since ``tiles.py``
    #: imports this module.
    tensix_tile_class = None
    dram_tile_class = None

    #: ``(noc_number, coord) -> None``, handed to every NUI so a NoC request
    #: addressed to a tile that does not exist yet gets one chance to bring it
    #: into being. Installed with :meth:`set_directory_miss_hook`; ``None``
    #: everywhere except under the wire bridge's on-demand materialisation of
    #: workers, and consulted only on a directory *miss*, so the resolve path
    #: pays nothing for it.
    _directory_miss_hook = None

    #: Class-level defaults so a device assembled without
    #: :meth:`_begin_construction` (direct ``TT_Device`` construction in unit
    #: tests) still has the attributes the registration path reads.
    noc_translation = False
    translated_displacements: dict = {}
    #: NoC 1 cells claimed by a translated coordinate; a mirror registration
    #: must not take one. Empty (and free) outside translated mode.
    _translated_noc1_keys: set = frozenset()

    def _begin_construction(
        self, profile, diagnostics, tensix_coords=None, noc_translation=False
    ):
        """Pre-tile setup shared by every architecture's ``__init__``.

        Must be the first statement of a concrete device's ``__init__``:
        tile construction reads ``self.profile`` and ``self.diagnostics``, and
        the tracing bus has to be live before the first tile publishes a
        construction event. Returns the Tensix coord list to instantiate.

        ``noc_translation`` selects the coordinate space the NoC directories
        are keyed in. It is an explicit argument and **not** read from the
        environment here, deliberately: the wire bridge decides the mode once,
        from :func:`~tt_sim.network.noc_translation.translation_source`, and
        hands the same answer to the device and to the convention guard — so
        the two cannot disagree. Reading the environment in this constructor
        as well would mean a user who exports
        ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` in their shell silently rebuilt
        every device in the test suite, replay guards included, in the other
        convention.
        """
        self.profile = profile
        #: Whether this device accepts translated NoC coordinates *in addition*
        #: to physical ones. Translation is additive here exactly as it is on
        #: silicon: the physical keys stay, and the translated ones name the
        #: same endpoints in a disjoint range.
        self.noc_translation = bool(noc_translation)
        #: ``coord -> (displaced_entry, claiming_nui)`` for every translated
        #: registration that took a cell some other registration already held.
        #: Expected to be empty on Wormhole and to hold exactly the Blackhole
        #: DRAM mirrors that legacy NoC 1 aliasing parks on live worker coords
        #: — see :meth:`_register_tile_internals`. Left on the device so a test
        #: can assert the set rather than trusting the prose.
        self.translated_displacements = {}
        self._translated_noc1_keys = set() if self.noc_translation else frozenset()
        # Saved so ``_build_tensix_tile`` can construct lazily-materialised
        # tiles with the same diagnostic flags as the originals.
        self.diagnostics = (
            DeviceTileDiagnostics() if diagnostics is None else diagnostics
        )
        # Opt-in structured tracing: if TT_SIM_TRACE*=<...> is set in the
        # environment, the bus is enabled and writers are registered before any
        # device-construction event is missed. The state-dump writer needs a
        # device reference, so ``__init__`` calls ``enable_from_env`` again
        # once the tiles exist.
        enable_from_env()
        if tensix_coords is None:
            tensix_coords = profile.tensix_unified_coords
        return tensix_coords

    def _tensix_physical_coord(self, coord):
        """Map a tile-directory coord to its SoC-physical NoC 0 coord.

        Identity by default — Blackhole keys tiles by physical coord. Wormhole
        overrides, since its tile coords are the "unified" band.
        """
        return coord

    def _build_tensix_tile(self, coord):
        """Construct one Tensix tile, with this device's diagnostic flags."""
        d = self.diagnostics
        return self.tensix_tile_class(
            coord[0],
            coord[1],
            *self._tensix_physical_coord(coord),
            d.reportBRISC(),
            d.reportNCRISC(),
            d.reportTRISC0(),
            d.reportTRISC1(),
            d.reportTRISC2(),
            d.reportNoC0(),
            d.reportNoC1(),
            d.getTensixCoprocessorDiagnostics(),
            profile=self.profile,
        )

    def _build_dram_tiles(self):
        """Construct one DRAM tile per channel described by the profile."""
        profile = self.profile
        # A channel whose NoC 1 worker endpoint is a different SoC-physical
        # coord than its NoC 0 one (Blackhole) names it; ``None`` mirrors the
        # NoC 0 coord (Wormhole).
        noc1_coords = profile.dram_channel_physical_noc1_coords or (
            (None,) * len(profile.dram_channel_unified_coords)
        )
        tiles = []
        for unified, physicals, noc1_coord in zip(
            profile.dram_channel_unified_coords,
            profile.dram_channel_physical_noc0_coords,
            noc1_coords,
        ):
            primary = physicals[0]
            tiles.append(
                self.dram_tile_class(
                    unified[0],
                    unified[1],
                    primary[0],
                    primary[1],
                    physicals[1:],
                    profile=profile,
                    noc1_endpoint_coord=noc1_coord,
                )
            )
        return tiles

    def __init__(
        self, device_memory, dram_tiles, tensix_tiles, eth_tiles=(), *, profile
    ):
        #: The architecture profile supplying every per-arch hardware constant.
        self.profile = profile
        self.dram_tiles = list(dram_tiles)
        self.eth_tiles = list(eth_tiles)
        self.tensix_tiles = []

        self.tile_directory = {}
        # NoC directories are shared by reference across every NUI on the
        # corresponding NoC. Storing them as instance attrs lets
        # ``add_tensix_tile`` extend them post-construction.
        self.noc_0_directory = {}
        self.noc_1_directory = {}
        #: ``id(nui) -> tile``, so a directory entry can name the tile behind
        #: it. Only the shadow check reads it; nothing on the NoC hot path does.
        self._tile_of_nui = {}
        #: Canonical (SoC-physical NoC 0) coord -> live Tensix tile. The set the
        #: shadow check protects: a NoC 1 mirror landing on one of these cells
        #: makes a real worker unreachable.
        self._tensix_by_canonical = {}
        #: Records (and warns / raises on) NoC 1 directory shadowing.
        self.shadow_reporter = ShadowReporter()
        # One free-cycle watermark per router-to-router link, per NoC. Shared
        # by reference across every NUI on that NoC, for the same reason the
        # directories are and one stronger: an injection port belongs to one
        # NIU, but a link is crossed by every tile whose route passes through
        # it, so the state cannot live on either end of a transfer. Created
        # before any tile is registered so ``_register_tile_internals`` — the
        # single fan-out point, used by ``add_tensix_tile`` too — can hand each
        # NUI its NoC's registry. Costs nothing with the cost model off: the
        # NUIs never reach it, because ``claim_route_links`` returns early.
        self.noc_link_registries = (NocLinkRegistry(), NocLinkRegistry())

        self.clocks = [MultiTileClock()]
        self.resets = [Reset([])]

        for tile in self.dram_tiles:
            self._register_tile_internals(tile)
        for tile in self.eth_tiles:
            self._register_tile_internals(tile)
        for tile in tensix_tiles:
            self.tensix_tiles.append(tile)
            self._register_tile_internals(tile)

        super().__init__(device_memory, self.clocks, self.resets)

        # Progress watchdog. Wired here rather than per-architecture: it was
        # Wormhole-only for a long time, which meant a wedged Blackhole kernel
        # hung silently with no ``[DEADLOCK]`` diagnostic at all.
        enabled, threshold = deadlock_config_from_env()
        unit_stall_enabled, unit_stall_threshold = unit_stall_config_from_env()
        self.deadlock_detector = DeadlockDetector(
            threshold,
            enabled,
            self.tensix_tiles,
            self.dram_tiles,
            unit_stall_enabled=unit_stall_enabled,
            unit_stall_threshold=unit_stall_threshold,
        )
        # Left unwired when disabled, so TT_SIM_DEADLOCK=0 costs literally
        # nothing per cycle rather than a call that returns immediately.
        if enabled:
            self.clocks[0].on_tick = self.deadlock_detector.tick
            # ...and let it name its own wake cycle, so the Phase 4 pump
            # cannot stride a fully dormant device past a scheduled sample.
            self.clocks[0].on_tick_wake = self.deadlock_detector.next_sample_cycle
        # Second tracing pass: wires the state-dump writer, which needs the
        # (now fully assembled) device to poll.
        enable_from_env(device=self)

    def _register_tile_internals(self, tile):
        """Insert a tile into the directory, NoCs, clocks, and resets.

        Used by ``__init__`` for the initial fan-out and by
        ``add_tensix_tile`` for tiles materialised later. The NoC directories
        are shared by reference, so inserting once per NoC makes the new tile
        reachable from every existing NUI on the same NoC.

        Both NoC directories key by the tile's canonical (SoC-physical
        NoC 0) coord — kernels supply that coord on either NoC, because
        ``NOC_0_X`` / ``NOC_0_Y`` are the identity on both architectures. Real
        tt-metal's bank-to-noc table (consulted by ``TensorAccessor`` and
        friends in the canonical ``programming_examples/``) instead writes
        per-NoC *mirror* coords directly into NoC 1's half of the table — so on
        NoC 1 the kernel actually emits the noc1-mirror of the canonical
        coord. For DRAM (and Tensix) we register both forms as keys:
        canonical on both NoCs, plus the noc1-mirror as an additional
        alias on NoC 1 so the bank-table flow resolves the right tile.
        Extra sub-endpoint aliases (e.g. DRAM channels with two
        worker-visible endpoints) get the same canonical + noc1-mirror
        pair of keys.

        **Under translation NoC 1 keeps one convention, not two**, and that is
        what makes the ambiguity described below stop existing rather than stop
        being reported. The unmirrored key goes on *both* architectures: a
        kernel under translation emits a translated coord for every core type
        the architecture translates, and for the one it does not (Wormhole
        DRAM) it emits the mirror the bank table holds — so nothing addresses a
        tile by an unmirrored NoC 1 coordinate any more, and keeping the key
        only leaves live workers shadowed. What survives alongside the
        translated keys is per-architecture geometry, not preference:

        - **Wormhole keeps its mirrors.** Its translated bands sit off the
          10x12 grid entirely, so a physical NoC 1 coordinate still names one
          tile — which is also what the ID translation table does on silicon,
          being the identity everywhere the physical grid lives. That is why
          untranslated Wormhole DRAM goes on working under translation, and it
          keeps every tile's ``NOC_NODE_ID`` self-coordinate resolvable.
        - **Blackhole drops them.** Its translated coords *are* NoC 0 coords,
          so ``(GRID-1-x, GRID-1-y)`` of one worker is the translated
          coordinate of another and a leftover mirror is a live misroute, not
          dead weight. See ``ArchProfile.translated_coords_off_physical_grid``,
          which is one fact with both consequences.

        **A key naming a cell the tile's NUI does not stand on is registered
        through an** :class:`~tt_sim.network.tt_noc.AliasedEndpoint`
        (:func:`_endpoint_at`), carrying that cell's own coord in this NoC's
        space. Routing is unchanged — the same NUI answers — but the hop model
        then times the packet to where it is actually going rather than to the
        controller's primary interface, which on Wormhole was wrong for every
        one of the 480 worker/DRAM-endpoint pairs on each NoC (mean 34 cycles,
        worst 90) and on Blackhole for every NoC 1 DRAM flight — 65 % of every
        destination its committed replay guards resolve. The invariant that
        catches a recurrence lives in
        ``tt_sim/network/noc_endpoint_consistency_test.py``.

        **NoC 1 precedence** — untranslated, where both conventions are live.
        On NoC 1 a tile is *physically* reachable at its mirror coord
        ``(GRID-1-x, GRID-1-y)``; the canonical (primary) coord is there only
        because ``get_noc_addr`` emits it unmirrored on both NoCs.
        The mirror formula maps coords across the DRAM / worker / eth bands, so
        the two conventions collide on some cells — e.g. with the truncated 4×5
        worker grid, Tensix ``(4, y)`` mirrors onto the DRAM column ``x = 5``
        (Tensix ``(4, 1)`` ↔ DRAM ``(5, 10)``), and DRAM ch0 ``(0, 11)`` mirrors
        onto eth ``(9, 0)``. When both a mirror and a primary want the same NoC 1
        cell the **mirror must win**, because the bank-to-noc table used by the
        canonical ``programming_examples/`` addresses DRAM over NoC 1 by its
        mirror; letting a Tensix/eth *primary* clobber a DRAM *mirror* routes a
        DRAM read into that tile's (smaller) L1 and overflows it. So:

        - mirror registrations are authoritative (plain assignment);
        - primary and alias registrations use ``setdefault`` — they fill only
          cells no mirror has claimed.

        Eth tiles additionally *skip* mirror registration entirely: an eth
        mirror would land on a DRAM canonical coord and steal that DRAM's own
        primary cell, and no bank-to-noc flow addresses eth by mirror anyway
        (single-chip kernels like ``hello_world_datatypes_kernel`` read eth by
        its canonical ``(1, 0)``).

        **Tensix mirrors are per-architecture** — see
        ``ArchProfile.noc1_tensix_mirror_aliases``. Wormhole registers them
        because its ``l1_bank_to_noc_xy`` genuinely emits mirrored worker
        coords on NoC 1; Blackhole does not, because tt-metal's
        ``virtual_noc0_coordinate`` early-outs on that architecture and so has
        never emitted one. Do not generalise either answer to the other arch.

        **Untranslated, NoC 1's table is therefore ambiguous by
        construction**: it holds two coordinate conventions in one
        ``(x, y)``-keyed dict, and where they collide only one tile is
        reachable. That is tolerable for *requests*,
        whose destination coord arrives from the kernel already in one known
        convention — and it is why nothing else may look a tile up here. A
        response is routed by the requesting endpoint itself
        (``NoCDataRequest.reply_to`` / ``NUI.send_response``), not by
        re-resolving a coordinate; doing the latter delivered ACKs and read
        responses to whichever tile happened to own the shadowed cell. See
        ``tt_sim/network/noc_routing_test.py``.

        **The ambiguity is reported, not silently tolerated.** Where a mirror
        takes a cell a live Tensix worker owns canonically — or a worker is
        built into a cell a mirror already holds — that worker is unreachable
        on NoC 1 and its traffic is ACKed by the impostor, so neither end can
        tell. Every such cell goes to :attr:`shadow_reporter`, which names the
        coordinate and both tiles (see ``tt_sim/network/noc_shadow.py``; set
        ``TT_SIM_NOC1_SHADOW=error`` to make it fatal).
        """
        coord = tile.get_coord_pair()
        assert coord not in self.tile_directory, f"tile already registered at {coord}"
        self.tile_directory[coord] = tile
        nui0 = tile.get_noc_nui(0)
        nui1 = tile.get_noc_nui(1)
        self._tile_of_nui[id(nui0)] = tile
        self._tile_of_nui[id(nui1)] = tile
        primary = nui0.get_id_pair()
        self.noc_0_directory[primary] = nui0
        register_mirror = getattr(tile, "register_noc1_mirror", True)
        # ...and Tensix mirrors are per-architecture: Blackhole's bank-to-noc
        # table never emits one, so registering them there only displaces live
        # workers. Wormhole's does, so there they must stay. See
        # ``ArchProfile.noc1_tensix_mirror_aliases``.
        if tile.is_tensix and not self.profile.noc1_tensix_mirror_aliases:
            register_mirror = False
        # A tile whose NoC 1 endpoint is a different SoC-physical coord than its
        # NoC 0 endpoint (Blackhole DRAM: NoC 0 (0,11), NoC 1 (0,1)) mirrors that
        # NoC 1 coord instead of the primary — otherwise kernels addressing the
        # channel over NoC 1 route to an empty grid cell.
        noc1_source = getattr(tile, "noc1_endpoint_coord", None) or primary
        translated_pairs = (
            self.translated_coords_for(tile) if self.noc_translation else ()
        )
        if not self.noc_translation:
            # Primary is non-authoritative on NoC 1: never clobber a mirror.
            self.noc_1_directory.setdefault(primary, nui1)
        elif not self.profile.translated_coords_off_physical_grid:
            # Under translation NoC 1 drops the *unmirrored* key on both
            # architectures; on one whose translated coords are numerically
            # physical coords it drops the mirrors as well, because there a
            # mirror is another tile's translated key rather than dead weight.
            register_mirror = False
        if register_mirror:
            self._register_noc1_mirror(self._noc1_mirror(noc1_source), nui1)
        for alias in getattr(tile, "noc_aliases", ()):
            self.noc_0_directory[alias] = _endpoint_at(nui0, alias)
            if not self.noc_translation:
                self.noc_1_directory.setdefault(
                    alias, _endpoint_at(nui1, self._noc1_mirror(alias))
                )
            if register_mirror:
                self._register_noc1_mirror(self._noc1_mirror(alias), nui1)
        # Before the shadow check, deliberately: a translated key makes its
        # tile reachable on NoC 1, so a check run first would report workers as
        # shadowed that translation has already rescued.
        if self.noc_translation:
            self._register_translated(tile, nui0, nui1, translated_pairs)
            # With the unmirrored keys gone, whether a NoC 1 key *is* a grid
            # cell has one answer per architecture, and the null-route path
            # needs it: see :attr:`NUI.keys_mirror_grid_cells`.
            nui1.keys_mirror_grid_cells = (
                not self.profile.translated_coords_off_physical_grid
            )
        self._check_noc1_shadowing(tile, nui1, primary, noc1_source, register_mirror)
        nui0.set_noc_directory(self.noc_0_directory)
        nui1.set_noc_directory(self.noc_1_directory)
        nui0.directory_miss_hook = self._directory_miss_hook
        nui1.directory_miss_hook = self._directory_miss_hook
        nui0.noc_link_registry = self.noc_link_registries[0]
        nui1.noc_link_registry = self.noc_link_registries[1]
        # Tensix tiles host the bulk of per-cycle work (5 baby RV cores +
        # the coprocessor); DRAM and eth tiles are mostly idle NUI traffic
        # so they should not pull the composite into threaded mode on
        # their own. Only count Tensix toward the auto-engage threshold.
        gated, always = tile.get_clock_partition()
        tile_clock = TileClock(gated, always=always, next_wake=tile.next_wake_cycle)
        tile._bind_clock(tile_clock)
        self.clocks[0].add_tile_clock(tile_clock, heavy=tile.is_tensix)
        self.resets[0].add_resetables(tile.get_resets())

    def _check_noc1_shadowing(self, tile, nui1, primary, noc1_source, register_mirror):
        """Report any live Tensix worker this registration made unreachable.

        Two directions, because a collision can be created from either end and
        the order tiles are built in is not fixed (the wire bridge materialises
        workers on demand):

        1. **This tile is the victim.** Its canonical coord was already claimed
           by somebody else's mirror, so the ``setdefault`` above did nothing.
        2. **This tile is the impostor.** Its mirror landed on a cell a live
           Tensix already owned canonically, and mirrors are authoritative.

        Only Tensix victims are reported. A DRAM tile losing its *canonical*
        NoC 1 cell is not a fault: tt-metal's ``dram_bank_to_noc_xy`` addresses
        DRAM on NoC 1 by mirror, and the canonical key is only an
        accommodation for the hand-written ``driver/`` kernels.

        **Neither direction fires under NoC coordinate translation**, and not
        by exemption — by the finding being false there. A tile that owns a
        translated NoC 1 key is reachable on NoC 1 at that key, whatever holds
        its physical cell, so direction 1 has no victim; and a mirror that
        :meth:`_register_noc1_mirror` declined never took a cell, so direction 2
        has no impostor. Reporting either anyway would put 56 "this worker is
        unreachable" lines on stderr for a Wormhole run in which all 80 are
        reachable — a loud, confident, wrong answer, which is worse than the
        silence this check was written to end.
        """
        if tile.is_tensix:
            self._tensix_by_canonical[primary] = tile
            holder = self.noc_1_directory.get(primary)
            if holder is not nui1 and not tile.translated_coords:
                # ``holder`` is usually another tile's NUI — possibly seen
                # through an ``AliasedEndpoint``, which is why it is unwrapped
                # before the identity lookup — but a cached ``NullEndpoint``
                # from an earlier miss shadows just as hard.
                self.shadow_reporter.report_displaced(
                    primary,
                    tile,
                    self._tile_of_nui.get(id(resolved_nui(holder)), holder),
                )
        if not register_mirror:
            return
        sources = (noc1_source,) + tuple(getattr(tile, "noc_aliases", ()))
        for source in sources:
            cell = self._noc1_mirror(source)
            # Asked of the directory rather than assumed from ``register_mirror``:
            # a mirror can be declined (see :meth:`_register_noc1_mirror`), and a
            # mirror that never claimed the cell displaced nobody. Identical to
            # the unconditional reading whenever nothing is declined, which is
            # every untranslated run.
            if self.noc_1_directory.get(cell) is not nui1:
                continue
            victim = self._tensix_by_canonical.get(cell)
            if victim is None or victim is tile or victim.translated_coords:
                continue
            self.shadow_reporter.report_displaced(cell, victim, tile)

    def _register_noc1_mirror(self, coord, nui1):
        """Claim ``coord`` on NoC 1 for ``nui1`` — unless translation owns it.

        Mirror registrations are authoritative among themselves (see
        :meth:`_register_tile_internals`), but a *translated* coordinate
        outranks them, and this is where that is enforced rather than left to
        registration order.

        **A backstop that can no longer fire**, kept because it is one
        membership test and because what makes it unreachable is a property of
        two architectures rather than of this function. Blackhole was the case
        it was written for — a Tensix core's translated coord is its physical
        coord, so every worker's mirror ``(16-x, 11-y)`` is some other worker's
        translated key — and Blackhole now registers no NoC 1 mirrors at all
        under translation. Wormhole registers them, but its translated coords
        are off the physical grid that every mirror lands in, so the two can
        never contend. Should a third architecture put a translated coord where
        a mirror wants to be, this keeps the translated one, which is right for
        the reason those mirrors are dropped elsewhere: the only flow that
        addresses a tile by mirror is tt-metal's bank-to-noc table, and
        wherever a translated coord exists that table emits translated coords
        instead.

        A no-op outside translated mode: :attr:`_translated_noc1_keys` is empty
        and this is one membership test on an empty set.
        """
        if coord in self._translated_noc1_keys:
            return
        self.noc_1_directory[coord] = _endpoint_at(nui1, coord)

    def translated_coords_for(self, tile):
        """``((translated, physical_noc0), ...)`` for ``tile`` under translation.

        Each entry pairs a translated coordinate the host and kernels address
        this tile by with **the SoC-physical NoC 0 cell it names**. The pairing
        is what keeps the hop model honest: a translated coordinate has no
        position on the physical torus the cost model walks, so the directory
        entry registered under it has to carry the physical cell instead (see
        :meth:`_register_translated`). Returning the translated coord alone
        would put ``(18..25, 18..27)`` into a 10x12 grid walk, which is the
        journey that does not exist.

        Empty means "this tile kind is not translated on this architecture",
        which is a real answer and not a gap: Wormhole does not translate DRAM
        (``wormhole_coordinate_manager.cpp``: "DRAM cores are not translated in
        Wormhole"), so its DRAM keeps its physical coords and its mirrored
        NoC 1 alias in every mode.

        The per-kind hooks below are what an architecture overrides. Dispatch
        is on :attr:`TTDeviceTile.tile_role` rather than on the tile class,
        because this module cannot import the concrete tile types — they
        import it.
        """
        role = tile.tile_role
        if role == "tensix":
            return self._translated_tensix_coords(tile)
        if role == "dram":
            return self._translated_dram_coords(tile)
        if role == "eth":
            return self._translated_eth_coords(tile)
        return ()

    def _translated_tensix_coords(self, tile):
        """Blackhole's answer, and the base one: identity.

        A Blackhole Tensix core's translated coordinate *is* a NoC 0 Tensix
        coordinate, compacted leftwards past harvested columns
        (``blackhole_coordinate_manager.cpp:fill_tensix_noc0_translated_mapping``);
        with nothing harvested it is the physical coord unchanged. Returning it
        rather than ``()`` matters even though the directory key is already
        there: it is what makes the NUI report it from ``NOC_ID_LOGICAL`` on
        **both** NoCs instead of the NoC 1 mirror, which is what the bank
        tables and ``is_local_bank`` expect.
        """
        coord = tile.get_coord_pair()
        return ((coord, coord),)

    def _translated_eth_coords(self, tile):
        return ()

    def _translated_dram_coords(self, tile):
        """From the profile, by channel index; ``()`` when the arch has none.

        The two translated coords are genuinely different sub-endpoints of the
        bank — the NoC 0 one and the NoC 1 one — so each pairs with its *own*
        physical cell, not with the channel's primary. Pairing both with the
        primary would charge every NoC 1 DRAM flight the distance to the NoC 0
        interface, which is the error ``AliasedEndpoint`` exists to stop.
        """
        table = self.profile.dram_channel_translated_coords
        if not table:
            return ()
        index = self.profile.dram_channel_unified_coords.index(tile.get_coord_pair())
        translated_noc0, translated_noc1 = table[index]
        physical_noc0 = self.profile.dram_channel_physical_noc0_coords[index][0]
        noc1_coords = self.profile.dram_channel_physical_noc1_coords
        physical_noc1 = physical_noc0 if noc1_coords is None else noc1_coords[index]
        return ((translated_noc0, physical_noc0), (translated_noc1, physical_noc1))

    def _register_translated(self, tile, nui0, nui1, pairs=None):
        """Add ``tile``'s translated coords to both NoC directories.

        **Additive on NoC 0, and the whole of the story on NoC 1.** NoC 0 keeps
        every physical key exactly as it was — one convention, keys that are
        the cells they name, nothing to give way — which is what lets
        translated mode be opt-in without disturbing anything. On NoC 1 the
        unmirrored key is not carried at all under translation and the mirror
        survives only where it is unambiguous, so what this method registers is
        most of that NoC's table rather than an addition to it (see
        :meth:`_register_tile_internals`).

        **Nothing is displaced on either architecture**, and that is now true
        by construction rather than by arithmetic. Wormhole's translated bands
        — workers at ``{18..25} x {18..27}``, eth at ``{18..25} x {16,17}`` —
        are entirely outside the 10x12 grid every mirror lands in. Blackhole's
        translated coords *are* physical coords and used to overwrite six DRAM
        mirrors here; those mirrors are no longer registered, which is the
        stronger form of the same answer, since a key that is never written
        cannot be resurrected by a change in registration order.
        :attr:`translated_displacements` is kept, and asserted empty, so that a
        change which reintroduces one has to say so.

        **Every entry goes in through** :func:`_endpoint_at`, on the *physical*
        cell the translated coord names — never on the translated coord itself.
        A translated coordinate is an identity, not a grid position: it has no
        place on the torus :func:`~tt_sim.network.tt_noc.noc_hop_count` walks,
        and handing one to that walk is how a cost-model run under translation
        used to spin. Registering the physical cell means the hop model sees
        the same journey it would have seen untranslated, which is the journey
        the packet actually makes — so translated mode is not merely
        non-fatal under ``TT_SIM_COST_MODEL``, it is timed correctly.
        """
        pairs = self.translated_coords_for(tile) if pairs is None else pairs
        if not pairs:
            return
        tile.set_translated_coords([translated for translated, _ in pairs])
        for translated, physical in pairs:
            for noc, directory, nui in (
                (0, self.noc_0_directory, nui0),
                (1, self.noc_1_directory, nui1),
            ):
                cell = physical if noc == 0 else self._noc1_mirror(physical)
                entry = _endpoint_at(nui, cell)
                existing = directory.get(translated)
                if existing is not None and resolved_nui(existing) is not nui:
                    self.translated_displacements[(noc, translated)] = (existing, nui)
                directory[translated] = entry
            self._translated_noc1_keys.add(translated)

    def noc1_mirror(self, canonical):
        """Mirror a canonical (NoC 0 physical) coord to NoC 1's coord space.

        NoC 1's origin is the bottom-right tile of the grid, so the same
        physical tile lives at ``(GRID_X-1-x, GRID_Y-1-y)`` on NoC 1. The grid
        dims come from the architecture profile (10 × 12 for Wormhole). The
        mapping is its own inverse, which is what lets a NoC 1 directory miss
        be tried both ways round (see the wire bridge's worker materialiser).
        """
        return (
            self.profile.noc_grid_x - 1 - canonical[0],
            self.profile.noc_grid_y - 1 - canonical[1],
        )

    #: Historical private name, kept because the mirror rule is quoted by coord
    #: in several docstrings and tests.
    _noc1_mirror = noc1_mirror

    def set_directory_miss_hook(self, hook):
        """Install the callback a NoC request consults when its destination
        coord is not in the directory.

        Handed to every NUI that already exists and to every tile registered
        afterwards. The wire bridge uses it to materialise a functional worker
        the moment a *peer* addresses it, which is the half of on-demand
        materialisation the host's go=GO cannot cover: a worker released from
        reset earlier can send to a tile the host has not launched on yet, and
        dropping that packet is a wrong answer rather than a hang.
        """
        self._directory_miss_hook = hook
        for tile in self.tile_directory.values():
            tile.get_noc_nui(0).directory_miss_hook = hook
            tile.get_noc_nui(1).directory_miss_hook = hook

    def register_tensix_tile(self, tile):
        """Register a TensixTile constructed after device __init__.

        Mirrors what ``__init__`` does for one tile: stitches the tile into
        the directory, both NoC directories, the central Clock / Reset
        aggregators and the progress watchdog. ``add_tensix_tile`` layers a
        coord-based constructor on top.
        """
        self.tensix_tiles.append(tile)
        self._register_tile_internals(tile)
        # The watchdog has to learn about the tile too, or a stall on a
        # lazily-materialised worker is invisible to it.
        self.deadlock_detector.add_tensix_tile(tile)

    def add_tensix_tile(self, coord):
        """Construct and register a Tensix tile at ``coord`` after __init__.

        Used by the wire bridge for lazy multi-Tensix materialisation: when
        tt-metal addresses a worker the simulator hasn't built yet, this stands
        up the matching Tensix tile and wires it into the directory, NoC
        topology, clock/reset aggregators and deadlock detector. Returns the
        new tile. Arch-agnostic — the per-arch part is
        :meth:`_tensix_physical_coord` / ``tensix_tile_class``.
        """
        tile = self._build_tensix_tile(coord)
        self.register_tensix_tile(tile)
        return tile

    def reset(self, reset_number=0):
        """Reset every registered component, waking every tile.

        A reset rewrites core state behind the pump's back, so no tile may
        stay dormant on the strength of a pre-reset quiescence decision.
        """
        for tile in self.tile_directory.values():
            tile.clock.wake()
        super().reset(reset_number)

    def shutdown(self):
        """Join any worker threads spawned by the per-tile clock pump.

        Safe to call on a single-tile / ``TT_SIM_THREADED=0`` device — the
        composite clock no-ops if no workers were ever started. Idempotent.
        """
        for clock in self.clocks:
            shutdown = getattr(clock, "shutdown", None)
            if shutdown is not None:
                shutdown()

    def read(self, coordinate_pair, address, size):
        assert coordinate_pair in self.tile_directory
        tile = self.tile_directory[coordinate_pair]
        # A host-side access is one of the three stimuli that can act on a
        # dormant tile (see ``TileClock``). Reads are woken too: they are rare
        # relative to cycles, and it removes any need to reason about which
        # MMIO reads have side effects.
        tile.clock.wake()
        return tile.read(address, size)

    def write(self, coordinate_pair, address, value, size=None):
        assert coordinate_pair in self.tile_directory
        tile = self.tile_directory[coordinate_pair]
        tile.clock.wake()
        tile.write(address, value, size)

    def deassert_soft_reset(self, coordinate_pair=None, core_type=None):
        if coordinate_pair is None:
            for pair, value in self.tile_directory.items():
                if value.is_tensix:
                    self.perform_soft_reset_change(clear_bit, pair, core_type)
        else:
            self.perform_soft_reset_change(clear_bit, coordinate_pair, core_type)

    def assert_soft_reset(self, coordinate_pair=None, core_type=None):
        if coordinate_pair is None:
            for pair, value in self.tile_directory.items():
                if value.is_tensix:
                    self.perform_soft_reset_change(set_bit, pair, core_type)
        else:
            self.perform_soft_reset_change(set_bit, coordinate_pair, core_type)

    def reset_tile(self, coordinate_pair):
        """Reset only the baby cores on a single tile.

        ``wormhole.reset()`` (the generic Device.reset) resets every baby
        core on every tile, which is fine for single-Tensix flows but
        clobbers PCs on tiles already running firmware in multi-Tensix
        setups. Callers bringing up one tile at a time should use this
        scoped variant.
        """
        tile = self.tile_directory[coordinate_pair]
        tile.clock.wake()
        for core in tile.get_resets():
            core.reset()

    def perform_soft_reset_change(
        self, bit_change_method, coordinate_pair, core_type=None
    ):
        if core_type is not None:
            if core_type == BabyRISCVCoreType.BRISC:
                bit = 11
            elif core_type == BabyRISCVCoreType.NCRISC:
                bit = 18
            elif core_type == BabyRISCVCoreType.TRISC0:
                bit = 12
            elif core_type == BabyRISCVCoreType.TRISC1:
                bit = 13
            elif core_type == BabyRISCVCoreType.TRISC2:
                bit = 14
            elif core_type == BabyRISCVCoreType.ERISC:
                # Per WormholeB0/EthernetTile/SoftReset.md — ERisc reuses
                # bit 11 in this tile's own RISCV_DEBUG_REG_SOFT_RESET_0.
                bit = 11
            else:
                raise NotImplementedError()
            existing_config = conv_to_uint32(self.read(coordinate_pair, 0xFFB121B0, 4))
            new_config = bit_change_method(existing_config, bit)
            if existing_config != new_config:
                self.write(coordinate_pair, 0xFFB121B0, conv_to_bytes(new_config))
        else:
            existing_config = conv_to_uint32(self.read(coordinate_pair, 0xFFB121B0, 4))
            new_config = bit_change_method(existing_config, 11)
            new_config = bit_change_method(new_config, 18)
            new_config = bit_change_method(new_config, 12)
            new_config = bit_change_method(new_config, 13)
            new_config = bit_change_method(new_config, 14)
            if existing_config != new_config:
                self.write(coordinate_pair, 0xFFB121B0, conv_to_bytes(new_config))


class DeviceTileDiagnostics:
    def __init__(
        self,
        brisc_diagnostics=False,
        ncrisc_diagnostics=False,
        trisc0_diagnostics=False,
        trisc1_diagnostics=False,
        trisc2_diagnostics=False,
        noc0_diagnostics=False,
        noc1_diagnostics=False,
        coprocessor_diagnostics=None,
    ):
        self.brisc_diagnostics = brisc_diagnostics
        self.ncrisc_diagnostics = ncrisc_diagnostics
        self.trisc0_diagnostics = trisc0_diagnostics
        self.trisc1_diagnostics = trisc1_diagnostics
        self.trisc2_diagnostics = trisc2_diagnostics
        self.noc0_diagnostics = noc0_diagnostics
        self.noc1_diagnostics = noc1_diagnostics
        self.coprocessor_diagnostics = coprocessor_diagnostics

    def reportBRISC(self):
        return self.brisc_diagnostics

    def reportNCRISC(self):
        return self.ncrisc_diagnostics

    def reportTRISC0(self):
        return self.trisc0_diagnostics

    def reportTRISC1(self):
        return self.trisc1_diagnostics

    def reportTRISC2(self):
        return self.trisc2_diagnostics

    def reportNoC0(self):
        return self.noc0_diagnostics

    def reportNoC1(self):
        return self.noc1_diagnostics

    def getTensixCoprocessorDiagnostics(self):
        return self.coprocessor_diagnostics


class TTDeviceTile(DeviceTile, ABC):
    # Unified-coord band layout:
    #   x ∈ {16..25}                       — DRAM (16-17, 16-18), Tensix workers (18-25, 16-25)
    #   y ∈ {14..15}                       — Ethernet tiles (16-23, 14-15)
    #   y ∈ {16..25}                       — DRAM + Tensix as above
    # Eth lives below the worker/DRAM band so the historical (18, 18) default
    # for Tensix is preserved verbatim.
    UNIFIED_COORD_X_MIN = 16
    UNIFIED_COORD_X_MAX = 25
    UNIFIED_COORD_Y_MIN = 14
    UNIFIED_COORD_Y_MAX = 25

    #: Whether this tile is a Tensix worker. The device treats Tensix tiles
    #: specially (they carry the heavy per-cycle clock load and are the only
    #: tiles that take part in soft reset). Overridden to ``True`` by
    #: ``TensixTile``; keeping the discriminator on the tile avoids the base
    #: device depending on the concrete Wormhole tile classes.
    is_tensix = False

    #: What kind of endpoint this tile is, for the device's per-kind hooks:
    #: ``"tensix"``, ``"dram"``, ``"eth"``, or ``"other"``. Distinct from the
    #: NUI's ``tile_kind`` (which selects a NoC *alignment* rule and is ``"T"``
    #: for eth as well as Tensix, because both are L1), and from
    #: :attr:`is_tensix` (a two-way split the clock partition needs). Used by
    #: :meth:`TT_Device.translated_coords_for`.
    tile_role = "other"

    def get_clock_partition(self):
        """Split ``get_clocks()`` into ``(gated, always)``.

        ``gated`` components are skipped while the tile is dormant; ``always``
        components are ticked every cycle regardless (see
        :class:`~tt_sim.device.clock.TileClock`). The concatenation must equal
        ``get_clocks()`` in order, because ``always`` is ticked after
        ``gated`` — so only a trailing slice may be moved across.
        Default: everything is gated.
        """
        return list(self.get_clocks()), []

    def next_wake_cycle(self, cycle_num):
        """The tile's aggregate :meth:`Clockable.next_wake_cycle`.

        The minimum over the gated components of when each next needs
        attention, with ``None`` (nobody needs anything) meaning the tile can
        go dormant until an external stimulus arrives. Short-circuits as soon
        as any component asks for the very next cycle, which is the common
        case on a live tile.

        This is what the pump consults; it fails safe for exactly the reason
        :meth:`~tt_sim.device.clock.Clockable.is_clock_idle` does — a
        component that has opted into neither predicate answers
        ``cycle_num + 1`` and keeps the tile awake forever. Subclasses
        override to put the cheapest discriminator first (a Tensix tile with
        a core out of reset always needs the next cycle, and finding that out
        costs five attribute reads rather than twenty probe calls).
        """
        soonest = None
        next_cycle = cycle_num + 1
        for probe in self._wake_probes:
            when = probe(cycle_num)
            if when is None:
                continue
            if when <= next_cycle:
                return next_cycle
            if soonest is None or when < soonest:
                soonest = when
        return soonest

    def clock_quiescent(self):
        """True when nothing on this tile can change without outside help.

        The Phase 1 boolean form, now derived from :meth:`next_wake_cycle` so
        the two cannot drift: a tile is quiescent exactly when no component
        names any future cycle at which it needs attention. Not on the pump's
        hot path any more — kept because it reads better at a call site than
        ``next_wake_cycle(...) is None``.
        """
        return self.next_wake_cycle(0) is None

    def _bind_clock(self, tile_clock):
        """Attach the tile's :class:`TileClock` and precompute its probes.

        Called by ``TT_Device`` once the tile clock exists. The wake hooks go
        onto the components that can be acted on from outside while the tile
        is dormant: the two NIUs (a NoC message from another tile) and the
        tile-control block (a soft-reset write from any path).
        """
        self.clock = tile_clock
        gated, _ = self.get_clock_partition()
        self._wake_probes = tuple(item.next_wake_cycle for item in gated)
        self.noc0_router.clock_owner = tile_clock
        self.noc1_router.clock_owner = tile_clock
        tile_ctrl = getattr(self, "tile_ctrl", None)
        if tile_ctrl is not None:
            tile_ctrl.clock_owner = tile_clock

    def __init__(self, coord_x, coord_y, noc0_router, noc1_router):
        # Reference the bounds via ``self`` so an architecture whose tile coords
        # occupy a different range (e.g. Blackhole's 17x12 physical grid) can
        # override the class constants in a subclass.
        if not (
            self.UNIFIED_COORD_X_MIN <= coord_x <= self.UNIFIED_COORD_X_MAX
            and self.UNIFIED_COORD_Y_MIN <= coord_y <= self.UNIFIED_COORD_Y_MAX
        ):
            raise Exception(
                f"Tile coordinate ({coord_x}, {coord_y}) is outside this "
                f"architecture's tile-coordinate range "
                f"(x in {self.UNIFIED_COORD_X_MIN}..{self.UNIFIED_COORD_X_MAX}, "
                f"y in {self.UNIFIED_COORD_Y_MIN}..{self.UNIFIED_COORD_Y_MAX})"
            )
        super().__init__(coord_x, coord_y, noc0_router, noc1_router)
        #: Set by ``TT_Device._register_tile_internals`` via ``_bind_clock``.
        self.clock = None
        self._wake_probes = ()
        # Extra SoC-physical NoC 0 coords that ``TT_Device`` registers as
        # noc-directory aliases on both NoCs (so kernels addressing either
        # sub-endpoint hit the same tile). Default empty; DRAMTile populates.
        self.noc_aliases: tuple[tuple[int, int], ...] = ()
        #: Coords this tile answers to when NoC coordinate translation is on.
        #: Empty unless the device is in translated mode; filled at
        #: registration from ``TT_Device.translated_coords_for``.
        self.translated_coords: tuple[tuple[int, int], ...] = ()

    def set_translated_coords(self, coords):
        """Adopt ``coords`` as this tile's identity in translated space.

        The first entry is the NoC 0 endpoint's translated coord and the last
        the NoC 1 endpoint's — the same for every tile except a Blackhole DRAM
        channel, whose two NoCs are genuinely different sub-endpoints of the
        bank and therefore have different translated coords.
        """
        self.translated_coords = tuple(coords)
        if not self.translated_coords:
            return
        self.noc0_router.set_translated_coord(self.translated_coords[0])
        self.noc1_router.set_translated_coord(self.translated_coords[-1])


# The concrete Wormhole device and its tiles moved to ``wormhole.py``. They are
# re-exported here so ``from tt_sim.device.tt_device import Wormhole`` (and the
# tile classes) keeps working. This is done lazily via module ``__getattr__``
# (PEP 562) rather than a top-level import, so it stays safe regardless of which
# module is imported first — a plain import would be circular, since
# ``wormhole`` imports the base classes from this module.
_REEXPORTED = ("Wormhole", "DRAMTile", "EthTile", "TensixTile")


def __getattr__(name):
    if name in _REEXPORTED:
        from tt_sim.device import wormhole

        return getattr(wormhole, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
