"""Materialise exactly the Tensix workers a program launches on.

The simulator used to build a fixed set of workers chosen before the program
was known — one by default, or whatever ``TT_SIM_TENSIX_COORDS`` /
``TT_SIM_TENSIX_CORES`` named. Both halves of that are bad: a user running a
multi-core program has to work out coordinates by hand, and raising the default
is not the answer, because **a materialised worker a program never launches on
is still live**. tt-metal's init handshake releases BRISC on every declared
worker, so it runs base firmware for the whole run; measured,
``add_2_integers_in_compute`` (a one-core program) costs 7.6 s with one worker
and 28.9 s with 72.

:class:`LazyTensixPool` gets both: no environment variable, and no tax. A
functional worker starts as a :class:`~tt_sim.bridge.cores.DeferredTensixCore`
— wire-identical to the ``NullCore`` fallback, but journalling what the host
says to it while it is held in reset — and becomes a real tile at the
first of three triggers.

**Trigger 1, a host write to a core that is out of reset.** This is the
ordinary one and it fires first in practice: tt-metal releases every declared
worker during device init, then writes kernel binaries only to the cores its
program runs on. So the first write that arrives after the DEASSERT is both the
signal and the deadline — the journal must stop there, because the host has by
then already polled the go-message to ``DONE`` (the stand-in answers it so, out
of its write shadow) and believes the firmware has run its init state.

Trigger 1 is also why **DPRINT materialises the whole declared grid**:
``DPrintServer::init_device`` writes the print-disabled magic to every core on
the device, and it does so after the workers have been released, so every one
of them looks used. That is a wall-time tax (measured on
``hello_world_datamovement_kernel``: 8 s with DPRINT off and one worker, 45 s
with it on and eighty), not a wrong answer, and the cheap mitigations are to
shrink the declared grid or pin the worker set — see
``docs/running-tt-metal-on-the-simulator.md`` §4.7. Removing it would mean
replaying the journal in two phases either side of the go-message settle so
that post-release writes stop being a trigger, which is a change to the part of
this design with the least slack in it; not worth it for a debugging mode.

**Trigger 2, the host's ``go=GO``.** A kernel launch names a core the program
runs on and nothing else (the grid-wide ``go=INIT`` handshake is not a launch).
Kept as a trigger in its own right for a launch that follows no writes at all —
a second launch reusing binaries already in L1.

**Trigger 3, a NoC directory miss.** This is the one that matters, and the
reason the host triggers alone are not enough. Under ``detail::LaunchProgram``
the host writes ``go=GO`` to each of the program's cores in turn, pumping
simulator cycles between wire messages; a worker launched early is running its
kernel thousands of cycles before the last worker is even mentioned. It can
write into a peer's L1, or increment a peer's semaphore, before that peer
exists. Without this trigger ``NUI.resolve_destination`` would null-route that
packet and the run would finish with a **silently wrong answer** — strictly
worse than the hang this whole change replaces. So ``TT_Device`` offers every
directory miss to :meth:`LazyTensixPool.on_directory_miss` first, and the tile
is stood up, replayed and registered *before* the packet is delivered.

Ordering, spelled out, for a worker B addressed by peer A before B is launched:

1. A's request reaches ``resolve_destination((B))`` on some NoC; the directory
   misses.
2. The miss hook materialises B: the tile is constructed, wired into both NoC
   directories, the clock and the watchdog, and B's journal — every host write
   and reset up to its release from reset, in arrival order — is replayed.
3. ``resolve_destination`` retries the lookup, now hits, and A's packet lands
   in B's real L1.
4. B's firmware runs the pending ``go=INIT`` on the cycles already in flight;
   whatever is left of that is settled at B's next host operation.
5. The host's later writes and ``go=GO`` for B find a real core and go straight
   through; nothing is replayed twice.

Nothing is buffered "in between": there is no window in which B exists but is
empty, because materialisation and replay complete inside the miss.

An explicit ``TT_SIM_TENSIX_COORDS`` / ``TT_SIM_TENSIX_CORES`` still pins the
set exactly — no pool is installed at all in that case — because the replay
guards and the cost-model gate want a fixed, reproducible grid.
"""

from tt_sim.bridge.cores import DeferredTensixCore, TensixCore


class _CatchUpTensixCore(TensixCore):
    """A NoC-materialised worker that still owes its init handshake.

    The NoC trigger fires from inside the clock, so it cannot run the firmware
    forward there and then. This defers that to the first host operation on the
    core — the first moment the simulator is provably not inside ``run`` — and
    then gets out of the way: after one settle every method is its parent's
    plus a single ``is None`` test.
    """

    def __init__(self, device, unified_coord, go_addr):
        super().__init__(device, unified_coord)
        self._go_addr = go_addr

    def _catch_up(self):
        if self._go_addr is not None:
            addr, self._go_addr = self._go_addr, None
            self.device.settle_go_message(self.unified, addr)

    def write(self, addr, data):
        self._catch_up()
        super().write(addr, data)

    def read(self, addr, size):
        self._catch_up()
        return super().read(addr, size)

    def assert_reset(self):
        self._catch_up()
        super().assert_reset()

    def deassert_reset(self):
        self._catch_up()
        super().deassert_reset()


class LazyTensixPool:
    """Builds functional workers on demand, from any of the three triggers.

    ``fabric`` gets a core factory that answers worker coords with a deferred
    core; ``device`` (a :class:`tt_sim.bridge.Device`) gets the NoC
    directory-miss hook. ``eager`` names workers to build up front — the
    architecture's default single worker, so the single-tile path is byte-for-
    byte what it always was.
    """

    def __init__(self, fabric, device, tensix_coord_map, *, eager=(), noc_alias=None):
        self.fabric = fabric
        self.device = device
        self.coord_map = tensix_coord_map
        #: ``translated coord -> physical coord``, or ``None`` when NoC
        #: coordinate translation is off. Set, it is also the signal that a
        #: NoC 1 coordinate can no longer be a mirror — see
        #: :meth:`on_directory_miss`, where getting this wrong would null-route
        #: a peer's packet and produce a silently wrong answer.
        self.noc_alias = noc_alias
        #: Worker coords with a real tile behind them.
        self.materialised = set()
        #: Coords materialised after start-up, in order, and why — diagnostics
        #: for a run that ends up with more workers than it expected.
        self.on_demand = []

        for coord in eager:
            self.materialise(coord)

        fabric.core_factory = self.deferred_core
        device.tt_device.set_directory_miss_hook(self.on_directory_miss)

    def deferred_core(self, coord):
        """Fabric core factory: a journalling stand-in for an unbuilt worker.

        Returns ``None`` for anything that is not a functional worker, so the
        fabric falls through to its ``NullCore`` for eth / pcie / arc /
        router-only endpoints exactly as before.
        """
        if coord not in self.coord_map:
            return None
        return DeferredTensixCore(coord, self.materialise)

    def materialise(self, coord, reason="host"):
        """Build the worker at ``coord``, hand it its journal, return its core.

        Idempotent, and safe to re-enter: the coord is marked before any work
        starts, so a packet that arrives at this same coord while we are
        standing it up does not start a second construction.

        Whatever the trigger, the tile has to *catch up* on the init handshake
        it slept through before anything else is done to it — replaying the
        journal leaves ``go=INIT`` in the mailbox and BRISC at its reset PC,
        with not one instruction executed. The **host** triggers settle it
        immediately, which is safe because they run between wire messages. The
        **NoC** trigger cannot: it fires from inside ``tt_device.run``, where a
        nested pump would re-enter the clock with tiles mid-cycle. It gets a
        :class:`_CatchUpTensixCore` instead, which settles at the first host
        operation — by which time the cycles already in flight have usually
        done it anyway, making the settle a single mailbox read.
        """
        if coord in self.materialised:
            return self.fabric.cores[coord]
        self.materialised.add(coord)
        unified = self.device.ensure_tensix_tile(coord)
        pending = self.fabric.cores.get(coord)
        core = TensixCore(self.device, unified)
        if isinstance(pending, DeferredTensixCore):
            # Replay uses the non-pumping device entry points throughout.
            pending_go = pending.replay(self.device, unified)
            if pending_go is not None:
                if reason == "host":
                    self.device.settle_go_message(unified, pending_go)
                else:
                    core = _CatchUpTensixCore(self.device, unified, pending_go)
            self.on_demand.append((coord, reason))
        self.fabric.register(coord, core)
        return core

    def on_directory_miss(self, noc_number, coord):
        """NoC directory miss: materialise the worker it names, if it is one.

        ``coord`` arrives in the NoC's own coordinate space. NoC 0 is keyed by
        the canonical SoC-physical coord — the same key the wire uses — so it
        needs no translation.

        NoC 1 is keyed in **two conventions at once** on an architecture whose
        workers carry mirror aliases (Wormhole): a tile claims both its
        canonical coord and its ``(GRID-1-x, GRID-1-y)`` mirror, the latter
        being what tt-metal's bank-to-noc table emits. Where the two collide
        the mirror registration wins, because it is a plain assignment and the
        canonical one a ``setdefault`` (see
        ``TT_Device._register_tile_internals``). So a NoC 1 miss at ``K`` is
        offered to ``mirror(K)`` *first*: that names the tile a device with
        every worker already built would have had at ``K``, and matching that
        exactly is the point — on-demand materialisation must not change which
        tile a coordinate resolves to, only when it comes into existence. The
        canonical reading is the fallback, for the keys no mirror claims (a
        Wormhole worker at ``x=9`` mirrors onto an empty column, for instance).

        **When ``coord`` is itself a functional worker this is a misdelivery**,
        and one on-demand materialisation *creates* rather than inherits: the
        packet lands on ``mirror(coord)`` — a worker the program may never have
        launched on — and the worker that really lives at ``coord`` is shadowed
        from the moment it is built. Statically, a device that built only these
        two workers would resolve ``coord`` correctly, so this is strictly worse
        than the construction-time hazard rather than a faithful copy of it. It
        is reported for exactly that reason (``TT_SIM_NOC1_SHADOW``); the
        behaviour is unchanged, because matching the all-workers-built device is
        still the least surprising answer available while the directory carries
        two conventions.

        **On Blackhole the mirror reading is skipped entirely**, because there
        is no second convention to reconcile: no Tensix mirror alias is ever
        registered there (``ArchProfile.noc1_tensix_mirror_aliases``), and
        nothing tt-metal emits addresses a Blackhole worker by its mirror. Left
        in, the inference would build a worker the program never launched on
        *and* deliver the packet to it, and then — the aliases being gone —
        fall through and build the real ``coord`` as well. DRAM is untouched by
        this: its mirrors are still registered, so a DRAM key never misses.

        **With NoC coordinate translation on, there is no mirror reading to do
        on either architecture.** The coordinate arrives translated, so it is
        first mapped back to the physical worker it names; and then the mirror
        arm is unreachable, because under translation neither arch's L1 bank
        table emits a mirrored worker coordinate (Wormhole's NoC 1 half becomes
        identical to its NoC 0 half; Blackhole's always was). Trying the mirror
        anyway would materialise the wrong worker and then find the key already
        claimed, leaving the real destination unbuilt and its packet
        null-routed.

        The two gates below stay **separate questions**, chained rather than
        merged: ``noc1_tensix_mirror_aliases`` says whether the *directory*
        carries Tensix mirror keys at all, and ``noc_alias`` says which
        *convention the wire is speaking*. They are not the same fact and
        neither implies the other — Wormhole under translation has the flag
        ``True`` (its mirror aliases are still registered, and correctly so,
        since the untranslated bank table needs them) while the wire carries no
        mirror at all. Collapsing them into one boolean would lose exactly that
        case.
        """
        if self.noc_alias is not None:
            # Identity for a coord the alias does not move — Blackhole's
            # workers, which are the same coordinate in both conventions.
            coord = self.noc_alias.get(coord, coord)
        elif (
            noc_number == 1 and self.device.tt_device.profile.noc1_tensix_mirror_aliases
        ):
            mirrored = self.device.tt_device.noc1_mirror(coord)
            if mirrored in self.coord_map:
                if coord in self.coord_map:
                    self.device.tt_device.shadow_reporter.report_materialised_mirror(
                        coord, mirrored
                    )
                self.materialise(mirrored, reason="noc")
                if coord in self.device.tt_device.noc_1_directory:
                    return
        if coord in self.coord_map:
            self.materialise(coord, reason="noc")
