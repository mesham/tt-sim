import os
import threading
from abc import ABC, abstractmethod

from tt_sim.device.reset import Resetable


class Clockable(ABC):
    #: Cycle at which this component's current *occupancy* ends, or ``None``
    #: when it retires everything in the tick it is handed (today's behaviour
    #: for every component in the tree).
    #:
    #: This is the consumer-facing half of Phase 4 (see
    #: ``docs/plans/event-driven-pump.md``). Setting it to ``cycle + N`` is a
    #: component saying "I am occupied for N cycles; do not tick me before
    #: then". :meth:`next_wake_cycle` turns that into the pump's stride and
    #: the component's own ``clock_tick`` is expected to no-op until it
    #: arrives (``TensixBackendUnit`` does exactly that). It is the hook
    #: §I's per-unit cycle-cost tables plug into — an ``MVMUL`` costing N
    #: cycles is ``self.busy_until = cycle_num + N`` at issue.
    #:
    #: A class attribute so no component pays for it until it opts in.
    busy_until = None

    @abstractmethod
    def clock_tick(self, cycle_num):
        raise NotImplementedError()

    def next_wake_cycle(self, cycle_num):
        """The next cycle at which ``clock_tick`` could change observable state.

        The Phase 4 tightening of :meth:`is_clock_idle` — "when do you next
        need attention" rather than "are you idle right now". Three possible
        answers, given that ``cycle_num`` has just been ticked:

        * ``cycle_num + 1`` — tick me again next cycle (the busy answer);
        * an integer ``> cycle_num + 1`` — I am armed; nothing I do is
          observable before then, so the pump may skip straight to it;
        * ``None`` — nothing internal to me will *ever* change; only somebody
          else acting on me can (the dormant answer, equivalent to
          ``is_clock_idle() == True``).

        The default derives the answer from :attr:`busy_until` and
        :meth:`is_clock_idle`, so every Phase 1 predicate is lifted into this
        contract unchanged and a component that has opted into neither still
        answers the conservative ``cycle_num + 1``. Override directly only
        when a component can name a deadline that neither expresses.
        """
        busy_until = self.busy_until
        if busy_until is not None and busy_until > cycle_num:
            return busy_until
        return None if self.is_clock_idle() else cycle_num + 1

    def is_clock_idle(self):
        """True when ``clock_tick`` would be a no-op for observable state.

        This is the opt-in half of the event-driven pump (see
        ``docs/plans/event-driven-pump.md``): a :class:`TileClock` stops
        ticking a whole tile once *every* component on it reports idle, and
        resumes on the next external stimulus (a NoC message arriving, a host
        read/write, a reset). The contract a component signs by overriding
        this is narrow and local:

            returning True means "ticking me right now changes nothing an
            observer could see, and nothing internal to me can change that —
            only somebody else acting on me can".

        The default is the conservative answer. A component that has not
        opted in keeps its tile permanently awake, so adding a new clockable
        can never silently break dormancy; it can only fail to speed it up.
        """
        return False


class Clock(Resetable):
    def __init__(self, clockables, on_tick=None):
        self.clock_items = list(clockables)
        self.clock_tick_num = 0
        self.on_tick = on_tick

    def add_clockable(self, clockable):
        self.clock_items.append(clockable)

    def add_clockables(self, clockables):
        self.clock_items.extend(clockables)

    def clock_tick(self, cycle):
        for item in self.clock_items:
            item.clock_tick(cycle)
        if self.on_tick is not None:
            self.on_tick(cycle)

    def reset(self):
        self.clock_tick_num = 0

    def run(self, num_iterations):
        for i in range(num_iterations):
            self.clock_tick(i + self.clock_tick_num)
        self.clock_tick_num += num_iterations


class TileClock(Clock):
    """A per-tile :class:`Clock` that stops ticking while the tile is idle.

    Phase 1 of the event-driven pump (``docs/plans/event-driven-pump.md``).
    The dispatch protocol is unchanged — every component still gets
    ``clock_tick(cycle_num)``, in exactly the order ``get_clocks()`` returns —
    but a tile whose every component reports :meth:`Clockable.is_clock_idle`
    is skipped entirely until something wakes it.

    Two lists, because they answer different questions:

    - ``clock_items`` are *gated*: skipped while dormant.
    - ``always_items`` are ticked every cycle regardless. Today that is only
      the tile-control block, whose ``clock_tick`` just latches the cycle
      counter feeding ``RISCV_DEBUG_REG_WALL_CLOCK_*`` — a pure function of
      the cycle number that an observer can read at any time, including while
      the tile is dormant. Ticking it costs one attribute store per tile.

    Waking is *pushed* by the only three things that can act on a dormant
    tile from outside: a NoC message landing in one of its NIUs
    (``NUI.transmit``), a host read/write through ``TT_Device``, and a reset.
    Each sets :attr:`awake` directly.

    Falling asleep is a *pull*, evaluated **after** the tick so the decision
    is made against post-tick state. Phase 4 replaced the boolean
    ``clock_quiescent()`` probe with ``next_wake(cycle)``, which returns the
    cycle at which the tile next needs attention (``None`` = never, until
    somebody acts on it). The tile then holds one of three states:

    ============================  ==============================
    ``awake``  /  ``wake_at``      meaning
    ============================  ==============================
    ``True``   /  *ignored*        tick me on the next cycle
    ``False``  /  ``N``            armed: tick me again at cycle N
    ``False``  /  ``None``         dormant: only an external stimulus wakes me
    ============================  ==============================

    ``wake()`` only ever sets ``awake``, so a wake racing a dormancy decision
    (a NoC ``transmit`` from a tile ticked later in the same cycle) always
    wins over an armed deadline.
    """

    def __init__(
        self, clockables, *, always=(), quiescent=None, next_wake=None, on_tick=None
    ):
        super().__init__(clockables, on_tick=on_tick)
        self.always_items = list(always)
        #: Callable returning True when the gated components are all idle, or
        #: None. The Phase 1 form, kept for callers that build a TileClock by
        #: hand; ``next_wake`` takes precedence when both are supplied.
        self.quiescent_probe = quiescent
        #: Callable ``(cycle) -> int | None`` — the tile's aggregate
        #: :meth:`Clockable.next_wake_cycle`. What ``TT_Device`` supplies.
        self.wake_probe = next_wake
        self.awake = True
        #: Cycle to resume at while ``not awake``; None means "never".
        self.wake_at = None
        #: Cycles skipped while dormant — diagnostics only.
        self.dormant_cycles = 0
        #: The owning :class:`MultiTileClock`, set by ``add_tile_clock``.
        self.pump = None

    @property
    def current_cycle(self):
        """The cycle currently being simulated, for lazily-sampled state.

        Phase 2: ``TensixTileControl`` reads the wall clock from here on
        demand instead of latching it every tick, which is what lets a
        dormant tile cost nothing at all — and is a prerequisite for Phase
        4's striding, where the skipped cycles have no tick to latch in.
        """
        pump = self.pump
        return pump.current_cycle if pump is not None else self.clock_tick_num

    def wake(self):
        self.awake = True

    def reset(self):
        super().reset()
        self.awake = True
        self.wake_at = None

    def next_event_cycle(self, cycle):
        """Earliest cycle after ``cycle`` at which this clock must run again.

        ``None`` when nothing can happen without an external stimulus. Used by
        :class:`MultiTileClock` to compute a stride; never returns a value
        ``<= cycle``.
        """
        if self.awake:
            return cycle + 1
        wake_at = self.wake_at
        if wake_at is None:
            return None
        return wake_at if wake_at > cycle else cycle + 1

    def clock_tick(self, cycle):
        if not self.awake:
            wake_at = self.wake_at
            if wake_at is None or cycle < wake_at:
                self.dormant_cycles += 1
                for item in self.always_items:
                    item.clock_tick(cycle)
                if self.on_tick is not None:
                    self.on_tick(cycle)
                return
            # An armed deadline has arrived.
            self.awake = True
        for item in self.clock_items:
            item.clock_tick(cycle)
        probe = self.wake_probe
        if probe is not None:
            wake_at = probe(cycle)
            if wake_at is None or wake_at > cycle + 1:
                self.awake = False
                self.wake_at = wake_at
        else:
            quiescent = self.quiescent_probe
            if quiescent is not None and quiescent():
                self.awake = False
                self.wake_at = None
        for item in self.always_items:
            item.clock_tick(cycle)
        if self.on_tick is not None:
            self.on_tick(cycle)


def _truthy(raw, default):
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _threading_enabled_from_env():
    return _truthy(os.environ.get("TT_SIM_THREADED"), False)


def _striding_enabled_from_env():
    return _truthy(os.environ.get("TT_SIM_PUMP_STRIDE"), True)


class MultiTileClock(Clock):
    """Drives one ``Clock`` per tile, one OS thread per *heavy* tile, barrier-synced.

    Looks like a ``Clock`` from the outside so ``Device.run()`` keeps working
    unchanged. Internally holds a list of per-tile Clocks split into
    ``_heavy_tile_clocks`` (Tensix tiles — substantial per-cycle work) and
    ``_cheap_tile_clocks`` (DRAM + eth — mostly idle NUI traffic). Worker
    threads are spawned only for the heavy set; the coordinator (the thread
    calling ``run``) ticks every cheap clock itself in parallel with the
    workers and joins the per-cycle barrier as one extra participant. This
    keeps barrier-participant count at ``len(heavy) + 1`` instead of dragging
    every DRAM/eth tile into a 26-way barrier wait every cycle.

    **Threading is opt-in.** The default is the sequential tick loop;
    threading only engages when ``TT_SIM_THREADED=1`` (or ``true``/``yes``/
    ``on``) is set in the environment AND at least two clocks are flagged
    heavy. Real-workload benchmarks under stock CPython show threading
    regresses wall time 1.5–2.4× because per-cycle Python work doesn't
    release the GIL long enough to amortise the barrier (see ROADMAP §A);
    the opt-in flag preserves the structural code path for free-threaded
    Python 3.13t experiments without affecting the default flow.

    **Striding (Phase 4).** The sequential path does not walk cycle-by-cycle:
    after ticking a cycle it asks every tile clock ``next_event_cycle()`` and
    jumps to the earliest answer, clamped to the end of the requested window.
    ``run(N)`` therefore still advances simulated time by exactly ``N`` — what
    changes is that the *number of ``clock_tick`` calls* is no longer equal to
    the number of cycles. See ``docs/plans/event-driven-pump.md`` ("What a
    cycle count means after Phase 4"). Set ``TT_SIM_PUMP_STRIDE=0`` to fall
    back to the cycle-by-cycle loop; striding is also declined automatically
    for any tile carrying ``always_items``, which by definition need a tick
    on every cycle.
    """

    def __init__(self, on_tick=None, *, force_sequential=False, stride=None):
        super().__init__([], on_tick=on_tick)
        self._tile_clocks: list[Clock] = []
        # Tile clocks flagged as "heavy" run on dedicated worker threads;
        # cheap clocks (DRAM, eth) are ticked by the coordinator each cycle
        # in parallel with the workers. Threading only engages when the
        # caller explicitly opts in (TT_SIM_THREADED=1 in env, or
        # force_sequential=False with the env var unset still defaults to
        # sequential) AND ≥2 clocks are marked heavy.
        self._heavy_tile_clocks: list[Clock] = []
        self._cheap_tile_clocks: list[Clock] = []
        self._threading_enabled = not force_sequential and _threading_enabled_from_env()
        self._striding_enabled = (
            _striding_enabled_from_env() if stride is None else stride
        )
        # False as soon as any registered tile clock carries always_items.
        self._stride_safe = True
        #: Cycles the pump jumped over rather than ticking — diagnostics only.
        self.stride_skipped_cycles = 0
        #: The cycle currently being simulated; between ``run`` calls this is
        #: ``clock_tick_num`` (i.e. one past the last cycle executed). Read
        #: lazily by ``TensixTileControl`` for the wall-clock registers.
        self.current_cycle = 0
        self._workers_started = False
        self._workers: list[threading.Thread] = []
        self._barrier: threading.Barrier | None = None
        self._cycles_to_run = 0
        self._base_cycle = 0
        # Coordinator/worker handshake. Each call to run() increments
        # ``_generation``; workers compare against their last-seen value so
        # spurious wakeups never cause a stale batch to be re-executed and
        # any worker that loops back while the coordinator is still
        # finishing the previous batch sees no new work.
        self._cv = threading.Condition()
        self._generation = 0
        self._workers_done = 0
        self._shutdown = False
        self._error: BaseException | None = None

    def add_tile_clock(self, tile_clock: Clock, *, heavy: bool = False) -> None:
        """Register a per-tile clock with the composite.

        ``heavy=True`` marks the clock as doing substantial per-cycle work
        (e.g. a Tensix tile with five RV cores and the coprocessor); a
        worker thread is spawned for it on the first threaded ``run()``.
        ``heavy=False`` (default) is right for cheap clocks like DRAM and
        eth tiles that only service NoC traffic — these are ticked
        inline by the coordinator thread to keep barrier-participant
        count low.
        """
        if self._workers_started:
            raise RuntimeError(
                "Cannot add a tile clock after worker threads have been spawned"
            )
        self._tile_clocks.append(tile_clock)
        tile_clock.pump = self
        if getattr(tile_clock, "always_items", None) or not hasattr(
            tile_clock, "next_event_cycle"
        ):
            # An always-list is "tick me on literally every cycle", which is
            # the one thing striding cannot honour. Phase 2 emptied the only
            # in-tree always-list (the wall clock is read lazily now), so this
            # is a guard for out-of-tree tiles rather than a live path.
            self._stride_safe = False
        if heavy:
            self._heavy_tile_clocks.append(tile_clock)
        else:
            self._cheap_tile_clocks.append(tile_clock)

    @property
    def _heavy_clock_count(self) -> int:
        return len(self._heavy_tile_clocks)

    def add_clockable(self, clockable):
        raise NotImplementedError(
            "MultiTileClock does not accept individual clockables; use add_tile_clock"
        )

    def add_clockables(self, clockables):
        raise NotImplementedError(
            "MultiTileClock does not accept individual clockables; use add_tile_clock"
        )

    def clock_tick(self, cycle):
        self.current_cycle = cycle
        for tile_clock in self._tile_clocks:
            tile_clock.clock_tick(cycle)
        if self.on_tick is not None:
            self.on_tick(cycle)

    def reset(self):
        super().reset()
        self.current_cycle = 0
        for tile_clock in self._tile_clocks:
            tile_clock.clock_tick_num = 0

    def run(self, num_iterations):
        if num_iterations <= 0:
            return
        if (
            not self._threading_enabled
            or len(self._tile_clocks) <= 1
            or self._heavy_clock_count <= 1
        ):
            if self._striding_enabled and self._stride_safe:
                self._run_strided(num_iterations)
            else:
                self._run_sequential(num_iterations)
            return
        self._run_threaded(num_iterations)

    def _run_sequential(self, num_iterations):
        for i in range(num_iterations):
            cycle = i + self.clock_tick_num
            self.current_cycle = cycle
            for tile_clock in self._tile_clocks:
                tile_clock.clock_tick(cycle)
            if self.on_tick is not None:
                self.on_tick(cycle)
        self.clock_tick_num += num_iterations
        self.current_cycle = self.clock_tick_num
        for tile_clock in self._tile_clocks:
            tile_clock.clock_tick_num = self.clock_tick_num

    def _run_strided(self, num_iterations):
        """Sequential tick loop that jumps over cycles nothing is armed for.

        Semantics are exactly ``_run_sequential``'s minus the cycles in which
        every tile clock declared it had nothing to do: simulated time still
        advances by ``num_iterations`` and a stride is clamped to the end of
        the window, so ``run`` can never overshoot the caller's poll interval.
        A tile is only skipped over while ``next_event_cycle`` says it is not
        needed, which is the same guarantee Phase 1's dormancy rests on —
        striding merely stops *visiting* dormant tiles once per skipped cycle.

        Computing the stride costs a probe per tile, which is real money on a
        workload that never strides (BRISC spins in the firmware loop from
        launch to teardown, so its tile wants every cycle). Hence the fast
        reject: an awake *heavy* tile needs the very next cycle whatever the
        rest of the grid says, and that is one attribute read. Being wrong in
        the "still busy" direction only costs a stride, never correctness, so
        the fast path may read a stale ``awake``; the exact pass below re-reads
        it and so sees a wake pushed by a tile ticked later in the same cycle.
        """
        tile_clocks = self._tile_clocks
        fast_reject = self._heavy_tile_clocks
        on_tick = self.on_tick
        cycle = self.clock_tick_num
        end = cycle + num_iterations
        while cycle < end:
            self.current_cycle = cycle
            for tile_clock in tile_clocks:
                tile_clock.clock_tick(cycle)
            if on_tick is not None:
                on_tick(cycle)
            nxt = cycle + 1
            if nxt < end:
                may_stride = True
                for tile_clock in fast_reject:
                    if tile_clock.awake:
                        may_stride = False
                        break
                if may_stride:
                    # Earliest cycle any tile needs attention again, clamped
                    # to the window. next_event_cycle never returns <= cycle,
                    # so this always makes progress.
                    nxt = end
                    for tile_clock in tile_clocks:
                        when = tile_clock.next_event_cycle(cycle)
                        if when is not None and when < nxt:
                            nxt = when
                    skipped = nxt - cycle - 1
                    if skipped > 0:
                        self.stride_skipped_cycles += skipped
                        for tile_clock in tile_clocks:
                            if not tile_clock.awake:
                                tile_clock.dormant_cycles += skipped
            cycle = nxt
        self.clock_tick_num = end
        self.current_cycle = end
        for tile_clock in tile_clocks:
            tile_clock.clock_tick_num = end

    def _run_threaded(self, num_iterations):
        if not self._workers_started:
            self._start_workers()
        n_workers = len(self._heavy_tile_clocks)
        with self._cv:
            self._error = None
            self._base_cycle = self.clock_tick_num
            self._cycles_to_run = num_iterations
            self._workers_done = 0
            self._generation += 1
            self._cv.notify_all()
        # Coordinator participates in the per-cycle barrier as one extra
        # party alongside the heavy-tile workers; it ticks the cheap (DRAM
        # + eth) clocks inline so that work overlaps with the heavy tiles
        # rather than dragging 22 idle clocks into the barrier wait.
        cheap_clocks = self._cheap_tile_clocks
        barrier = self._barrier
        assert barrier is not None
        barrier_broken = False
        for i in range(num_iterations):
            cycle = self.clock_tick_num + i
            self.current_cycle = cycle
            try:
                for cheap in cheap_clocks:
                    cheap.clock_tick(cycle)
            except Exception as exc:
                with self._cv:
                    if self._error is None:
                        self._error = exc
                barrier.abort()
                barrier_broken = True
                break
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                barrier_broken = True
                break
        if barrier_broken:
            # Restore for the next batch — abort() leaves the barrier in a
            # broken state until reset.
            barrier.reset()
        err: BaseException | None = None
        with self._cv:
            self._cv.wait_for(lambda: self._workers_done == n_workers)
            err = self._error
            self._error = None
        if err is not None:
            raise err
        self.clock_tick_num += num_iterations
        for tile_clock in self._tile_clocks:
            tile_clock.clock_tick_num = self.clock_tick_num

    def _start_workers(self):
        # +1 party for the coordinator thread (the caller of ``run``).
        n_parties = len(self._heavy_tile_clocks) + 1
        self._barrier = threading.Barrier(n_parties, action=self._barrier_action)
        self._workers = []
        for idx, tile_clock in enumerate(self._heavy_tile_clocks):
            t = threading.Thread(
                target=self._worker_loop,
                args=(tile_clock,),
                name=f"tt-sim-tile-{idx}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        self._workers_started = True

    def _barrier_action(self):
        if self.on_tick is not None:
            self.on_tick(self.current_cycle)

    def _worker_loop(self, tile_clock):
        barrier = self._barrier
        assert barrier is not None
        last_seen_generation = 0
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda: self._shutdown or self._generation > last_seen_generation
                )
                if self._shutdown:
                    return
                last_seen_generation = self._generation
                base = self._base_cycle
                cycles = self._cycles_to_run
            try:
                for i in range(cycles):
                    cycle = base + i
                    tile_clock.clock_tick(cycle)
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        break
            except Exception as exc:
                with self._cv:
                    if self._error is None:
                        self._error = exc
                barrier.abort()
            with self._cv:
                self._workers_done += 1
                self._cv.notify_all()

    def shutdown(self):
        if not self._workers_started:
            return
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers_started = False
