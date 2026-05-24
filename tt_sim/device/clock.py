import os
import threading
from abc import ABC, abstractmethod

from tt_sim.device.reset import Resetable


class Clockable(ABC):
    @abstractmethod
    def clock_tick(self, cycle_num):
        raise NotImplementedError()


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


def _threading_disabled_from_env():
    raw = os.environ.get("TT_SIM_THREADED")
    if raw is None:
        return False
    return raw.strip().lower() in ("0", "false", "no", "off")


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

    Falls back to a plain sequential tick loop when fewer than two clocks
    are flagged heavy or when ``TT_SIM_THREADED=0`` is set in the
    environment.
    """

    def __init__(self, on_tick=None, *, force_sequential=False):
        super().__init__([], on_tick=on_tick)
        self._tile_clocks: list[Clock] = []
        # Tile clocks flagged as "heavy" run on dedicated worker threads;
        # cheap clocks (DRAM, eth) are ticked by the coordinator each cycle
        # in parallel with the workers. Threading only engages when ≥2
        # clocks are marked heavy — otherwise the barrier overhead outweighs
        # the work.
        self._heavy_tile_clocks: list[Clock] = []
        self._cheap_tile_clocks: list[Clock] = []
        self._force_sequential = force_sequential or _threading_disabled_from_env()
        self._workers_started = False
        self._workers: list[threading.Thread] = []
        self._barrier: threading.Barrier | None = None
        self._cycles_to_run = 0
        self._base_cycle = 0
        self._current_cycle = 0
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
        for tile_clock in self._tile_clocks:
            tile_clock.clock_tick(cycle)
        if self.on_tick is not None:
            self.on_tick(cycle)

    def reset(self):
        super().reset()
        for tile_clock in self._tile_clocks:
            tile_clock.clock_tick_num = 0

    def run(self, num_iterations):
        if num_iterations <= 0:
            return
        if (
            self._force_sequential
            or len(self._tile_clocks) <= 1
            or self._heavy_clock_count <= 1
        ):
            self._run_sequential(num_iterations)
            return
        self._run_threaded(num_iterations)

    def _run_sequential(self, num_iterations):
        for i in range(num_iterations):
            cycle = i + self.clock_tick_num
            for tile_clock in self._tile_clocks:
                tile_clock.clock_tick(cycle)
            if self.on_tick is not None:
                self.on_tick(cycle)
        self.clock_tick_num += num_iterations
        for tile_clock in self._tile_clocks:
            tile_clock.clock_tick_num = self.clock_tick_num

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
            self._current_cycle = cycle
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
            self.on_tick(self._current_cycle)

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
