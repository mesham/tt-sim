"""Progress watchdog.

When the simulator has made no observable forward progress for a configured
number of cycles, a multi-line diagnostic is written to stderr naming the
likely cause (core PC frozen / oscillating, NoC requests outstanding, Tensix
backend wedged). The detector keeps running after each report and re-fires
once per window so the user is told the stall is still ongoing without
flooding output.

Dormant while every BabyRISCV core is in soft reset, since "all cores in
reset" is the normal pre-launch state, not a deadlock.

**Spinning wedges, not just frozen ones.** A wedged device does not have to go
quiet: a core parked in a poll loop retires instructions as fast as the pump
will let it, so the whole simulator sits at 100 % CPU making no forward
progress. The progress signature therefore takes each core's *code footprint* —
the set of 64-byte PC buckets in its recent-PC window — rather than the single
PC bucket it is in at the sample point. An instantaneous bucket is defeated by
any spin loop that straddles a bucket boundary: consecutive samples land either
side of it, the signature changes, and the watchdog re-baselines for ever. That
is exactly what a wedged tt-metal firmware looks like (the go-message wait loop
spans two buckets), and it kept the watchdog silent through a two-launch hang
that ran for 25 minutes at 95 % CPU — see
``tt_sim/pe/tensix/sync_mutex_queue_test``.

A footprint alone would be too eager — a small loop that is genuinely computing
also revisits its own buckets — so the signature carries each active core's
**register file** beside it. That is the difference between the two: a core
doing work moves a register every iteration, a core polling an unchanged flag
re-loads the same value into the same register for ever. A stall is reported
only when *nothing* moved: no new code, no register, no NoC counter, no Tensix
thread state, for the whole window.

The limit is loops longer than the confirmation window can walk: a long
straight-line loop keeps reaching buckets it has not been in yet right to the
end of the burst, and is read as progress. Tight branchy poll loops — what
firmware and kernel waits actually compile to — are caught.

**Sampled, not per-cycle.** Taking the progress signature walks every tile,
every baby core, every NIU and every Tensix thread; on a full 8x10 worker grid
that is ~3 µs per Tensix tile, and doing it on every cycle made the watchdog
92 % of an 80-worker idle pump. It is now taken once every
``threshold // _SAMPLES_PER_WINDOW`` cycles instead, which is all the
resolution a 50,000-cycle window can use. Two consequences, both deliberate:

* **Detection is late by up to one sample interval** (plus a short
  confirmation, below): a stall is reported between ``threshold`` and
  ``threshold + threshold // _SAMPLES_PER_WINDOW + _CONFIRM_TICKS`` cycles
  after the last observable change, rather than exactly at ``threshold``. With
  the defaults that is 50,000–56,314 cycles. Setting a small
  ``TT_SIM_DEADLOCK_THRESHOLD`` shrinks the interval with it (it is a fraction
  of the threshold), so a user who wants prompt detection still gets it.
* **Sampling alone could alias** — a workload whose period happens to match
  the sample interval would look frozen at every sample point. So a stall that
  clears the threshold is *confirmed* by re-taking the signature on
  ``_CONFIRM_TICKS`` consecutive ticks before anything is printed. A genuinely
  wedged device passes trivially (nothing changes at all); a core that is
  actually retiring instructions moves its PC within a handful of cycles and
  aborts the confirmation. The confirmation also refills the recent-PC window
  with consecutive-cycle PCs, so the report reads exactly as it did when the
  detector polled every cycle.

The window is measured in **simulated cycles**, not in pump ticks. The
detector keeps the two from drifting by naming its own wake cycle:
:meth:`DeadlockDetector.next_sample_cycle` is handed to ``MultiTileClock`` as
``on_tick_wake``, so the Phase 4 pump clamps every stride to the next
scheduled sample. Without it a **fully dormant** device — every tile answering
``next_event_cycle() is None`` — was strided over in one jump per ``run()``
call, so a 200,000-cycle ``run`` took exactly one sample and a wedged-but-
dormant device could go unreported indefinitely. That was previously argued
safe on the grounds that dormancy implies every baby core is in soft reset
(``TensixTile.next_wake_cycle`` answers ``cycle + 1`` for any ``soft_active``
core), which is the one state the watchdog deliberately ignores. True today,
but it is an invariant in a *different* module holding up this one's latency
bound; naming the wake cycle makes the bound local. The cost is one call per
*stride decision*, and a stride decision is only reached when no Tensix tile
wants the next cycle — i.e. exactly when the pump has nothing else to do.

Full dormancy is deliberately **not** treated as a deadlock signal in itself.
It is not provable that nothing can wake such a device: under the wire bridge
the host writes to it between ``run`` calls, and "every core in soft reset" is
the ordinary pre-launch and post-completion state.

Configured via two env vars, read in :func:`deadlock_config_from_env`:

* ``TT_SIM_DEADLOCK`` — falsy disables the detector. Default on.
* ``TT_SIM_DEADLOCK_THRESHOLD`` — cycle count between reports. Default 50000.

**Per-unit stalls, which the global signature cannot see.** Everything above
asks "did *anything* change". A Tensix backend unit can be wedged for ever
while the answer stays yes: nothing back-pressures the baby RISC-V cores on
Tensix instruction issue, so the thread that issued the blocked instruction
carries on, the kernel runs to the end, and the rest of the device keeps
making progress right past a unit that will never retire. That is the second
half of the Blackhole ``UNPACR_NOP``/``UNP_ZEROSRC`` deadlock (ROADMAP.md,
"Unpacker dvalid deadlock"): the unpacker re-ran its latched instruction
17,329 times and the launch still reported done.

:meth:`DeadlockDetector._watch_unit_stalls` closes that. Any unit exposing
``blocked_on()`` — today the two unpackers — is watched, and a unit blocked on
the *same* latched instruction for ``TT_SIM_UNIT_STALL_THRESHOLD`` consecutive
cycles is reported, naming the thread, the opcode and the resource waited on.

**The count is exact, not sampled.** A blocked unit is picked up at the arming
cadence, and from then on it is re-checked *every cycle* until it clears; the
run restarts from zero if it does. Counting at a sample cadence instead would
have been a false-positive generator, because a kernel is a loop: an unpacker
that legitimately blocks once per tile, on the same instruction each time, is
blocked at consecutive sample points and a sampled counter cannot tell that
apart from never having moved. The cost of exactness is bounded — the
per-cycle re-check runs only while some unit is actually blocked, and only over
those units — and the ordinary full signature scan keeps its own cadence
throughout, so nothing above this paragraph changes.

The arming pass has **its own cadence**
(:attr:`DeadlockDetector.unit_stall_arm_interval`,
``TT_SIM_UNIT_STALL_THRESHOLD // 8``) rather than riding the global
signature's. Sharing it made the two knobs silently coupled: end-to-end
latency was ``unit_stall_threshold + deadlock_threshold / 8``, so with the
defaults a report needed 16,250 cycles of blocked run, and a user who lowered
``TT_SIM_UNIT_STALL_THRESHOLD`` to catch a short wedge could not make the
report come any sooner. Latency is now ``1.125x`` this check's own threshold,
whatever the other one is set to.

**The cycle threshold is a heuristic and cannot be anything else.** It was
calibrated on a survey of every replay guard, example and differential op test
in the tree (41 workloads, both arches, with and without ``TT_SIM_COST_MODEL``),
whose worst legitimate case was **3,528 cycles** — Wormhole ``sfpumath``,
unpacker 1's ``UNPACR_NOP`` waiting for the SFPU to hand SrcB back — giving
10,000 a 2.8x margin. That survey named its own blind spot: no in-tree workload
pipelined core-to-core, and the legitimate bound is architecturally the
downstream pipeline's stall, not a constant.

``examples/pipestall`` was then built to close it, and it does not close in the
threshold's favour. It is the same eltwise-add pipeline as ``nine`` with a
reverse credit semaphore, so the producer's Tensix backs up behind the consumer
*core*: pack parks in ``cb_reserve_back``, math cannot free Dst, and the
unpackers block on a Src bank for exactly as long as the other core takes.
Measured on Wormhole, with the consumer's per-chunk cost as the only variable
(``PIPESTALL_DELAY``, one runtime argument), longest blocked run:

===========  ======  =========  ====================
delay iters  credits  out CB     longest blocked run
===========  ======  =========  ====================
0            1        1 page     28
100          1        1 page     556
200          1        1 page     1,056  (the frozen guard)
500          1        1 page     2,556
1,000        1        1 page     5,056
2,000        1        1 page     **10,056**
4,000        1        1 page     **20,056**
1,000        2        1 page     5,018
1,000        4        1 page     4,653
1,000        1        4 pages    4,646
2,000        2        2 pages    **10,020**
===========  ======  =========  ====================

Three things follow, and none of them is a tuning problem:

* The blocked run is **linear in the consumer's cost** (``5 * delay + 56``,
  r = 1.000) with no ceiling. 10,000 cycles is reached by a consumer doing
  ~10,000 cycles of ordinary work per tile — a real one packs, writes to DRAM
  and waits on a NoC round trip, so this is not an adversarial figure.
* **Buffering does not save it.** Four credits and a four-page output CB each
  bought under 8 %: once the producer is rate-limited rather than
  latency-limited, the steady-state stall is one consumer turnaround however
  deep the pipe. A threshold cannot be justified by "the user can add slack".
* Raising the threshold to cover this would defeat the other end. The check
  must also catch a *short* wedge, and a minimal ``UNPACR_NOP`` reproduction
  runs ~18,300 cycles end to end — so at the current 1.125x latency anything
  above ~16,000 misses it entirely. The interval between "high enough not to
  false-positive on a correct pipeline" and "low enough to catch a short
  reproduction" is **empty**.

So 10,000 stays, and stays a **warning**: read it as "a unit has been blocked
for longer than a single-core workload ever needs, look at why", not as
"something is wrong". It is the wrong shape for a verdict, and no constant is
the right one.

**Why it is still on by default, and what was changed instead.** It is
simultaneously useful and capable of firing on correct code, and those two
facts pull in opposite directions:

* it has already earned its keep on a third-party kernel — a genuinely broken
  eight-core GEMM produced a report on all eight workers and none on the
  corrected form, each naming unit, thread, opcode and Src bank, which is the
  diagnosis rather than a silent timeout; and
* ``pipestall`` shows it firing on a *correct* pipeline whenever the downstream
  consumer costs more than ~2,000 cycles a tile.

Three fixes were considered and two rejected. **Making it opt-in** forfeits the
whole value: nobody who does not already suspect a wedged unit will set the
variable, and the GEMM's author did not. **Suppressing it when the block is
"explained" by a downstream consumer still making progress** is unsound, and
unsound in the one direction that matters: the true positive *also* runs on a
device that is still making progress — that is the check's entire premise, and
the reason the global signature cannot see it — so the same suppression that
silences ``pipestall`` silences the GEMM.

What changed is what the report *says* and how often it says it.

* **It reads as a hint.** The old text named one cause (the ``UNPACR_NOP``
  dvalid deadlock) and asserted the kernel would report done anyway, which on a
  deep pipeline is a false accusation. It now gives both readings, says the
  check cannot separate them, and names the two lines that can.
* **It is de-duplicated** to one report per unit per waited-on instruction,
  instead of once per detection window. The old count was **run-length
  dependent and therefore meaningless as a severity measure** — the same broken
  GEMM gave 576 reports on one run and 352 on a longer one. The new count is
  "how many units are affected", which is a real quantity, and it drops
  ``pipestall`` from four identical warnings to one.
* **It gets a retraction.** When a reported unit does unblock,
  ``[UNIT STALL CLEARED]`` says so and says what it means. That is the piece
  that lets a user decide in seconds without waiting for teardown: a deep
  pipeline recovers and prints it, a wedge never does.

So the verdict a user acts on is always one of the two follow-up lines —
``[UNIT STALL CLEARED]`` (recovered; it was the pipeline) or ``[UNIT WEDGED]``
(cannot recover; it is the bug) — and ``[UNIT STALL]`` itself only says where
to look. Neither the threshold nor the terminal check moved.

**The terminal wedge, which is a proof rather than a heuristic.** What
separates the ``UNPACR_NOP`` deadlock from a slow pipeline is not duration but
whether the wait can still be satisfied. Handing a Src bank back takes an
instruction, and no thread can issue one from soft reset — so a unit still
blocked once **every baby core on its tile is in reset** can never be
satisfied, whatever the cycle count says. :meth:`_watch_unit_stalls` reports
that as ``[UNIT WEDGED]`` after a short grace period
(``_WEDGE_CONFIRM_CYCLES``, covering a backend instruction still in flight when
the reset landed). It needs no threshold, so it fires on a reproduction far too
short for the cycle check, and it cannot fire on a correct kernel: across the
41 surveyed workloads plus every ``pipestall`` configuration above, no unit has
ever ended a launch still blocked. It is the check that should eventually
*raise*; it stays a warning until it has run a while over the tree without
firing, which is the same discipline the cycle threshold got.

* ``TT_SIM_UNIT_STALL`` — falsy disables both per-unit checks. Default on.
* ``TT_SIM_UNIT_STALL_THRESHOLD`` — consecutive blocked cycles before an
  ``[UNIT STALL]`` report. Default 10000. Deliberately has **no effect on**
  ``[UNIT WEDGED]``, which is not a cycle count: turning this knob up to quiet
  the hint cannot turn the proof off with it.
"""

import os
import sys
from collections import deque

from tt_sim.pe.rv.babyriscv import BabyRISCV
from tt_sim.util.bits import get_nth_bit
from tt_sim.util.conversion import conv_to_uint32

DEFAULT_THRESHOLD = 50000
# Consecutive cycles one backend unit may stay blocked on a single latched
# instruction before it is reported. A heuristic, not a bound — see the module
# docstring: 2.8x the worst legitimate wait among single-core workloads
# (3,528 cycles, Wormhole ``sfpumath``), but ``examples/pipestall`` shows a
# *correct* core-to-core pipeline passing it whenever its consumer costs more
# than ~2,000 cycles per tile. The terminal ``[UNIT WEDGED]`` check is the one
# that decides; this one only points.
DEFAULT_UNIT_STALL_THRESHOLD = 10000
# Window of recent PCs kept per active core for the diagnostic report.
_RECENT_PC_WINDOW = 8
# 64-byte bucket for the change-detection signature: tight spin loops live
# inside one bucket, real iteration walks across them.
_PC_BUCKET_BYTES = 64
# Span used to label a stall as "oscillating" rather than "frozen" or
# "advancing slowly" when writing the report.
_TIGHT_OSCILLATION_BYTES = 32
# Signature samples taken per threshold window. The sample interval is
# ``threshold // _SAMPLES_PER_WINDOW`` (>= 1), so this trades detection
# latency (at most one interval late) against per-cycle cost.
_SAMPLES_PER_WINDOW = 8
# Consecutive ticks the signature must also hold still for before a sampled
# stall is reported — the anti-aliasing guard described in the module docstring.
_CONFIRM_TICKS = 64
# Of those ticks, how many must pass with no core reaching a PC bucket it has
# not already been in. "The PC moved" cannot be the confirmation criterion,
# because a spinning core's PC moves every cycle — what separates a spin from
# progress is that a spin's *code footprint* stops growing, and it stops within
# one trip round the loop. A loop longer than this many instructions keeps
# adding buckets to the end of the window and is read as progress, which is the
# documented limit of the check.
_CONFIRM_SETTLE_TICKS = _CONFIRM_TICKS // 2
# Cycles a blocked unit must stay blocked *after* its whole tile has gone into
# soft reset before the wedge is called terminal. Only covers a backend
# instruction that was already in flight when the reset landed and could still
# hand the bank back; those retire in single-digit cycles, so this is generous
# by an order of magnitude and costs nothing (it is reached only when a unit is
# blocked at teardown, which no correct workload in the tree ever is).
_WEDGE_CONFIRM_CYCLES = 64


def _truthy(val):
    return val is not None and val.strip().lower() in {"1", "true", "yes", "on"}


def _config_from_env(env, enable_var, threshold_var, default_threshold):
    if env is None:
        env = os.environ
    enabled_raw = env.get(enable_var)
    enabled = True if enabled_raw is None else _truthy(enabled_raw)
    threshold_raw = env.get(threshold_var)
    try:
        threshold = (
            int(threshold_raw) if threshold_raw is not None else default_threshold
        )
    except ValueError:
        threshold = default_threshold
    if threshold <= 0:
        threshold = default_threshold
    return enabled, threshold


def deadlock_config_from_env(env=None):
    """Return ``(enabled, threshold)`` from environment variables."""
    return _config_from_env(
        env, "TT_SIM_DEADLOCK", "TT_SIM_DEADLOCK_THRESHOLD", DEFAULT_THRESHOLD
    )


def unit_stall_config_from_env(env=None):
    """Return ``(enabled, threshold)`` for the per-unit stall check.

    Deliberately its own pair of variables rather than a share of
    ``TT_SIM_DEADLOCK_THRESHOLD``: that one is a *no-progress window* and a
    user who shrinks it to surface stalls sooner would otherwise drag this
    threshold down through the measured legitimate maximum and start warning
    about correct kernels.
    """
    return _config_from_env(
        env,
        "TT_SIM_UNIT_STALL",
        "TT_SIM_UNIT_STALL_THRESHOLD",
        DEFAULT_UNIT_STALL_THRESHOLD,
    )


class DeadlockDetector:
    def __init__(
        self,
        threshold,
        enabled,
        tensix_tiles,
        dram_tiles,
        unit_stall_enabled=True,
        unit_stall_threshold=DEFAULT_UNIT_STALL_THRESHOLD,
    ):
        self.threshold = threshold
        self.enabled = enabled
        #: Per-unit stall check — see the module docstring. Independent of the
        #: global signature in every respect: its own threshold, its own
        #: bookkeeping, and it is deliberately *not* cleared by
        #: :meth:`_rebaseline`, because the state that re-baselines the global
        #: window (every core back in soft reset at the end of a launch) is
        #: exactly the state a wedged unit survives into.
        self.unit_stall_enabled = unit_stall_enabled
        self.unit_stall_threshold = unit_stall_threshold
        #: Cycles between *arming* passes for the per-unit watch. Its own
        #: cadence, not the global signature's: sharing the latter made the two
        #: knobs silently coupled, so end-to-end detection latency was
        #: ``TT_SIM_UNIT_STALL_THRESHOLD + TT_SIM_DEADLOCK_THRESHOLD / 8`` and
        #: lowering the unit-stall threshold alone could not make a report
        #: arrive sooner. Latency is now ``1.125x`` this check's own threshold
        #: whatever the other one is set to.
        self.unit_stall_arm_interval = max(
            1, unit_stall_threshold // _SAMPLES_PER_WINDOW
        )
        #: ``(coord, unit_id) -> [unit, blocked_on tuple, cycle it was armed,
        #: cycles the owning tile has been fully in reset while still blocked,
        #: whether ``[UNIT STALL]`` has already been printed for it]`` for every
        #: unit currently under per-cycle watch. Empty on a device with nothing
        #: blocked, which is the common case and the one the sampling cadence is
        #: preserved for.
        self._stall_watch = {}
        #: ``(coord, unit_id, blocked_on tuple)`` already reported. The
        #: de-duplication key — see the module docstring: once per unit per
        #: waited-on instruction, not once per detection window, so the number
        #: of reports counts affected units rather than how long the run was.
        self._stall_reported = set()
        #: Units exposing ``blocked_on()``, resolved once per tile.
        self.stallable_units = []
        #: ``coord -> (tile, cores)``, for the terminal-wedge check.
        self._tiles_by_coord = {}
        #: Cycles between signature samples. One when the threshold is small
        #: enough that per-cycle sampling is what the user asked for.
        self.sample_interval = max(1, threshold // _SAMPLES_PER_WINDOW)
        self.tensix_tiles = []
        self.dram_tiles = list(dram_tiles)

        # Per-tile bookkeeping: keep (coord, tile, cores) tuples so multi-Tensix
        # reports can attribute stalls back to a specific tile. recent_pcs key
        # is (tile_coord, core_type) so cores on different tiles don't collide.
        self.tile_cores = []
        self.nuis = []
        for tile in self.dram_tiles:
            coord = tile.get_coord_pair()
            self.nuis.append((coord, 0, tile.get_noc_nui(0)))
            self.nuis.append((coord, 1, tile.get_noc_nui(1)))
        self.recent_pcs = {}
        #: Per core, the PC buckets seen at the last _RECENT_PC_WINDOW *sampled*
        #: ticks — the core's code footprint, and the signature's PC component.
        #: Fed only at the sampling cadence: the per-cycle confirmation pass
        #: judges growth separately (``_confirm_seen``) rather than sliding this
        #: window at a different rate and changing it under its own feet.
        self.pc_windows = {}
        #: Per core, the architectural registers worth hashing: everything bar
        #: PC / next-PC. Resolved once at registration.
        self.data_registers = {}
        #: During a confirmation burst, per core, the buckets visited so far
        #: (seeded from ``pc_windows``). Growth means the core is walking new
        #: code, i.e. it is not spinning.
        self._confirm_seen = {}
        self._confirm_quiet = 0
        self._next_unit_arm = 0

        for tile in tensix_tiles:
            self.add_tensix_tile(tile)

        self._rebaseline(0)

    def add_tensix_tile(self, tile):
        """Register a TensixTile added after device construction.

        Extends ``tile_cores``, ``nuis``, ``recent_pcs`` for the new tile and
        invalidates the cached signature so the watchdog re-baselines against
        the larger tile set rather than reporting a spurious stall.
        """
        self.tensix_tiles.append(tile)
        coord = tile.get_coord_pair()
        cores = [tile.brisc, tile.ncrisc, tile.trisc0, tile.trisc1, tile.trisc2]
        self.tile_cores.append((coord, tile, cores))
        self._tiles_by_coord[coord] = (tile, cores)
        self.nuis.append((coord, 0, tile.get_noc_nui(0)))
        self.nuis.append((coord, 1, tile.get_noc_nui(1)))
        for unpacker in tile.tensix_coprocessor.getBackend().unpacker_units:
            self.stallable_units.append(
                (coord, f"Unpacker {unpacker.unpacker_id}", unpacker)
            )
        for core in cores:
            self.recent_pcs[(coord, core.core_type)] = deque(maxlen=_RECENT_PC_WINDOW)
            self.pc_windows[(coord, core.core_type)] = deque(maxlen=_RECENT_PC_WINDOW)
            # PC and next-PC live in the same register list as the GPRs, and
            # they move on every cycle whatever the core is doing — including a
            # spin. They are the footprint's job; hashing them here would make
            # the register component say "progress" for every spinning core.
            self.data_registers[(coord, core.core_type)] = [
                r
                for r in core.register_file.registers
                if r is not core.pc_register and r is not core.nextpc_register
            ]
        self._rebaseline(getattr(self, "stall_since", 0))

    def _rebaseline(self, cycle):
        """Forget the current window; nothing is stalled until re-observed.

        Touches the *global* signature only. ``_stall_watch`` survives on
        purpose — see :attr:`unit_stall_enabled`.
        """
        self.last_signature = None
        self.stall_since = cycle
        self._confirm_left = None
        self._confirm_seen = {}
        self._confirm_quiet = 0
        self._next_full_sample = cycle if self.enabled else float("inf")
        # ``_next_unit_arm`` is deliberately not reset here, for the same reason
        # ``_stall_watch`` is not: re-baselining is what the *end of a launch*
        # does, and a unit wedged across one must stay counted.
        self._next_sample_cycle = self._next_full_sample

    def _read_soft_reset(self, tile):
        return conv_to_uint32(tile.tile_ctrl.RISCV_DEBUG_REG_SOFT_RESET_0)

    def next_sample_cycle(self, cycle):
        """Earliest cycle at which :meth:`tick` must run again, or ``None``.

        The pump's ``on_tick_wake`` probe (see the module docstring): it is
        consulted alongside every tile clock's ``next_event_cycle`` when
        :class:`~tt_sim.device.clock.MultiTileClock` computes a stride, so a
        scheduled sample is never jumped over however dormant the device is.
        ``None`` only when the detector is disabled, in which case it is not
        wired at all. Never returns a value ``<= cycle``, so it cannot stall
        the pump.
        """
        nxt = self._next_sample_cycle
        if nxt == float("inf"):
            return None
        return nxt if nxt > cycle else cycle + 1

    def tick(self, cycle):
        # Hot path: one comparison per pumped cycle. ``_next_sample_cycle`` is
        # the whole gate — the sampling interval, the confirmation phase (which
        # sets it to the next cycle), the per-unit stall watch (likewise) and
        # ``enabled`` all express themselves through it, so nothing else is
        # read on the common path.
        if cycle < self._next_sample_cycle:
            return
        self._sample(cycle)

    def _schedule(self, cycle, next_full_sample):
        """Arm the next full signature scan, keeping any stall watch per-cycle.

        The full scan's cadence is what it always was; the watch just borrows
        the same wake-up so :meth:`tick` stays a single comparison. The
        per-unit arming pass has its own cadence and so contributes its own
        candidate wake-up — whichever of the two is sooner wins.
        """
        self._next_full_sample = next_full_sample
        if self._stall_watch:
            self._next_sample_cycle = cycle + 1
        elif self.unit_stall_enabled:
            self._next_sample_cycle = min(next_full_sample, self._next_unit_arm)
        else:
            self._next_sample_cycle = next_full_sample

    def _sample(self, cycle):
        if not self.enabled:
            self._next_sample_cycle = float("inf")
            return

        # --- per-unit stall check, independent of the global signature ---
        # Runs ahead of everything below, including the ``not active`` early
        # return: "every core back in soft reset" ends a launch and rebaselines
        # the global window, and a unit wedged at that point is precisely the
        # case this exists for.
        if self._stall_watch:
            self._watch_unit_stalls(cycle)
        if self.unit_stall_enabled and cycle >= self._next_unit_arm:
            self._arm_unit_stall_watch(cycle)
            self._next_unit_arm = cycle + self.unit_stall_arm_interval
        if cycle < self._next_full_sample:
            # Woken for the unit watch (per-cycle) or for its arming pass, but
            # the heavy signature scan is not due yet.
            self._schedule(cycle, self._next_full_sample)
            return

        if cycle < self.stall_since:
            # The clock rewound (device reset); re-baseline rather than
            # measuring a negative window.
            self.stall_since = cycle

        confirming = self._confirm_left is not None
        active = []  # list of (coord, core, pc)
        footprint_grew = False
        for coord, tile, cores in self.tile_cores:
            reset_val = self._read_soft_reset(tile)
            for core in cores:
                key = (coord, core.core_type)
                bit = BabyRISCV.CORE_TYPE_TO_SOFT_RESET_BIT[core.core_type]
                if get_nth_bit(reset_val, bit) == 1:
                    self.recent_pcs[key].clear()
                    self.pc_windows[key].clear()
                else:
                    pc = conv_to_uint32(core.register_file["pc"].read())
                    self.recent_pcs[key].append(pc)
                    bucket = pc // _PC_BUCKET_BYTES
                    if confirming:
                        # Judge growth against what this core had already been
                        # seen in, so re-walking its own loop is not growth.
                        seen = self._confirm_seen.setdefault(
                            key, set(self.pc_windows[key])
                        )
                        if bucket not in seen:
                            seen.add(bucket)
                            footprint_grew = True
                    else:
                        self.pc_windows[key].append(bucket)
                    active.append((coord, core, pc))

        if not active:
            self._rebaseline(cycle)
            self._schedule(cycle, cycle + self.sample_interval)
            return

        # A core's *code footprint* over the recent-PC window, not the single PC
        # it happens to be at right now. See the module docstring: an
        # instantaneous bucket makes any spin loop that straddles a bucket
        # boundary look like progress, because consecutive samples land in
        # different buckets. The set of buckets recently visited holds still for
        # such a loop and moves for a core that is actually walking new code.
        pc_footprint_sig = tuple(
            (
                coord,
                core.core_type,
                frozenset(self.pc_windows[(coord, core.core_type)]),
                core.unknown_instructions,
            )
            for coord, core, pc in active
        )
        # Architectural register state, which is what keeps the footprint rule
        # from calling a working loop a deadlock. A core computing something
        # moves a register every iteration however small its loop is; a core
        # polling an unchanged flag re-loads the same value into the same
        # register for ever, so its register file is bit-identical sample to
        # sample. ``Register.value`` is a plain ``bytes`` attribute (see
        # ``pe/register/register.py``), so this is ~34 attribute reads per
        # active core per sample and nothing on the per-cycle path.
        reg_sig = tuple(
            (
                coord,
                core.core_type,
                tuple(r.value for r in self.data_registers[(coord, core.core_type)]),
            )
            for coord, core, _pc in active
        )
        nui_sig = tuple(
            (coord, idx, tuple(nui.nui_counters.counters))
            for coord, idx, nui in self.nuis
        )
        tensix_sig = tuple(
            (
                coord,
                t,
                tile.tensix_coprocessor.threads[t].hasInflightInstructions(),
                tile.tensix_coprocessor.CoprocessorDoneCheck(t),
            )
            for coord, tile, _cores in self.tile_cores
            for t in range(3)
        )

        signature = (pc_footprint_sig, reg_sig, nui_sig, tensix_sig)
        if signature != self.last_signature:
            # Progress. Back to sampled polling from here.
            self.last_signature = signature
            self.stall_since = cycle
            self._confirm_left = None
            self._schedule(cycle, cycle + self.sample_interval)
            return

        if confirming:
            # Cycle-by-cycle confirmation. The criterion is not "the PC did not
            # move" — a spinning core's PC moves every cycle — but "no core
            # reached code it had not already been in". A spin's footprint stops
            # growing within one trip round its loop; a core walking new
            # instructions keeps adding buckets right to the end of the burst.
            self._confirm_left -= 1
            self._confirm_quiet = 0 if footprint_grew else self._confirm_quiet + 1
            if self._confirm_left > 0:
                self._schedule(cycle, cycle + 1)
                return
            settled = self._confirm_quiet >= _CONFIRM_SETTLE_TICKS
            self._confirm_left = None
            self._confirm_seen = {}
            self._confirm_quiet = 0
            if not settled:
                # Still reaching new code: sampling had aliased, and this is a
                # working device. Start the window again.
                self.stall_since = cycle
                self._schedule(cycle, cycle + self.sample_interval)
                return
            self._report(cycle, active)
            # Re-arm: one report per window, as before.
            self.stall_since = cycle
            self._schedule(cycle, cycle + self.sample_interval)
            return

        if cycle - self.stall_since >= self.threshold:
            # Sampled evidence of a stall; confirm it cycle-by-cycle before
            # printing anything.
            self._confirm_left = _CONFIRM_TICKS
            self._confirm_seen = {}
            self._confirm_quiet = 0
            self._schedule(cycle, cycle + 1)
        else:
            self._schedule(cycle, cycle + self.sample_interval)

    def _arm_unit_stall_watch(self, cycle):
        """Put every currently-blocked unit under per-cycle watch.

        Runs at :attr:`unit_stall_arm_interval`, so it is one ``blocked_on()``
        call per unit per interval on a healthy device — and ``blocked_on()``
        reads a boolean and returns ``None``. The count deliberately starts at
        zero here rather than trying to reconstruct how long the unit had
        already been blocked: undercounting by up to one interval delays a true
        report and can never manufacture a false one.
        """
        watch = self._stall_watch
        for coord, name, unit in self.stallable_units:
            key = (coord, name)
            # Never re-arm a unit already under watch: that would reset its
            # count once per interval and the threshold would be unreachable.
            if key in watch:
                continue
            waiting = unit.blocked_on()
            if waiting is not None:
                watch[key] = [unit, waiting, cycle, 0, False, False]

    def _tile_fully_in_reset(self, coord):
        """Is every baby core on ``coord``'s tile held in soft reset?

        The state in which a blocked unit's wait can no longer be satisfied:
        the Src bank is handed back by an instruction some thread has to issue,
        and no thread can issue anything from reset.
        """
        entry = self._tiles_by_coord.get(coord)
        if entry is None:
            return False
        tile, cores = entry
        reset_val = self._read_soft_reset(tile)
        for core in cores:
            bit = BabyRISCV.CORE_TYPE_TO_SOFT_RESET_BIT[core.core_type]
            if get_nth_bit(reset_val, bit) == 0:
                return False
        return True

    def _watch_unit_stalls(self, cycle):
        """Per-cycle re-check of the watched units. Exact by construction.

        A unit drops out the moment it is no longer blocked on the same latched
        instruction, so what survives to the threshold really is one unbroken
        run of blocked cycles — which is the whole reason this is not folded
        into the sampled signature above.

        Two independent reports come out of this loop, and the second is the
        one that can be trusted absolutely — see the module docstring:

        * the **cycle threshold**, a heuristic ("this is taking a suspiciously
          long time"), which a correct core-to-core pipeline can trip; and
        * the **terminal wedge**, a proof ("this can no longer be satisfied"),
          reached when the whole tile has gone into soft reset with the unit
          still blocked.

        A unit dropping out after it was reported is itself information — it
        settles the hint the other way — so that transition prints
        ``[UNIT STALL CLEARED]`` rather than passing silently.
        """
        reset_cache = {}
        for key, entry in list(self._stall_watch.items()):
            unit, waiting, since, in_reset, reported, wedged = entry
            now = unit.blocked_on()
            if now is None or now != waiting:
                del self._stall_watch[key]
                # Not after a wedge report: a unit whose block "clears" once the
                # tile it is on has been torn down and rebuilt did not recover,
                # and retracting the proof on that basis would be a lie.
                if reported and not wedged:
                    self._report_unit_stall_cleared(key, waiting, cycle - since, cycle)
                continue
            coord = key[0]
            if coord not in reset_cache:
                reset_cache[coord] = self._tile_fully_in_reset(coord)
            if reset_cache[coord]:
                # Grace period: a Src bank can still be handed back by a
                # backend instruction that was already in flight when the cores
                # were reset, and that retires within a handful of cycles.
                entry[3] = in_reset + 1
                if entry[3] == _WEDGE_CONFIRM_CYCLES:
                    entry[5] = True
                    self._report_unit_wedged(key, waiting, cycle)
            else:
                entry[3] = 0
            if not reported and cycle - since >= self.unit_stall_threshold:
                # One report per unit per waited-on instruction, for the whole
                # run — not one per detection window. Repeats carried no new
                # information (same unit, same latched instruction, same bank)
                # and made the report *count* a function of how long the run
                # was rather than of how much is wrong. ``since`` is
                # deliberately not re-armed, so the blocked run keeps
                # accumulating and ``[UNIT STALL CLEARED]`` can state its true
                # length.
                key_seen = (key[0], key[1], waiting)
                if key_seen not in self._stall_reported:
                    self._stall_reported.add(key_seen)
                    entry[4] = True
                    self._report_unit_stall(key, waiting, cycle - since, cycle)

    def _report_unit_wedged(self, key, waiting, cycle):
        coord, name = key
        thread, opcode, which, bank = waiting
        tile_label = f"tile={coord} " if len(self.tile_cores) > 1 else ""
        print(
            "\n".join(
                [
                    f"[UNIT WEDGED cycle={cycle}] {tile_label}{name} is still "
                    f"blocked with every baby core on the tile in soft reset",
                    f"  {opcode} from thread {thread}: waiting for {which} bank "
                    f"{bank} to be given back by the Matrix Unit (dvalid was set "
                    f"and never cleared)",
                    "  Nothing can satisfy that wait any more — handing the bank",
                    "  back takes an instruction, and no thread can issue one from",
                    "  reset. The launch behind this unit will have reported done;",
                    "  on silicon the *next* launch is where it hangs. See",
                    '  ROADMAP.md, "Unpacker dvalid deadlock".',
                    "  This is a proof, not a threshold: no cycle count is",
                    "  involved, no correct kernel can reach it, and it fires on a",
                    "  reproduction far too short for the blocked-cycle hint. Act",
                    "  on it.",
                    "  (TT_SIM_UNIT_STALL=0 to disable)",
                ]
            ),
            file=sys.stderr,
        )

    def _report_unit_stall(self, key, waiting, blocked_cycles, cycle):
        """The hint. Deliberately declines to say which of the two it is.

        Both readings are printed because the check genuinely cannot separate
        them, and the two lines that *can* are named so the reader knows what
        to wait for rather than acting on this one. See the module docstring.
        """
        coord, name = key
        thread, opcode, which, bank = waiting
        tile_label = f"tile={coord} " if len(self.tile_cores) > 1 else ""
        print(
            "\n".join(
                [
                    f"[UNIT STALL cycle={cycle}] {tile_label}{name} has been "
                    f"blocked for {blocked_cycles} cycles while the rest of the "
                    f"device kept running",
                    f"  {opcode} from thread {thread}: waiting for {which} bank "
                    f"{bank} to be given back by the Matrix Unit (dvalid was set "
                    f"and never cleared)",
                    "  This is a hint, not a verdict: it has two readings and "
                    "this check cannot tell them apart.",
                    "    * A deep pipeline. A unit waits on the whole downstream",
                    "      chain — math, Dst, the output CB, the packer, and a",
                    "      consumer that may be on another core — so a consumer",
                    "      costing ~10000 cycles a tile reaches this legitimately.",
                    "      See examples/pipestall.",
                    "    * A real wedge: a Src bank handed to the Matrix Unit and",
                    "      never handed back. Nothing back-pressures the baby",
                    "      RISC-V cores on Tensix instruction issue, so the kernel",
                    "      behind the unit reports done either way. See ROADMAP.md,",
                    '      "Unpacker dvalid deadlock".',
                    "  Rather than act on the line above, wait for whichever of",
                    "  these names this same unit:",
                    "    * [UNIT STALL CLEARED] — it recovered. Deep pipeline;",
                    "      nothing is wrong.",
                    "    * [UNIT WEDGED], at the end of the launch — it never can.",
                    "      That one is a proof, and is the one to act on.",
                    "  Reported once per unit per waited-on instruction, so this",
                    "  will not repeat however long the run is.",
                    "  (TT_SIM_UNIT_STALL=0 to disable, "
                    "TT_SIM_UNIT_STALL_THRESHOLD=N to tune)",
                ]
            ),
            file=sys.stderr,
        )

    def _report_unit_stall_cleared(self, key, waiting, blocked_cycles, cycle):
        """The retraction, and half the value of the hint above.

        Printed only for a unit that was actually reported, so it costs nothing
        on a quiet device. It is what lets a user settle the question during
        the run instead of waiting to see whether ``[UNIT WEDGED]`` turns up at
        teardown.
        """
        coord, name = key
        thread, opcode, which, bank = waiting
        tile_label = f"tile={coord} " if len(self.tile_cores) > 1 else ""
        print(
            "\n".join(
                [
                    f"[UNIT STALL CLEARED cycle={cycle}] {tile_label}{name} "
                    f"unblocked after {blocked_cycles} cycles: {opcode} from "
                    f"thread {thread} got {which} bank {bank} back",
                    "  So the [UNIT STALL] above was a deep pipeline, not a wedge, "
                    "and needs no action.",
                    "  Further blocks of this unit on this instruction are not "
                    "reported.",
                ]
            ),
            file=sys.stderr,
        )

    def _report(self, cycle, active):
        lines = [
            f"[DEADLOCK cycle={cycle}] no observable progress "
            f"for {self.threshold} cycles"
        ]

        for coord, core, _pc in active:
            pcs = list(self.recent_pcs[(coord, core.core_type)])
            if not pcs:
                continue
            pc_lo, pc_hi = min(pcs), max(pcs)
            span = pc_hi - pc_lo
            if span == 0:
                where = f"frozen at {hex(pc_lo)} (memory stall or `j .`)"
            elif span <= _TIGHT_OSCILLATION_BYTES:
                where = (
                    f"oscillating in [{hex(pc_lo)}, {hex(pc_hi)}] "
                    f"({len(set(pcs))} unique PCs) — likely polling"
                )
            else:
                where = f"PCs span [{hex(pc_lo)}, {hex(pc_hi)}]"
            extra = ""
            if core.unknown_instructions > 0:
                extra = f"; {core.unknown_instructions} unknown instructions seen"
            tile_label = f"tile={coord} " if len(self.tile_cores) > 1 else ""
            lines.append(f"  {tile_label}{core.core_type.name}: {where}{extra}")

        for coord, idx, nui in self.nuis:
            outstanding_reads = 0
            outstanding_writes = 0
            for i in range(16):
                outstanding_reads += nui.nui_counters[16 + i]
                outstanding_writes += nui.nui_counters[32 + i]
            if outstanding_reads or outstanding_writes:
                lines.append(
                    f"  NoC tile={coord} nui={idx}: "
                    f"{outstanding_reads} read(s), {outstanding_writes} write(s) "
                    f"outstanding ({len(nui.outstanding_noc_requests)} unresolved)"
                )

        for coord, tile, _cores in self.tile_cores:
            for t in range(3):
                inflight = tile.tensix_coprocessor.threads[t].hasInflightInstructions()
                not_done = tile.tensix_coprocessor.CoprocessorDoneCheck(t)
                if inflight or not_done:
                    tile_label = f"tile={coord} " if len(self.tile_cores) > 1 else ""
                    lines.append(
                        f"  {tile_label}Tensix thread {t}: "
                        f"frontend inflight={inflight}, backend busy={not_done}"
                    )
            # "backend busy" is a symptom. A blocked unpacker names its own
            # cause exactly -- it is waiting for a Src bank the Matrix Unit has
            # not given back -- and that is the shape of every dvalid deadlock,
            # including the one perfbench/tensixbench hit on Blackhole silicon
            # (ROADMAP, "Unpacker dvalid deadlock"). Say it rather than making
            # the reader infer it.
            for unpacker in tile.tensix_coprocessor.getBackend().unpacker_units:
                waiting = unpacker.blocked_on()
                if waiting is None:
                    continue
                thread, opcode, which, bank = waiting
                tile_label = f"tile={coord} " if len(self.tile_cores) > 1 else ""
                lines.append(
                    f"  {tile_label}Unpacker {unpacker.unpacker_id} blocked on "
                    f"{opcode} from thread {thread}: waiting for {which} bank "
                    f"{bank} to be given back by the Matrix Unit (dvalid was "
                    f"set and never cleared)"
                )

        lines.append(
            "  (TT_SIM_DEADLOCK=0 to disable, "
            "TT_SIM_DEADLOCK_THRESHOLD=N to tune window)"
        )
        print("\n".join(lines), file=sys.stderr)
