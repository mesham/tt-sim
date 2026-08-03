"""Progress watchdog.

When the simulator has made no observable forward progress for a configured
number of cycles, a multi-line diagnostic is written to stderr naming the
likely cause (core PC frozen / oscillating, NoC requests outstanding, Tensix
backend wedged). The detector keeps running after each report and re-fires
once per window so the user is told the stall is still ongoing without
flooding output.

Dormant while every BabyRISCV core is in soft reset, since "all cores in
reset" is the normal pre-launch state, not a deadlock.

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

The window is measured in **simulated cycles**, not in pump ticks. The two
differ only when the Phase 4 pump strides (i.e. when every tile is dormant, in
which case there is by definition no progress to observe).

Configured via two env vars, read in :func:`deadlock_config_from_env`:

* ``TT_SIM_DEADLOCK`` — falsy disables the detector. Default on.
* ``TT_SIM_DEADLOCK_THRESHOLD`` — cycle count between reports. Default 50000.
"""

import os
import sys
from collections import deque

from tt_sim.pe.rv.babyriscv import BabyRISCV
from tt_sim.util.bits import get_nth_bit
from tt_sim.util.conversion import conv_to_uint32

DEFAULT_THRESHOLD = 50000
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


def _truthy(val):
    return val is not None and val.strip().lower() in {"1", "true", "yes", "on"}


def deadlock_config_from_env(env=None):
    """Return ``(enabled, threshold)`` from environment variables."""
    if env is None:
        env = os.environ
    enabled_raw = env.get("TT_SIM_DEADLOCK")
    enabled = True if enabled_raw is None else _truthy(enabled_raw)
    threshold_raw = env.get("TT_SIM_DEADLOCK_THRESHOLD")
    try:
        threshold = (
            int(threshold_raw) if threshold_raw is not None else DEFAULT_THRESHOLD
        )
    except ValueError:
        threshold = DEFAULT_THRESHOLD
    if threshold <= 0:
        threshold = DEFAULT_THRESHOLD
    return enabled, threshold


class DeadlockDetector:
    def __init__(self, threshold, enabled, tensix_tiles, dram_tiles):
        self.threshold = threshold
        self.enabled = enabled
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
        self.nuis.append((coord, 0, tile.get_noc_nui(0)))
        self.nuis.append((coord, 1, tile.get_noc_nui(1)))
        for core in cores:
            self.recent_pcs[(coord, core.core_type)] = deque(maxlen=_RECENT_PC_WINDOW)
        self._rebaseline(getattr(self, "stall_since", 0))

    def _rebaseline(self, cycle):
        """Forget the current window; nothing is stalled until re-observed."""
        self.last_signature = None
        self.stall_since = cycle
        self._confirm_left = None
        self._next_sample_cycle = cycle if self.enabled else float("inf")

    def _read_soft_reset(self, tile):
        return conv_to_uint32(tile.tile_ctrl.RISCV_DEBUG_REG_SOFT_RESET_0)

    def tick(self, cycle):
        # Hot path: one comparison per pumped cycle. ``_next_sample_cycle`` is
        # the whole gate — the sampling interval, the confirmation phase (which
        # sets it to the next cycle) and ``enabled`` all express themselves
        # through it, so nothing else is read on the common path.
        if cycle < self._next_sample_cycle:
            return
        self._sample(cycle)

    def _sample(self, cycle):
        if not self.enabled:
            self._next_sample_cycle = float("inf")
            return
        if cycle < self.stall_since:
            # The clock rewound (device reset); re-baseline rather than
            # measuring a negative window.
            self.stall_since = cycle

        active = []  # list of (coord, core, pc)
        for coord, tile, cores in self.tile_cores:
            reset_val = self._read_soft_reset(tile)
            for core in cores:
                bit = BabyRISCV.CORE_TYPE_TO_SOFT_RESET_BIT[core.core_type]
                if get_nth_bit(reset_val, bit) == 1:
                    self.recent_pcs[(coord, core.core_type)].clear()
                else:
                    pc = conv_to_uint32(core.register_file["pc"].read())
                    self.recent_pcs[(coord, core.core_type)].append(pc)
                    active.append((coord, core, pc))

        if not active:
            self._rebaseline(cycle)
            self._next_sample_cycle = cycle + self.sample_interval
            return

        pc_bucket_sig = tuple(
            (coord, core.core_type, pc // _PC_BUCKET_BYTES, core.unknown_instructions)
            for coord, core, pc in active
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

        signature = (pc_bucket_sig, nui_sig, tensix_sig)
        if signature != self.last_signature:
            # Progress. Back to sampled polling from here.
            self.last_signature = signature
            self.stall_since = cycle
            self._confirm_left = None
            self._next_sample_cycle = cycle + self.sample_interval
            return

        if self._confirm_left is not None:
            self._confirm_left -= 1
            if self._confirm_left > 0:
                self._next_sample_cycle = cycle + 1
                return
            self._confirm_left = None
            self._report(cycle, active)
            # Re-arm: one report per window, as before.
            self.stall_since = cycle
            self._next_sample_cycle = cycle + self.sample_interval
            return

        if cycle - self.stall_since >= self.threshold:
            # Sampled evidence of a stall; confirm it cycle-by-cycle before
            # printing anything.
            self._confirm_left = _CONFIRM_TICKS
            self._next_sample_cycle = cycle + 1
        else:
            self._next_sample_cycle = cycle + self.sample_interval

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

        lines.append(
            "  (TT_SIM_DEADLOCK=0 to disable, "
            "TT_SIM_DEADLOCK_THRESHOLD=N to tune window)"
        )
        print("\n".join(lines), file=sys.stderr)
