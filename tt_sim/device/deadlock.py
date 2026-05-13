"""Per-cycle progress watchdog.

When the simulator has made no observable forward progress for a configured
number of cycles, a multi-line diagnostic is written to stderr naming the
likely cause (core PC frozen / oscillating, NoC requests outstanding, Tensix
backend wedged). The detector keeps running after each report and re-fires
once per window so the user is told the stall is still ongoing without
flooding output.

Dormant while every BabyRISCV core is in soft reset, since "all cores in
reset" is the normal pre-launch state, not a deadlock.

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

        self.last_signature = None
        self.stalled_for = 0

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
        self.last_signature = None
        self.stalled_for = 0

    def _read_soft_reset(self, tile):
        return conv_to_uint32(tile.tile_ctrl.RISCV_DEBUG_REG_SOFT_RESET_0)

    def tick(self, cycle):
        if not self.enabled:
            return

        active = []  # list of (coord, core)
        for coord, tile, cores in self.tile_cores:
            reset_val = self._read_soft_reset(tile)
            for core in cores:
                bit = BabyRISCV.CORE_TYPE_TO_SOFT_RESET_BIT[core.core_type]
                if get_nth_bit(reset_val, bit) == 1:
                    self.recent_pcs[(coord, core.core_type)].clear()
                else:
                    pc = conv_to_uint32(core.register_file["pc"].read())
                    self.recent_pcs[(coord, core.core_type)].append(pc)
                    active.append((coord, core))

        if not active:
            self.stalled_for = 0
            self.last_signature = None
            return

        pc_bucket_sig = tuple(
            (
                coord,
                core.core_type,
                conv_to_uint32(core.register_file["pc"].read()) // _PC_BUCKET_BYTES,
                core.unknown_instructions,
            )
            for coord, core in active
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
        if signature == self.last_signature:
            self.stalled_for += 1
            if self.stalled_for >= self.threshold:
                self._report(cycle, active)
                self.stalled_for = 0
        else:
            self.stalled_for = 0
            self.last_signature = signature

    def _report(self, cycle, active):
        lines = [
            f"[DEADLOCK cycle={cycle}] no observable progress "
            f"for {self.threshold} cycles"
        ]

        for coord, core in active:
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
