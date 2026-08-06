"""Firmware-loop recognition: park a baby core spinning in a pure poll loop.

The full-grid affordability fix (ROADMAP item 2, and the gap recorded in
``docs/upstream-examples-status.md``): tt-metal's grid-wide init releases BRISC
on **every** worker in the declared grid, and each BRISC then spins in the
firmware go-message poll until teardown. A spinning core never reports
:meth:`~tt_sim.device.clock.Clockable.is_clock_idle`, so Phase 1 tile dormancy
buys nothing and simulator wall clock is linear in *materialised* tiles
regardless of which tiles the program launches kernels on.

A core spinning on the go message is architecturally idle even though it
retires instructions. This module recognises that state and lets the core
answer ``next_wake_cycle -> None`` (dormant) so the whole tile can stride —
**without ever diverging from the unparked run by a single cycle**. The
observed loop on the real firmware (Blackhole ``nine`` replay, tt-metal 0.74,
captured before this was designed):

* BRISC go-wait: 18 instructions over two blocks (``0x3ec0..0x3ef4`` /
  ``0x42d0..0x42dc``), loads only L1 ``0x520`` (launch_msg_rd_ptr), ``0x4f3``
  (go signal byte), ``0x6c`` and ``0xff``; no stores.
* NCRISC idle: 8 instructions at ``0x5da8..0x5dc4``, loads only L1 ``0x68``.

The predicate — deliberately much stronger than "the PC revisits a window"
-------------------------------------------------------------------------

A core is parked only after a three-stage proof, per attempt:

1. **SEEK** — the current PC must be revisited within a small budget
   (the core is mid-loop; a straight-line kernel never comes back).
2. **RECORD** — one full anchor-to-anchor iteration is executed with every
   load and store routed through a recording proxy. The full register file
   (GPRs, PC, next-PC, and the FP file when present) is snapshotted at every
   tick. The iteration must contain **no stores at all** (which also excludes
   ``.ttinsn`` Tensix pushes — they are stores to ``0xFFE40000``) and **no
   loads outside plain RAM** (``AddressableMemory`` /
   ``SparseAddressableMemory`` leaves: L1, local data RAM, IRAM). MMIO is
   rejected wholesale, which is what keeps a wall-clock timeout loop, a
   mailbox pop, an NIU-counter poll or a PC-buffer wait from ever parking —
   their loaded values change (or have side effects) without any write.
   Registers at the second anchor arrival must equal the first's: the
   iteration is a **fixed point** — executing it changes nothing.
3. **VERIFY** — a second full iteration must reproduce the recorded
   trajectory *exactly*, tick for tick (PC and full register file), and end
   with the watched memory equal to the snapshot taken when VERIFY began.
   This closes the window in which a value changes mid-RECORD after its load.

Only then does the core park. The watch set is the union of every load range
and every fetched instruction word of the loop, merged into a handful of
(component, offset, size, snapshot) spans resolved once.

Cycle-exactness (why this cannot move a replay by even one poll)
----------------------------------------------------------------

* While the tile is **awake**, a parked core still executes normally on every
  tick — parked-ness only changes its ``next_wake_cycle`` answer. Awake
  behaviour is therefore byte-identical to an unparked run.
* Cycles are only skipped while the **whole tile** is dormant, and nothing on
  a dormant tile ticks, so nothing tile-internal can write memory. Every
  external stimulus (host read/write, ``NUI.transmit``, soft-reset write,
  reset) already wakes the tile — the Phase 1 wake set.
* The parked core's ``next_wake_cycle`` re-reads the watched spans every time
  it is consulted (i.e. after every executed tick) and refuses dormancy the
  moment any byte differs from the snapshot, so a stride can never begin, or
  survive, a change to anything the loop can observe.
* On the first tick after a gap the core's registers are **restored from the
  recorded trajectory** at the exact phase the unparked run would have
  reached: the loop is a verified fixed point over memory that provably did
  not change during the gap, so state at wake is bit-identical to having
  executed every skipped iteration.

Deliberate scope limits (each falls back to today's behaviour, never to a
wrong answer):

* Detection is skipped while ``TT_SIM_COST_MODEL`` is active — the RV cost
  scoreboard carries absolute cycle numbers, so trajectory restore would need
  a time-translation argument this first version does not attempt.
* Detection is skipped while the structured trace bus is enabled or the
  core's snoop diagnostics are on — a skipped cycle publishes no events, and
  a user watching instructions should see the spin.
* Loops longer than :data:`MAX_LOOP_TICKS` ticks per iteration are never
  parked (the observed firmware loops are 8–18).

Kill switch: ``TT_SIM_FIRMWARE_IDLE=0`` disables everything (default on,
following the ``TT_SIM_PUMP_STRIDE`` precedent). ``TT_SIM_FIRMWARE_IDLE_DEBUG``
prints one line per park/unpark decision to stderr.
"""

import os
import sys

from tt_sim.memory.memory import (
    AddressableMemory,
    MemorySpace,
    SparseAddressableMemory,
)

# Detection state machine values (ints for cheap hot-path comparison).
SPIN_IDLE = 0
SPIN_SEEK = 1
SPIN_RECORD = 2
SPIN_VERIFY = 3
SPIN_PARKED = 4

#: Longest loop (in ticks per iteration) that will be considered for parking.
#: The observed firmware wait loops are 8-18 ticks; this is generous headroom
#: while still aborting quickly on straight-line kernel code.
MAX_LOOP_TICKS = 64
#: Ticks SEEK waits for the anchor PC to come round again.
SEEK_BUDGET = MAX_LOOP_TICKS + 16
#: Maximum merged watch spans; more than this means the loop is too weird.
MAX_WATCH_RANGES = 8
#: Ticks between detection attempts while running unparked.
BASE_ATTEMPT_INTERVAL = 100
#: Attempt-interval cap under exponential backoff (compute-heavy cores fail
#: detection quickly and repeatedly; backing off bounds the proxy overhead).
#: The cap is also the worst-case parking latency once a core enters its wait
#: loop — firmware init fails every attempt on the way in, so the core arrives
#: at the loop with the interval at the cap. 800 keeps that latency small
#: against the ~10k-cycle launches while a failed attempt stays cheap (SEEK is
#: one extra compare per tick and RECORD aborts on the first store).
MAX_ATTEMPT_INTERVAL = 800


def _truthy(raw, default):
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def firmware_idle_enabled_from_env(env=None):
    """Kill switch: ``TT_SIM_FIRMWARE_IDLE`` (default on)."""
    if env is None:
        env = os.environ
    return _truthy(env.get("TT_SIM_FIRMWARE_IDLE"), True)


def firmware_idle_debug_from_env(env=None):
    if env is None:
        env = os.environ
    return _truthy(env.get("TT_SIM_FIRMWARE_IDLE_DEBUG"), False)


class _ObservedMemory:
    """Recording proxy handed to the ISA executors during RECORD/VERIFY.

    Reads are noted (their addresses are classified when the watch set is
    built, and anything that is not plain RAM aborts the attempt); writes
    abort the attempt outright. Both delegate to the real memory, so observed
    execution is architecturally identical.
    """

    __slots__ = ("_spin", "_real")

    def __init__(self, spin, real):
        self._spin = spin
        self._real = real

    def read(self, addr, size):
        self._spin._note_load(addr, size)
        return self._real.read(addr, size)

    def write(self, addr, value, size=None):
        self._spin._abort_violation("store during loop")
        return self._real.write(addr, value, size)

    def __getattr__(self, name):
        return getattr(self._real, name)


class FirmwareSpin:
    """Per-core spin recognition state. Owned by :class:`BabyRISCV`."""

    __slots__ = (
        "state",
        "next_attempt",
        "interval",
        "last_bucket",
        "debug",
        "anchor_pc",
        "budget",
        "traj",
        "loads",
        "load_seen",
        "violation",
        "verify_i",
        "watch",
        "loop_len",
        "phase",
        "last_cycle",
        "unknown_at_start",
        "parks",
        "skipped_cycles",
    )

    def __init__(self, debug=False):
        self.state = SPIN_IDLE
        self.next_attempt = BASE_ATTEMPT_INTERVAL
        self.interval = BASE_ATTEMPT_INTERVAL
        #: 128-byte PC bucket seen at the last checkpoint. An attempt is only
        #: made when two successive checkpoints land in the same bucket — a
        #: core marching through init code never matches, so it pays nothing
        #: beyond the checkpoint itself, and the backoff resets whenever the
        #: core moves to new code so a wait loop entered *after* a long init
        #: is recognised promptly rather than at the backed-off cadence.
        self.last_bucket = -1
        self.debug = debug
        self.anchor_pc = None
        self.budget = 0
        #: Recorded trajectory: one ``(pc_bytes, regs_tuple)`` per tick of the
        #: iteration, where ``regs_tuple`` is every register's raw bytes bar
        #: x0 (GPRs 1-31, PC, next-PC, and the FP file when allocated). Entry
        #: ``i`` is the state at the *start* of tick ``i``.
        self.traj = []
        self.loads = []
        self.load_seen = set()
        self.violation = None
        self.verify_i = 0
        #: Parked watch set: tuple of ``(component, offset, size, snapshot)``.
        self.watch = ()
        self.loop_len = 0
        self.phase = 0
        self.last_cycle = 0
        self.unknown_at_start = 0
        #: Times this core has been parked (diagnostics).
        self.parks = 0
        #: Cycles this core did not execute because the pump skipped them
        #: while it was parked (diagnostics).
        self.skipped_cycles = 0

    # -- detection ---------------------------------------------------------

    def checkpoint(self, core, cycle):
        """The IDLE-state trigger, reached every :attr:`interval` ticks.

        Begins a detection attempt only when the PC is in the same 128-byte
        bucket as at the previous checkpoint — i.e. the core has plausibly
        settled into a small loop. Otherwise it notes the new bucket and
        resets the backoff, so a wait loop entered after a long stretch of
        straight-line init is attempted at the base cadence.
        """
        bucket = int.from_bytes(core.pc_register.value, "little") >> 7
        if bucket != self.last_bucket:
            self.last_bucket = bucket
            self.interval = BASE_ATTEMPT_INTERVAL
            self.next_attempt = cycle + BASE_ATTEMPT_INTERVAL
            return
        self.begin_seek(core, cycle)

    def begin_seek(self, core, cycle):
        """Start a detection attempt. Called from the IDLE hot path."""
        if core.snoop or core.bus.enabled or core.rv_cost is not None:
            # See the module docstring: diagnostics want to see the spin, and
            # the cost model's absolute-cycle scoreboard is out of scope.
            self.next_attempt = cycle + self.interval
            return
        if not core.active:
            # A core out of soft reset but never start()ed executes nothing —
            # its frozen state would satisfy every check as a 1-tick "loop".
            # There is no loop to recognise; leave the tile awake as today.
            self.next_attempt = cycle + self.interval
            return
        self.state = SPIN_SEEK
        self.anchor_pc = None
        self.budget = SEEK_BUDGET

    def tick(self, core, cycle):
        """Drive one clock tick for any non-IDLE state.

        Always executes the core's real tick exactly once; the state machine
        only observes (SEEK/RECORD/VERIFY) or restores-then-executes (PARKED).
        """
        state = self.state
        if state == SPIN_PARKED:
            self._parked_tick(core, cycle)
            return
        if not core.active:
            # Stopped mid-observation (a driver called stop() directly):
            # nothing is executing, so there is nothing to observe.
            self._abort(core, cycle)
            core._base_tick(cycle)
            return
        pc_bytes = core.pc_register.value
        if state == SPIN_SEEK:
            anchor = self.anchor_pc
            if anchor is None:
                self.anchor_pc = pc_bytes
            elif pc_bytes == anchor:
                # Anchor came round: record one full iteration through the
                # proxy, starting with this tick.
                self.state = SPIN_RECORD
                self.traj = [(pc_bytes, self._regs_snapshot(core))]
                self.loads = []
                self.load_seen = set()
                self.violation = None
                self.unknown_at_start = core.unknown_instructions
                self.budget = MAX_LOOP_TICKS
                core._exec_memory = _ObservedMemory(self, core.visible_memory)
                core._base_tick(cycle)
                if self.violation is not None:
                    self._abort(core, cycle)
                return
            core._base_tick(cycle)
            self.budget -= 1
            if self.budget <= 0 and self.state == SPIN_SEEK:
                self._abort(core, cycle)
            return
        if state == SPIN_RECORD:
            if pc_bytes == self.anchor_pc:
                # Second anchor arrival: the iteration is complete. It is a
                # candidate only if it returned every register to its anchor
                # value — a fixed point.
                if self._regs_snapshot(core) != self.traj[0][1]:
                    self._abort(core, cycle)
                    core._base_tick(cycle)
                    return
                self.loop_len = len(self.traj)
                self.state = SPIN_VERIFY
                self.verify_i = 0
                # VERIFY re-runs the whole iteration against the snapshot
                # taken now, so a value that changed mid-RECORD (after its
                # load) fails verification rather than being snapshotted
                # stale. Build the watch first; it can abort.
                if not self._build_watch(core):
                    self._abort(core, cycle)
                    core._base_tick(cycle)
                    return
                self._verify_tick(core, cycle)
                return
            self.traj.append((pc_bytes, self._regs_snapshot(core)))
            core._base_tick(cycle)
            self.budget -= 1
            if self.violation is not None or (
                self.budget <= 0 and self.state == SPIN_RECORD
            ):
                self._abort(core, cycle)
            return
        # SPIN_VERIFY
        self._verify_tick(core, cycle)

    def _verify_tick(self, core, cycle):
        i = self.verify_i
        pc_bytes = core.pc_register.value
        entry = self.traj[i]
        if pc_bytes != entry[0] or self._regs_snapshot(core) != entry[1]:
            self._abort(core, cycle)
            core._base_tick(cycle)
            return
        core._base_tick(cycle)
        if self.violation is not None or core.unknown_instructions != (
            self.unknown_at_start
        ):
            self._abort(core, cycle)
            return
        i += 1
        if i < self.loop_len:
            self.verify_i = i
            return
        # Full second iteration reproduced. The watched memory must still
        # equal the snapshot VERIFY started from, or a write raced the pass.
        for component, offset, size, snapshot in self.watch:
            if component.read(offset, size) != snapshot:
                self._abort(core, cycle)
                return
        self._park(core, cycle)

    def _park(self, core, cycle):
        core._exec_memory = core.visible_memory
        self.state = SPIN_PARKED
        self.parks += 1
        # The next tick (this one just executed trajectory index 0's
        # instruction... no: _verify_tick executed index loop_len-1 last, so
        # the core is back at the anchor and the *next* tick is phase 0.
        self.phase = 0
        self.last_cycle = cycle
        self.interval = BASE_ATTEMPT_INTERVAL
        if self.debug:
            lo = min(int.from_bytes(p, "little") for p, _ in self.traj)
            hi = max(int.from_bytes(p, "little") for p, _ in self.traj)
            spans = ", ".join(
                f"{type(c).__name__}+{off:#x}x{size}" for c, off, size, _ in self.watch
            )
            print(
                f"[firmware-idle] {core.core_label} parked at cycle {cycle}: "
                f"pure poll loop of {self.loop_len} ticks in "
                f"[{lo:#x}, {hi:#x}], watching {spans}",
                file=sys.stderr,
            )
        core.spin_parked = True

    def _parked_tick(self, core, cycle):
        traj = self.traj
        loop_len = self.loop_len
        gap = cycle - self.last_cycle
        phase = self.phase
        if gap > 1:
            # The pump skipped cycles while the whole tile was dormant.
            # Memory provably did not change (nothing on the tile ticked and
            # every external write wakes it before landing), so the unparked
            # run would have executed ``gap - 1`` more ticks of the fixed
            # point — restore the exact state it would have reached.
            steps = gap - 1
            self.skipped_cycles += steps
            phase = (phase + steps) % loop_len
            self.phase = phase
            registers = core.register_file.registers
            snapshot = traj[phase][1]
            for i in range(1, len(registers)):
                registers[i].value = snapshot[i - 1]
        if core.pc_register.value != traj[phase][0]:
            # The loop exited (a watched value changed and the previous tick
            # branched out). Back to normal execution.
            self._unpark(core, cycle, "left the recorded loop")
            core._base_tick(cycle)
            return
        core._base_tick(cycle)
        self.phase = (phase + 1) % loop_len
        self.last_cycle = cycle

    def wake_check(self, core, cycle):
        """The parked core's ``next_wake_cycle``: dormant only while every
        watched byte still equals its snapshot.

        Consulted after every executed tick (it is the tile's dormancy
        probe), so a stride can neither begin nor survive a change to
        anything the loop can observe.
        """
        for component, offset, size, snapshot in self.watch:
            if component.read(offset, size) != snapshot:
                self._unpark(core, cycle, "watched value changed")
                return cycle + 1
        return None

    def on_reset(self, core):
        """Reset/soft-reset/restart: drop everything, back to square one."""
        core._exec_memory = core.visible_memory
        core.spin_parked = False
        self.state = SPIN_IDLE
        self.traj = []
        self.watch = ()
        self.interval = BASE_ATTEMPT_INTERVAL
        self.next_attempt = BASE_ATTEMPT_INTERVAL
        self.last_bucket = -1
        self.violation = None

    # -- internals ---------------------------------------------------------

    def _regs_snapshot(self, core):
        registers = core.register_file.registers
        return tuple(r.value for r in registers[1:])

    def _note_load(self, addr, size):
        if self.violation is not None:
            return
        key = (addr, size)
        if key in self.load_seen:
            return
        self.load_seen.add(key)
        self.loads.append(key)

    def _abort_violation(self, reason):
        if self.violation is None:
            self.violation = reason

    def _abort(self, core, cycle):
        core._exec_memory = core.visible_memory
        self.state = SPIN_IDLE
        self.traj = []
        self.loads = []
        self.load_seen = set()
        self.violation = None
        # Exponential backoff: a core that fails detection is doing real work
        # and should not keep paying for the recording proxy.
        self.interval = min(self.interval * 2, MAX_ATTEMPT_INTERVAL)
        self.next_attempt = cycle + self.interval

    def _unpark(self, core, cycle, reason):
        core.spin_parked = False
        self.state = SPIN_IDLE
        self.next_attempt = cycle + self.interval
        # The core is leaving the loop for new code; make the next checkpoint
        # a plain bucket note rather than an immediate attempt.
        self.last_bucket = -1
        if self.debug:
            print(
                f"[firmware-idle] {core.core_label} unparked at cycle {cycle} "
                f"({reason}); {self.skipped_cycles} spin cycles skipped so far",
                file=sys.stderr,
            )

    def _build_watch(self, core):
        """Resolve loads + fetched code into (component, offset, size,
        snapshot) spans. False if anything is not plain RAM."""
        intervals = []
        for addr, size in self.loads:
            intervals.append((addr, addr + size))
        for pc_bytes, _ in self.traj:
            pc = int.from_bytes(pc_bytes, "little")
            intervals.append((pc, pc + 4))
        intervals.sort()
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                if end > merged[-1][1]:
                    merged[-1][1] = end
            else:
                merged.append([start, end])
        if len(merged) > MAX_WATCH_RANGES:
            return False
        watch = []
        for start, end in merged:
            resolved = _resolve_plain_ram(core.visible_memory, start, end - start)
            if resolved is None:
                return False
            component, offset = resolved
            size = end - start
            watch.append((component, offset, size, component.read(offset, size)))
        self.watch = tuple(watch)
        return True


def _resolve_plain_ram(memory, addr, size):
    """Resolve ``addr`` to a plain-RAM leaf ``(component, offset)``.

    ``None`` for anything else — MMIO components, unmapped addresses, spans
    that cross a mapping boundary. Follows at most a few nested
    :class:`MemorySpace` levels (merged maps normally hold leaves directly).
    """
    for _ in range(4):
        memory_map = getattr(memory, "memory_map", None)
        if memory_map is None:
            return None
        addr_range, target = memory_map.locate(addr)
        if addr_range is None or target is None:
            return None
        if addr + size - 1 > addr_range.high:
            return None
        if isinstance(target, (AddressableMemory, SparseAddressableMemory)):
            offset = addr - addr_range.low
            if offset + size > target.getSize():
                return None
            return target, offset
        if isinstance(target, MemorySpace):
            addr = addr - addr_range.low
            memory = target
            continue
        return None
    return None
