"""Architecture-agnostic device wrapper — cycle pumping + reset tracking.

Cycle pumping rule (locked-in plan decision): after every read/write through
this Device, ``tt_device.run(cycles_per_poll)`` is invoked iff at least one
BRISC is out of reset. This mirrors how a tt-metal host drives the device: it
writes the go signal, then polls the go-message mailbox, pumping cycles between
polls until the firmware flips it to ``RUN_MSG_DONE``.

That rule has one documented hole, and :meth:`Device.settle_profiler_flush`
closes it — see its docstring: the device profiler's readback is the one host
transaction whose answer is still being *written* at the moment the host asks
for it.

The underlying tt-sim device (Wormhole / Blackhole) and its coord map are
injected by the driver, so nothing here is architecture-specific.
"""

import os
import sys

from tt_sim.device.tt_device import DeviceTileDiagnostics
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.pe.tensix.util import TensixCoprocessorDiagnostics
from tt_sim.util.conversion import conv_to_uint32

# Map from individual env var → (group, field). Group "rv"/"noc" fields land
# on DeviceTileDiagnostics; group "co" fields land on TensixCoprocessorDiagnostics.
_DIAG_VARS = {
    "TT_SIM_DIAG_BRISC": ("rv", "brisc_diagnostics"),
    "TT_SIM_DIAG_NCRISC": ("rv", "ncrisc_diagnostics"),
    "TT_SIM_DIAG_TRISC0": ("rv", "trisc0_diagnostics"),
    "TT_SIM_DIAG_TRISC1": ("rv", "trisc1_diagnostics"),
    "TT_SIM_DIAG_TRISC2": ("rv", "trisc2_diagnostics"),
    "TT_SIM_DIAG_NOC0": ("noc", "noc0_diagnostics"),
    "TT_SIM_DIAG_NOC1": ("noc", "noc1_diagnostics"),
    "TT_SIM_DIAG_CO_ISSUED": ("co", "issued_instructions"),
    "TT_SIM_DIAG_CO_CONFIG": ("co", "configurations_set"),
    "TT_SIM_DIAG_CO_UNPACK": ("co", "unpacking"),
    "TT_SIM_DIAG_CO_PACK": ("co", "packing"),
    "TT_SIM_DIAG_CO_FPU": ("co", "fpu_calculations"),
    "TT_SIM_DIAG_CO_SFPU": ("co", "sfpu_calculations"),
    "TT_SIM_DIAG_CO_THCON": ("co", "thcon"),
}


def _truthy(val):
    return val is not None and val.strip().lower() in {"1", "true", "yes", "on"}


def diagnostics_from_env(env=None):
    """Build a DeviceTileDiagnostics from TT_SIM_DIAG_* env vars.

    Aggregates (TT_SIM_DIAG_ALL, _TRISC, _NOC, _CO) expand to their member
    flags first; an individual var present in env (even set to a falsy value)
    overrides whatever the aggregates produced. This lets users say
    ``TT_SIM_DIAG_ALL=1 TT_SIM_DIAG_NCRISC=0`` to mean "everything except NCRISC".
    """
    if env is None:
        env = os.environ

    flags = {name: False for name in _DIAG_VARS}

    if _truthy(env.get("TT_SIM_DIAG_ALL")):
        for name in flags:
            flags[name] = True
    if _truthy(env.get("TT_SIM_DIAG_TRISC")):
        for name in ("TT_SIM_DIAG_TRISC0", "TT_SIM_DIAG_TRISC1", "TT_SIM_DIAG_TRISC2"):
            flags[name] = True
    if _truthy(env.get("TT_SIM_DIAG_NOC")):
        for name in ("TT_SIM_DIAG_NOC0", "TT_SIM_DIAG_NOC1"):
            flags[name] = True
    if _truthy(env.get("TT_SIM_DIAG_CO")):
        for name, (group, _) in _DIAG_VARS.items():
            if group == "co":
                flags[name] = True

    for name in flags:
        if name in env:
            flags[name] = _truthy(env.get(name))

    rv_kwargs = {}
    co_kwargs = {}
    for name, (group, field) in _DIAG_VARS.items():
        if group == "co":
            co_kwargs[field] = flags[name]
        else:
            rv_kwargs[field] = flags[name]

    return DeviceTileDiagnostics(
        coprocessor_diagnostics=TensixCoprocessorDiagnostics(**co_kwargs),
        **rv_kwargs,
    )


def enabled_diagnostic_names(diagnostics):
    """Return a list of short names for diagnostics that are enabled (for logging)."""
    short = {
        "TT_SIM_DIAG_BRISC": "BRISC",
        "TT_SIM_DIAG_NCRISC": "NCRISC",
        "TT_SIM_DIAG_TRISC0": "TRISC0",
        "TT_SIM_DIAG_TRISC1": "TRISC1",
        "TT_SIM_DIAG_TRISC2": "TRISC2",
        "TT_SIM_DIAG_NOC0": "NOC0",
        "TT_SIM_DIAG_NOC1": "NOC1",
        "TT_SIM_DIAG_CO_ISSUED": "CO_ISSUED",
        "TT_SIM_DIAG_CO_CONFIG": "CO_CONFIG",
        "TT_SIM_DIAG_CO_UNPACK": "CO_UNPACK",
        "TT_SIM_DIAG_CO_PACK": "CO_PACK",
        "TT_SIM_DIAG_CO_FPU": "CO_FPU",
        "TT_SIM_DIAG_CO_SFPU": "CO_SFPU",
        "TT_SIM_DIAG_CO_THCON": "CO_THCON",
    }
    co = diagnostics.getTensixCoprocessorDiagnostics()
    on = []
    for name, (group, field) in _DIAG_VARS.items():
        target = co if group == "co" else diagnostics
        if getattr(target, field):
            on.append(short[name])
    return on


def link_contention_summary(device):
    """One line of ``NocLinkRegistry`` counters, or ``""`` when nothing claimed.

    A pure diagnostic — it reads three counters and computes no cycle. It exists
    because ``waits == 0`` is the only thing that distinguishes "this workload
    has no link contention" from "the link term is dead code", and until now
    that distinction needed a one-off instrumentation patch to see. Printing it
    at shutdown makes every simulator run say which of the two it was.

    ``device`` is the :class:`Device` wrapper or a bare tt-sim device; ``None``
    and a device built before the registries existed both answer ``""``, as does
    a run with the cost model off, where every counter stays at zero.
    """
    tt_device = getattr(device, "tt_device", device)
    registries = getattr(tt_device, "noc_link_registries", None)
    if not registries:
        return ""
    claims = sum(r.claims for r in registries)
    if not claims:
        return ""
    waits = sum(r.waits for r in registries)
    waited = sum(r.cycles_waited for r in registries)
    parts = [
        f"noc{i}: {r.claims} claims, {r.waits} waits, {r.cycles_waited} cycles"
        for i, r in enumerate(registries)
    ]
    return (
        f"link contention: {claims} claims, {waits} waits, "
        f"{waited} cycles waited ({'; '.join(parts)})"
    )


def profiler_flush_summary(device):
    """One line about the device-profiler readback, or ``""`` when it never ran.

    Says how many launches' profiler flushes the bridge had to wait for and
    what that cost in simulated cycles, so a profiled run states the size of
    the perturbation instead of leaving it to be inferred. A run with the
    profiler off never arms the mechanism and answers ``""``.
    """
    if not isinstance(device, Device):
        return ""
    if not (device.profiler_flush_settles or device.profiler_flush_timeouts):
        return ""
    line = (
        f"profiler flush: {device.profiler_flush_settles} settles, "
        f"{device.profiler_flush_cycles} extra cycles"
    )
    if device.profiler_flush_timeouts:
        line += f", {device.profiler_flush_timeouts} TIMED OUT"
    return line


class Device:
    """Wire-bridge wrapper around a tt-sim device (cycle pump + reset tracking).

    Architecture-agnostic: ``device_factory`` builds the underlying tt-sim
    device (``driver/wormhole`` passes a Wormhole factory, ``driver/blackhole``
    a Blackhole one), and ``tensix_coord_map`` maps a translated worker coord to
    the tile-directory coord the device is keyed by. The device only needs the
    common ``TT_Device`` surface (read/write, soft reset, ``add_tensix_tile``,
    ``run``, ``shutdown``), so the same wrapper drives either architecture.
    """

    # A wire DEASSERT deasserts the master BRISC plus whichever subordinate
    # RISCs the program enabled. tt-metal's launch message carries a bitmask
    # (``kernel_config_msg_t.enables``) where bit i selects processor i in this
    # order (tt-metal ``program.cpp``: ``enables |= 1 << processor_index``).
    _ENABLE_BIT_TO_CORE = (
        BabyRISCVCoreType.BRISC,
        BabyRISCVCoreType.NCRISC,
        BabyRISCVCoreType.TRISC0,
        BabyRISCVCoreType.TRISC1,
        BabyRISCVCoreType.TRISC2,
    )

    #: ``go_msg_t`` is a 4-byte union whose ``signal`` is the top byte;
    #: ``RUN_MSG_GO`` is a kernel launch (``hostdev/dev_msgs.h``). Same test the
    #: fabric's cores use — a launch is the only run-state that arms a
    #: profiler flush, because it is the only one that produces markers.
    _GO_MSG_SIZE = 4
    _RUN_MSG_GO = 0x80

    def __init__(
        self,
        device_factory,
        tensix_coord_map,
        *,
        cycles_per_poll: int = 100,
        diagnostics=None,
        launch_enables_offset=None,
    ):
        self.tt_device = device_factory(diagnostics or DeviceTileDiagnostics())
        self.tensix_coord_map = tensix_coord_map
        self.cycles_per_poll = cycles_per_poll
        # L1 offset of the launch message's ``enables`` bitmask, or None to keep
        # the legacy BRISC-only deassert (Wormhole, whose captured flows are all
        # single-RISC). Blackhole injects it so NCRISC/TRISC kernels run.
        self.launch_enables_offset = launch_enables_offset
        # unified_coord -> True if BRISC has been deasserted.
        self._brisc_running: dict[tuple[int, int], bool] = {}
        # Device-profiler readback state (see settle_profiler_flush). All three
        # stay empty for the whole of a run with the profiler off, which is what
        # makes the mechanism structurally inert there rather than merely quiet.
        self._profiler_ctrl_addr: dict[tuple[int, int], int] = {}
        self._profiler_flush_pending: set[tuple[int, int]] = set()
        self._profiler_flush_failed: set[tuple[int, int]] = set()
        #: Diagnostics for ``profiler_flush_summary``.
        self.profiler_flush_settles = 0
        self.profiler_flush_cycles = 0
        self.profiler_flush_timeouts = 0

    def ensure_tensix_tile(self, translated):
        """Lazily materialise the TensixTile addressed by a translated coord.

        Called on first access to a Tensix worker coord that isn't yet
        backed by a tt-sim tile. Builds the tile through
        ``add_tensix_tile`` (which registers it in both NoC directories under
        its canonical SoC-physical NoC 0 coord) and registers it for
        BRISC-reset tracking. Idempotent — returns the unified coord on repeat
        calls.
        """
        unified = self.tensix_coord_map[translated]
        if unified not in self.tt_device.tile_directory:
            self.tt_device.add_tensix_tile(unified)
        self.register_tensix(unified)
        return unified

    def register_tensix(self, unified):
        self._brisc_running.setdefault(unified, False)

    def write(self, unified, addr, data):
        self._note_profiler_write(unified, addr, data)
        self.tt_device.write(unified, addr, data)
        self._maybe_pump()

    def read(self, unified, addr, size):
        if (
            size == Device._PROFILER_CTRL_BYTES
            and unified in self._profiler_flush_pending
            and self._profiler_ctrl_addr.get(unified) == addr
        ):
            self.settle_profiler_flush(unified, addr)
        result = bytes(self.tt_device.read(unified, addr, size))
        self._maybe_pump()
        return result

    def assert_reset(self, unified):
        self._brisc_running[unified] = False
        self.tt_device.assert_soft_reset(unified)

    def deassert_reset(self, unified):
        self.deassert_reset_without_pump(unified)
        self._maybe_pump()

    def deassert_reset_without_pump(self, unified):
        # A wire DEASSERT brings the master BRISC plus every subordinate the
        # launch message enabled out of soft reset. BRISC is always released:
        # it is the dispatch master and also runs during the grid-wide init
        # handshake (before any launch message is written). NCRISC/TRISC are
        # released only when their ``enables`` bit is set, so a BRISC-only
        # program doesn't run their (idle) base firmware, which would otherwise
        # perturb BRISC's go-message completion.
        cores = {BabyRISCVCoreType.BRISC}
        if self.launch_enables_offset is not None:
            enables = conv_to_uint32(
                self.tt_device.read(unified, self.launch_enables_offset, 4)
            )
            for bit, core in enumerate(self._ENABLE_BIT_TO_CORE):
                if enables & (1 << bit):
                    cores.add(core)
        for core in cores:
            self.tt_device.deassert_soft_reset(unified, core)
        # Clear stale CPU state before the deasserted cores run. Scope this to
        # the one tile: the global ``reset()`` clobbers the PCs of every other
        # tile already running firmware, which corrupts the whole grid once
        # more than one worker is materialised.
        self.tt_device.reset_tile(unified)
        self._brisc_running[unified] = True

    #: Go-message polling budget when catching a late-materialised worker up on
    #: an init handshake it slept through: cycles per chunk, and the cap. The
    #: cap is a safety valve, not a deadline — if the firmware has not answered
    #: by then the run is already broken and the watchdog will say so.
    _SETTLE_CHUNK = 100
    _SETTLE_CAP = 200_000

    def settle_go_message(self, unified, addr):
        """Run cycles until the go-message at ``addr`` reads ``RUN_MSG_DONE``.

        This is the host's own ``wait_until_cores_done`` loop, performed on the
        host's behalf for a worker that did not exist when the host ran it. The
        host was answered ``DONE`` out of the deferred core's zero-fill and has
        long moved on; what still has to happen is the firmware *executing*
        that run-state, because the launch that follows depends on it.

        Returns the number of cycles spent, ``None`` if it never settled.
        """
        spent = 0
        while self.tt_device.read(unified, addr, 4)[3] != 0x00:
            if spent >= Device._SETTLE_CAP:
                return None
            self.tt_device.run(Device._SETTLE_CHUNK)
            spent += Device._SETTLE_CHUNK
        return spent

    def write_without_pump(self, unified, addr, data):
        """A host write applied to the device without running any cycles.

        Only the replay of a deferred worker's journal uses this (and its
        reset sibling): that replay can be triggered from *inside*
        ``tt_device.run`` — a peer's NoC packet resolving to a worker that
        does not exist yet — and pumping there would re-enter the clock while
        a tile is mid-cycle.
        """
        self._note_profiler_write(unified, addr, data)
        self.tt_device.write(unified, addr, data)

    # ------------------------------------------------------------------
    # The device profiler's readback
    # ------------------------------------------------------------------
    #
    # tt-metal's kernel profiler keeps a per-RISC marker buffer in L1 and a
    # 32-word control vector (``kernel_profiler::ControlBuffer``) beside it.
    # BRISC's ``finish_profiler()`` publishes the run: it stamps each RISC's
    # ``DEVICE_BUFFER_END_INDEX``, then for each RISC writes
    # ``HOST_BUFFER_END_INDEX`` and NoC-pushes that RISC's markers to DRAM,
    # then flushes and finally sets ``PROFILER_DONE``. **All of that happens
    # after BRISC has already written ``RUN_MSG_DONE`` to the go mailbox** —
    # in tt-metal 0.74, ``brisc.cc:575`` sets DONE inside the
    # ``DeviceZoneScopedMainN("BRISC-FW")`` block whose destructor is what
    # calls ``finish_profiler()``.
    #
    # The host stops driving the clock at DONE. Measured on Blackhole
    # (``perfbench/mechbench``, ``elw`` at 8 tiles, cost model on): the host's
    # last go-message poll and its first control-vector read are *adjacent*
    # wire messages, so BRISC gets exactly ``cycles_per_poll`` (100) cycles of
    # tail. It reaches ``risc_finished_profiling()`` — the L1 markers and the
    # ``DEVICE_BUFFER_END_INDEX`` words are all there — and no further, so the
    # host reads ``HOST_BUFFER_END_INDEX_BR_ER == 0 == ..._NC`` and
    # ``DeviceProfiler::readRiscProfilerResults`` returns early for *both* the
    # DRAM and the L1 source. The result is a ``profile_log_device.csv`` with
    # nothing but its header: no counter samples and no zone markers either.
    # Nothing is lost in transit; the answer is simply not written yet.
    #
    # The workaround was ``TT_SIM_CYCLES_PER_POLL=5000``, which paid the extra
    # cycles on *every* wire message of the run. Worse, it hid how narrow the
    # margin was: at 100 cycles with the cost model off the same run got
    # through exactly one iteration of the publish loop, so it produced a log
    # that looked fine while silently carrying only BRISC's markers.
    #
    # What is done instead: recognise the control vector when the host writes
    # it, arm on a launch, and — at the one read that consumes it — run cycles
    # until the firmware sets ``PROFILER_DONE``. Triggering on the *read*
    # rather than on the DONE edge is deliberate: it is the moment the value
    # is consumed, so a run that never reads a control vector never pumps a
    # cycle, and a run that does pumps only up to what that read needs.

    #: ``PROFILER_L1_CONTROL_VECTOR_SIZE`` (32) uint32s, and the two indices
    #: into ``kernel_profiler::ControlBuffer`` this needs.
    _PROFILER_CTRL_BYTES = 128
    _PROFILER_CTRL_WORDS = 32
    _PROFILER_DONE_WORD = 19
    _PROFILER_CORE_COUNT_PER_DRAM_WORD = 17
    #: Cycles per chunk while settling, and the cap. The cap is a safety valve
    #: for a program whose firmware never publishes (a crashed BRISC, a build
    #: with the profiler compiled out of the *firmware* but not the host): it
    #: bounds the wait, says so, and gives up on that worker for the rest of
    #: the run rather than paying again on every launch. Measured need on the
    #: mechbench case is under 5 000 cycles, so this is ~40x headroom.
    _PROFILER_FLUSH_CHUNK = 100
    _PROFILER_FLUSH_CAP = 200_000

    @staticmethod
    def _looks_like_profiler_control_vector(data):
        """Is ``data`` the control vector tt-metal writes before a launch?

        Fingerprinted on the *shape* of ``kernel_profiler::ControlBuffer``, not
        on an address, because the profiler's L1 offset is release-specific and
        arrives over the wire like the rest of the layout. Three disjoint bands
        must be zero — the five host and five device end indices plus
        ``FW_RESET_{H,L}`` (0-11), ``PROFILER_DONE`` and
        ``TRACE_REPLAY_STATUS`` (19-20), and the padding past the last enum
        member (26-31) — leaving only the 12-18 window (DRAM address, run
        counter, NoC x/y, flat id, cores-per-DRAM, dropped-zone mask) free.
        ``CORE_COUNT_PER_DRAM`` must additionally be non-zero: the firmware
        divides by it, so it never legitimately is, and requiring it stops an
        all-zero 128-byte payload from matching.

        Verified against the three control-vector writes a Blackhole
        ``mechbench`` run makes (two at device init, one between launches).
        If a future release reshapes the vector this stops matching, and the
        readback reverts to the pre-fix behaviour rather than misfiring.
        """
        if len(data) != Device._PROFILER_CTRL_BYTES:
            return False
        words = [
            conv_to_uint32(bytes(data[i * 4 : i * 4 + 4]))
            for i in range(Device._PROFILER_CTRL_WORDS)
        ]
        if any(words[0:12]) or any(words[19:21]) or any(words[26:32]):
            return False
        return words[Device._PROFILER_CORE_COUNT_PER_DRAM_WORD] != 0

    def _note_profiler_write(self, unified, addr, data):
        """Learn the control vector's address, and arm on a launch.

        Only Tensix coords are considered (``_brisc_running`` is the registry
        of those), and arming needs an address already learnt, so the ordering
        the host uses — control vector at device init, ``go=GO`` per launch —
        is the only one that reaches :meth:`settle_profiler_flush`.
        """
        if unified not in self._brisc_running:
            return
        if len(data) == Device._PROFILER_CTRL_BYTES:
            if Device._looks_like_profiler_control_vector(data):
                self._profiler_ctrl_addr[unified] = addr
            return
        if (
            len(data) == Device._GO_MSG_SIZE
            and data[Device._GO_MSG_SIZE - 1] == Device._RUN_MSG_GO
            and unified in self._profiler_ctrl_addr
        ):
            self._profiler_flush_pending.add(unified)

    def _profiler_done(self, unified, addr):
        word = self.tt_device.read(unified, addr + Device._PROFILER_DONE_WORD * 4, 4)
        return conv_to_uint32(bytes(word)) != 0

    def _profiler_writes_landed(self, unified):
        """Have the marker pushes reached DRAM, not merely left the worker?

        ``PROFILER_DONE`` is set one instruction after
        ``profiler_noc_async_flush_posted_write()``, which waits for the NIU's
        *sent* counter — so the last RISCs' payloads can still be in flight
        when the host starts reading the DRAM buffer, and measurably were:
        with the wait ending at ``PROFILER_DONE``, ``mechbench elw`` recovered
        BRISC, NCRISC and TRISC 0 but dropped TRISC 1 and TRISC 2, whose
        pushes are the last two the loop issues.

        Scoped to the worker and the DRAM tiles — where a profiler payload is
        sent from and lands — rather than the whole grid, so a peer still
        running a kernel cannot hold the wait open.
        """
        tiles = list(self.tt_device.dram_tiles)
        worker = self.tt_device.tile_directory.get(unified)
        if worker is not None:
            tiles.append(worker)
        for tile in tiles:
            if not (
                tile.noc0_router.is_clock_idle() and tile.noc1_router.is_clock_idle()
            ):
                return False
        return True

    def settle_profiler_flush(self, unified, addr):
        """Run cycles until the run is published *and* its pushes have landed.

        Called from :meth:`read` at the host's control-vector read, once per
        launch per worker. Returns the cycles spent (0 when the firmware had
        already finished), or ``None`` on giving up.
        """
        self._profiler_flush_pending.discard(unified)
        if unified in self._profiler_flush_failed:
            return None
        if not self._brisc_running.get(unified):
            # Nothing can advance: the pump is gated on BRISC being out of
            # reset, exactly as _maybe_pump is.
            return 0
        if self._profiler_done(unified, addr) and self._profiler_writes_landed(unified):
            return 0
        spent = 0
        published = False
        while spent < Device._PROFILER_FLUSH_CAP:
            self.tt_device.run(Device._PROFILER_FLUSH_CHUNK)
            spent += Device._PROFILER_FLUSH_CHUNK
            published = published or self._profiler_done(unified, addr)
            if published and self._profiler_writes_landed(unified):
                self.profiler_flush_settles += 1
                self.profiler_flush_cycles += spent
                return spent
        self._profiler_flush_failed.add(unified)
        self.profiler_flush_timeouts += 1
        self.profiler_flush_cycles += spent
        print(
            f"[server] WARNING: device-profiler flush on worker {unified} did "
            f"not publish (PROFILER_DONE still 0) after "
            f"{Device._PROFILER_FLUSH_CAP} cycles — the readback for this "
            f"worker will be short or empty, and no further launch on it will "
            f"be waited for.",
            file=sys.stderr,
            flush=True,
        )
        return None

    def _maybe_pump(self):
        if any(self._brisc_running.values()):
            self.tt_device.run(self.cycles_per_poll)
