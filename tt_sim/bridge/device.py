"""Architecture-agnostic device wrapper — cycle pumping + reset tracking.

Cycle pumping rule (locked-in plan decision): after every read/write through
this Device, ``tt_device.run(cycles_per_poll)`` is invoked iff at least one
BRISC is out of reset. This mirrors how a tt-metal host drives the device: it
writes the go signal, then polls the go-message mailbox, pumping cycles between
polls until the firmware flips it to ``RUN_MSG_DONE``.

The underlying tt-sim device (Wormhole / Blackhole) and its coord map are
injected by the driver, so nothing here is architecture-specific.
"""

import os

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
        self.tt_device.write(unified, addr, data)
        self._maybe_pump()

    def read(self, unified, addr, size):
        result = bytes(self.tt_device.read(unified, addr, size))
        self._maybe_pump()
        return result

    def assert_reset(self, unified):
        self._brisc_running[unified] = False
        self.tt_device.assert_soft_reset(unified)

    def deassert_reset(self, unified):
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
        self._maybe_pump()

    def _maybe_pump(self):
        if any(self._brisc_running.values()):
            self.tt_device.run(self.cycles_per_poll)
