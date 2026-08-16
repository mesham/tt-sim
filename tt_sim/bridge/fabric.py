"""Routes (core, address, ...) protocol operations to the right Core instance.

Unknown coordinates lazily allocate a ``NullCore`` — matches the observed
behaviour of the zero-stub against tt-metal init traffic for eth / pcie /
arc / router-only endpoints and for worker coords the user didn't list in
``TT_SIM_TENSIX_COORDS`` (see ``__main__.py``).
"""

import os
import sys
from collections.abc import Callable

from .cores import NullCore
from .hostlink import stop_host


class Fabric:
    def __init__(self):
        self.cores: dict[tuple[int, int], object] = {}
        # Wire coord -> internal coord, for a run with NoC coordinate
        # translation enabled: under translation the host addresses Wormhole
        # workers at (18..25, 18..27) and Blackhole DRAM at {17,18}x{12..23},
        # while everything downstream of here — the core registry, the worker
        # guards, TT_SIM_TENSIX_COORDS, LazyTensixPool — keeps naming tiles by
        # the SoC-physical coord the descriptor lists. Translating once, here,
        # is why translated mode needed no change in any of them. Empty (and
        # skipped) in the default untranslated mode.
        self.wire_alias: dict[tuple[int, int], tuple[int, int]] = {}
        # Wire coords that *prove* the host is in the other convention. See
        # ``install_convention_guard``: this is the whole defence against a
        # forgotten TT_METAL_MOCK_CLUSTER_DESC_PATH, which otherwise reads as a
        # plausible run with quietly wrong answers.
        self.foreign_coords: frozenset[tuple[int, int]] = frozenset()
        self.convention_callback: Callable[[tuple[int, int]], None] | None = None
        # Invoked the first time a NullCore-backed coord receives a write
        # targeting L1 above NullCore.USER_DATA_ADDR_THRESHOLD — i.e. past
        # the kernel firmware / init scratch region, which is a strong
        # signal the host is treating this coord as a real worker. The
        # callback is wired by ``__main__.py`` to warn about coords the
        # user forgot to list in TT_SIM_TENSIX_COORDS. Without this the
        # silent NullCore zero-fill becomes a "shards 1-N return zeros"
        # debugging adventure.
        self.unmapped_callback: Callable[[tuple[int, int]], None] | None = None
        # Invoked when a NullCore-backed coord receives a go=GO (kernel launch).
        # Wired by ``__main__.py`` to error out: a launch on an un-materialised
        # worker means the program needs more cores than tt-sim was started with.
        self.kernel_launch_callback: Callable[[tuple[int, int]], None] | None = None
        # Optional ``coord -> core | None`` consulted before the NullCore
        # fallback. ``LazyTensixPool`` installs one that answers with a
        # ``DeferredTensixCore`` for functional worker coords, so a worker the
        # simulator has not built yet still journals what the host says to it.
        # Anything the factory declines (and every caller that installs none)
        # falls through to NullCore exactly as before.
        self.core_factory: Callable[[tuple[int, int]], object | None] | None = None

    def register(self, coord, core):
        self.cores[coord] = core

    def _core(self, coord):
        wire = coord
        if self.wire_alias:
            coord = self.wire_alias.get(coord, coord)
        core = self.cores.get(coord)
        if core is None:
            # Tested against the coord *as it arrived*, never the aliased one:
            # in translated mode a legitimate translated coord aliases onto a
            # physical coord that is itself a wrong-convention coord, so
            # checking after aliasing accuses every correct message.
            # Only ever reached on a directory miss, and a wrong-convention
            # coord always is one: the convention we are in is exactly the set
            # of coords that got registered.
            if wire in self.foreign_coords and self.convention_callback is not None:
                self.convention_callback(wire)
            if self.core_factory is not None:
                core = self.core_factory(coord)
            if core is None:
                core = NullCore(
                    coord,
                    on_user_data_write=self.unmapped_callback,
                    on_kernel_launch=self.kernel_launch_callback,
                )
            self.cores[coord] = core
        return core

    def write(self, coord, addr, data):
        self._core(coord).write(addr, data)

    def read(self, coord, addr, size):
        return self._core(coord).read(addr, size)

    def assert_reset(self, coord):
        self._core(coord).assert_reset()

    def deassert_reset(self, coord):
        self._core(coord).deassert_reset()


def install_convention_guard(
    fabric,
    *,
    translated,
    wire_alias,
    foreign_coords,
    reason,
    descriptor_hint,
    wire_addr=None,
    host_stopper=stop_host,
    on_error=None,
):
    """Make a coordinate-convention mismatch a loud failure, not a wrong answer.

    NoC coordinate translation is decided by the tt-metal *host*, from the
    cluster descriptor ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` names, and the
    simulator's own mode is derived from the same variable (inherited through
    UMD's ``uv_spawn``). They therefore agree — except when they do not: a host
    program that forgets to export it emits untranslated coordinates at a
    simulator keyed for translated ones, and the reverse happens when a
    descriptor is exported but the simulator was started with
    ``TT_SIM_NOC_TRANSLATION=0``.

    Without a guard that failure is *quiet*, which is the specific shape this
    project has repeatedly lost time to: every wrong-convention coordinate is
    an unregistered coordinate, so it lands on a ``NullCore`` that zero-fills
    reads and swallows writes. The program then runs to completion and reports
    wrong numbers, or hangs waiting on a core that was never written to.

    ``foreign_coords`` is the discriminating set — coords that belong to the
    *other* convention and to no tile in this one. It fires on the first such
    coord, which in practice is within the first few messages on both
    architectures: on Wormhole the conventions differ for every worker, so
    firmware upload trips it; on Blackhole worker coords are identical in both
    conventions (a Blackhole core's translated coord *is* its NoC 0 coord) and
    the discriminator is DRAM and eth, so the first buffer write trips it.

    Fails the same way ``install_worker_guards``' launch guard does, and for
    the same measured reason: stop the tt-metal host first, because UMD blocks
    in ``recv_from_device`` with no timeout and exiting without it strands the
    host for ever.
    """
    fired = []

    def report(coord):
        if fired:
            return
        fired.append(coord)
        ours = "translated" if translated else "untranslated (SoC-physical)"
        theirs = "an untranslated (SoC-physical)" if translated else "a translated"
        lines = [
            "[server] ERROR: NoC coordinate-convention mismatch.",
            f"[server]   tt-sim is keyed for {ours} coordinates ({reason}),",
            f"[server]   but the host addressed {coord[0]}-{coord[1]}, which is "
            f"{theirs} coordinate.",
        ]
        if translated:
            lines.append(
                f"[server]   The host program is not exporting "
                f"TT_METAL_MOCK_CLUSTER_DESC_PATH={descriptor_hint} — export it "
                f"in the same shell that runs the tt-metal binary."
            )
        else:
            lines.append(
                "[server]   The host exported TT_METAL_MOCK_CLUSTER_DESC_PATH "
                "with noc_translation: true, but this server was started with "
                "translation off (TT_SIM_NOC_TRANSLATION=0). Unset it, or point "
                "the host at an untranslated descriptor."
            )
        lines.append(
            "[server]   Continuing would zero-fill every access to that "
            "coordinate and silently produce wrong results, so the run is being "
            "stopped."
        )
        print("\n".join(lines), file=sys.stderr, flush=True)
        if on_error is not None:
            on_error(coord)
            return
        if host_stopper("a coordinate-convention mismatch", addr=wire_addr):
            os._exit(1)

    fabric.wire_alias = dict(wire_alias)
    fabric.foreign_coords = frozenset(foreign_coords)
    fabric.convention_callback = report
    return fired


def install_worker_guards(
    fabric,
    tensix_pool,
    tensix_coord_map,
    *,
    wire_addr=None,
    host_stopper=stop_host,
    lazy=False,
):
    """Wire the two "worker isn't there" diagnostics onto ``fabric``.

    Both fire on wire traffic to a *functional worker* the simulator did not
    materialise — the failure mode that otherwise shows up as zeros in the
    output or a hang with no explanation:

    * host writes to it — warn once, naming the coord and the env var to add;
    * a kernel launch (``go=GO``) on it — a hard error, because the program
      needs more cores than tt-sim was started with and its peers will block
      forever on NoC traffic this core will never send.

    Arch-agnostic and shared by every server entry point on purpose: this used
    to be inline in ``driver/wormhole/server/__main__.py`` only, so a Blackhole
    run silently swallowed exactly the traffic Wormhole shouted about.

    **The launch guard ends the run, not just the server.** It used to print and
    ``os._exit(1)``; measured, that was itself the hang the user then reported —
    UMD blocks in ``recv_from_device`` with no timeout, so killing the simulator
    strands the host for ever (120 s and counting on ``examples/one`` with one
    mistyped coord, versus a 7 s self-diagnosed failure when the server stayed
    up). The exit is now conditional on having *first* stopped the tt-metal host
    (:mod:`tt_sim.bridge.hostlink`). If no host can be identified we keep
    serving instead: the run then reaches its own end with meaningless results,
    which is a worse answer than stopping but a much better one than never
    answering at all. Either way the ERROR naming the coord is already on
    screen.

    ``wire_addr`` is the address the server dialled, used to identify the host;
    it defaults to ``$NNG_SOCKET_ADDR``. ``host_stopper`` is injectable so tests
    can drive both outcomes.

    **``lazy=True``** says a :class:`~tt_sim.bridge.materialise.LazyTensixPool`
    is installed, so every functional worker can be built on demand and neither
    guard has anything to say about one: the warning and the hard error both
    described a *pinned* pool that the program had outgrown, which is a state
    that no longer exists. What survives in both modes is the third case — a
    launch aimed at a coord that is not a functional worker at all (off-grid,
    or a tile kind this architecture does not model, such as a Blackhole eth
    core). That one is reported and *not* fatal: a launch there cannot hang the
    host (the NullCore reads back ``RUN_MSG_DONE`` at once), and the detection
    is a 4-byte-write heuristic, so stopping the run on it would trade a
    diagnostic for a false kill.
    """
    pool = set(tensix_pool)
    reported_non_workers = set()

    def warn_unmapped(coord):
        if coord in tensix_coord_map and coord not in pool:
            print(
                f"[server] WARNING: wire traffic to functional worker "
                f"{coord[0]}-{coord[1]} (tile {tensix_coord_map[coord]}) "
                f"— not in TT_SIM_TENSIX_COORDS, traffic silently "
                f"NullCore-swallowed. Add `{coord[0]}-{coord[1]}` to "
                f"TT_SIM_TENSIX_COORDS to materialise it.",
                file=sys.stderr,
                flush=True,
            )

    def error_on_unmaterialised_launch(coord):
        if coord not in tensix_coord_map:
            # Not a functional worker on this architecture — nothing could
            # have materialised it, in either mode. Say so once and carry on.
            if coord in reported_non_workers:
                return
            reported_non_workers.add(coord)
            print(
                f"[server] ERROR: kernel launch (go=GO) sent to "
                f"{coord[0]}-{coord[1]}, which is not a functional worker in "
                f"soc_descriptor.yaml — tt-sim has no tile to run it on and "
                f"the traffic is being zero-filled. Results from that core "
                f"are meaningless.",
                file=sys.stderr,
                flush=True,
            )
            return
        if coord in pool or lazy:
            return
        configured = ",".join(f"{x}-{y}" for x, y in sorted(pool))
        where = f"{coord[0]}-{coord[1]}"
        print(
            f"[server] ERROR: kernel launch (go=GO) sent to functional worker "
            f"{where} (tile {tensix_coord_map[coord]}), which tt-sim did not "
            f"materialise — the program runs on more cores than tt-sim was "
            f"started with. The host is now waiting for a go-message "
            f"completion from a tile that does not exist. Add `{where}` to "
            f"TT_SIM_TENSIX_COORDS (currently: {configured}), or raise "
            f"TT_SIM_TENSIX_CORES.",
            file=sys.stderr,
            flush=True,
        )
        # End the *run*, not just the server. Exiting here without taking the
        # host with us is what turned this diagnostic into a hang: UMD has no
        # "simulator died" path, so the host blocks in recv for ever. Only exit
        # once the host is confirmed stopped; otherwise keep serving, so the
        # host reaches its own (wrong, loudly announced) conclusion rather than
        # never reaching one.
        if host_stopper(
            f"a kernel launch on worker {where} to complete", addr=wire_addr
        ):
            os._exit(1)

    fabric.unmapped_callback = warn_unmapped
    fabric.kernel_launch_callback = error_on_unmaterialised_launch
