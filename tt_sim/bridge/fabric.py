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
        core = self.cores.get(coord)
        if core is None:
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
