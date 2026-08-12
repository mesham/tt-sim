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

    def register(self, coord, core):
        self.cores[coord] = core

    def _core(self, coord):
        core = self.cores.get(coord)
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
    fabric, tensix_pool, tensix_coord_map, *, wire_addr=None, host_stopper=stop_host
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
    """
    pool = set(tensix_pool)

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
        if coord not in tensix_coord_map or coord in pool:
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
