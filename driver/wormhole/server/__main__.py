"""Entry point: ``python -m driver.wormhole.server``.

UMD spawns ``run.sh`` with ``NNG_SOCKET_ADDR`` already set in the environment;
the server binds there and immediately ack's with an EXIT message.
"""

import argparse
import contextlib
import os
import sys

from .fabric import Fabric
from .trace import TraceWriter
from .transport import Transport


def _parse_tensix_pool(env):
    """Return the physical worker coords the bridge should pre-construct.

    Two ways to specify the pool, in precedence order:

    - ``TT_SIM_TENSIX_COORDS``: comma-separated ``x-y`` physical NoC coords,
      e.g. ``"1-1,2-1"`` — exact control over which workers exist.
    - ``TT_SIM_TENSIX_CORES``: a bare count ``N`` — materialise N workers at a
      sensible default coord set (``coords.default_tensix_coords``, column-major
      from ``1-1``), so you can drive by core *count* without naming coords.

    Setting both is an error (ambiguous). Neither defaults to ``[(1, 1)]`` — the
    single-tile coord every wormhole example targets. Coords are validated
    against the SoC-descriptor-derived ``TENSIX_COORD_MAP`` so typos surface
    immediately rather than as silent NullCore zero-fills at runtime.
    """
    from .coords import TENSIX_COORD_MAP, default_tensix_coords

    coords_raw = env.get("TT_SIM_TENSIX_COORDS")
    cores_raw = env.get("TT_SIM_TENSIX_CORES")
    if coords_raw is not None and cores_raw is not None:
        raise SystemExit(
            "set only one of TT_SIM_TENSIX_COORDS (explicit coords) or "
            "TT_SIM_TENSIX_CORES (a core count), not both"
        )

    if cores_raw is not None:
        try:
            n = int(cores_raw.strip())
        except ValueError:
            raise SystemExit(
                f"TT_SIM_TENSIX_CORES must be an integer, got {cores_raw!r}"
            ) from None
        try:
            return default_tensix_coords(n)
        except ValueError as e:
            raise SystemExit(f"TT_SIM_TENSIX_CORES: {e}") from None

    raw = coords_raw if coords_raw is not None else "1-1"
    pool = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_s, _, y_s = chunk.partition("-")
        physical = (int(x_s), int(y_s))
        if physical not in TENSIX_COORD_MAP:
            raise SystemExit(
                f"TT_SIM_TENSIX_COORDS: {chunk!r} is not a functional worker "
                f"in soc_descriptor.yaml"
            )
        pool.append(physical)
    if not pool:
        raise SystemExit("TT_SIM_TENSIX_COORDS is set but empty")
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(prog="driver.wormhole.server")
    ap.add_argument(
        "--addr",
        default=None,
        help="nng IPC address (default: $NNG_SOCKET_ADDR)",
    )
    ap.add_argument(
        "--log-protocol",
        action="store_true",
        help="print every protocol message to stderr",
    )
    ap.add_argument(
        "--mock-tensix",
        action="store_true",
        help="skip building a tt-sim Wormhole; every core is NullCore",
    )
    ap.add_argument(
        "--cycles-per-poll",
        type=int,
        default=100,
        metavar="N",
        help="simulator cycles to run after each wire message (default 100)",
    )
    ap.add_argument(
        "--record",
        metavar="FILE",
        default=None,
        help="record every wire message (and READ reply) to FILE",
    )
    args = ap.parse_args(argv)

    addr = args.addr or os.environ.get("NNG_SOCKET_ADDR")
    if not addr:
        print(
            "error: NNG_SOCKET_ADDR not set and --addr not provided",
            file=sys.stderr,
        )
        return 2

    print(f"[server] listening on {addr}", file=sys.stderr, flush=True)

    fabric = Fabric()
    device = None
    if not args.mock_tensix:
        # Late import so --mock-tensix avoids the (slow) Wormhole construction.
        from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
        from .cores import DramCore, EthCore, TensixCore
        from .device import Device, diagnostics_from_env, enabled_diagnostic_names

        tensix_pool = _parse_tensix_pool(os.environ)
        diagnostics = diagnostics_from_env()
        device = Device(cycles_per_poll=args.cycles_per_poll, diagnostics=diagnostics)
        for translated, unified in DRAM_COORD_MAP.items():
            fabric.register(translated, DramCore(device, unified))
        for translated, unified in ETH_COORD_MAP.items():
            fabric.register(translated, EthCore(device, unified))
        # Eagerly materialise the worker tiles the user asked for. Every
        # other worker coord falls through to NullCore (matching the
        # zero-stub behaviour tt-metal's grid-wide init traffic tolerates),
        # which is critical for keeping the cycle pump cheap — see the
        # ROADMAP §A "Multi-Tensix threading" perf note.
        for physical in tensix_pool:
            unified = TENSIX_COORD_MAP[physical]
            device.ensure_tensix_tile(physical)
            fabric.register(physical, TensixCore(device, unified))

        # Surface the most common "silent zero-fill" config bug: host
        # traffic addresses a functional_worker that wasn't pre-built via
        # TT_SIM_TENSIX_COORDS. Without this warning the user sees
        # mismatch output (or just zeros) with no hint that a missing
        # coord was the cause.
        _tensix_pool_set = set(tensix_pool)

        def _warn_unmapped_worker(coord):
            if coord in TENSIX_COORD_MAP and coord not in _tensix_pool_set:
                print(
                    f"[server] WARNING: wire traffic to functional worker "
                    f"{coord[0]}-{coord[1]} (unified {TENSIX_COORD_MAP[coord]}) "
                    f"— not in TT_SIM_TENSIX_COORDS, traffic silently "
                    f"NullCore-swallowed. Add `{coord[0]}-{coord[1]}` to "
                    f"TT_SIM_TENSIX_COORDS to materialise it.",
                    file=sys.stderr,
                    flush=True,
                )

        fabric.unmapped_callback = _warn_unmapped_worker

        def _error_kernel_launch_on_unmaterialised(coord):
            # go=GO reached a coord tt-sim didn't materialise. If it's a
            # functional worker, the program launches a kernel on a core the
            # user didn't start — a silent NullCore swallow here would surface
            # only as a downstream hang (the peer cores wait on NoC traffic
            # this core never sends). Fail loudly and immediately instead.
            if coord not in TENSIX_COORD_MAP or coord in _tensix_pool_set:
                return
            configured = ",".join(f"{x}-{y}" for x, y in sorted(_tensix_pool_set))
            print(
                f"[server] ERROR: kernel launch (go=GO) sent to functional "
                f"worker {coord[0]}-{coord[1]} (unified {TENSIX_COORD_MAP[coord]}), "
                f"which tt-sim did not materialise — the program runs on more "
                f"cores than tt-sim was started with. Add `{coord[0]}-{coord[1]}` "
                f"to TT_SIM_TENSIX_COORDS (currently: {configured}), or raise "
                f"TT_SIM_TENSIX_CORES.",
                file=sys.stderr,
                flush=True,
            )
            # Stop the server immediately. tt-metal's UMD has no "simulator
            # died" path — it blocks forever on its next go-message poll no
            # matter how we close the socket — so the host still needs a Ctrl-C.
            # But the message above prints the instant the launch is attempted
            # (i.e. right when the hang begins), so the reason is on screen, and
            # exiting here means we don't leave an orphaned server behind.
            os._exit(1)

        fabric.kernel_launch_callback = _error_kernel_launch_on_unmaterialised

        enabled = enabled_diagnostic_names(diagnostics)
        print(
            f"[server] tt-sim Wormhole ready "
            f"(tensix={tensix_pool}, dram={list(DRAM_COORD_MAP)}, "
            f"cycles_per_poll={args.cycles_per_poll})",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[server] diagnostics: {', '.join(enabled) if enabled else 'none'}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            "[server] --mock-tensix: all cores are NullCore",
            file=sys.stderr,
            flush=True,
        )

    transport = Transport(addr, log_protocol=args.log_protocol)

    try:
        with contextlib.ExitStack() as stack:
            if args.record:
                tracer = stack.enter_context(TraceWriter(args.record))
                transport.trace_writer = tracer
                print(
                    f"[server] recording trace to {args.record}",
                    file=sys.stderr,
                    flush=True,
                )

            transport.serve(fabric)
    finally:
        # Join per-tile worker threads spawned by MultiTileClock so they
        # don't outlive the process on graceful exit or Ctrl-C. Safe even
        # when --mock-tensix is set (no Wormhole was built).
        if device is not None:
            device.wormhole.shutdown()

    print(
        f"[server] shutdown after {transport.msg_count} messages",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
