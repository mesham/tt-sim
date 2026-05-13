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

    Reads ``TT_SIM_TENSIX_COORDS``: comma-separated ``x-y`` physical NoC
    coords, e.g. ``"1-1,2-1"``. Whitespace tolerated. Defaults to
    ``[(1, 1)]`` — the single-tile coord every wormhole example targets.
    Validates each entry against the SoC-descriptor-derived
    ``TENSIX_COORD_MAP`` so typos surface immediately rather than as
    silent NullCore zero-fills at runtime.
    """
    from .coords import TENSIX_COORD_MAP

    raw = env.get("TT_SIM_TENSIX_COORDS", "1-1")
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
    if not args.mock_tensix:
        # Late import so --mock-tensix avoids the (slow) Wormhole construction.
        from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP
        from .cores import DramCore, TensixCore
        from .device import Device, diagnostics_from_env, enabled_diagnostic_names

        tensix_pool = _parse_tensix_pool(os.environ)
        diagnostics = diagnostics_from_env()
        device = Device(cycles_per_poll=args.cycles_per_poll, diagnostics=diagnostics)
        for translated, unified in DRAM_COORD_MAP.items():
            fabric.register(translated, DramCore(device, unified))
        # Eagerly materialise the worker tiles the user asked for. Every
        # other worker coord falls through to NullCore (matching the
        # zero-stub behaviour tt-metal's grid-wide init traffic tolerates),
        # which is critical for keeping the cycle pump cheap — see the
        # ROADMAP §A "Multi-Tensix threading" perf note.
        for physical in tensix_pool:
            unified = TENSIX_COORD_MAP[physical]
            device.ensure_tensix_tile(physical)
            fabric.register(physical, TensixCore(device, unified))
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

    print(
        f"[server] shutdown after {transport.msg_count} messages",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
