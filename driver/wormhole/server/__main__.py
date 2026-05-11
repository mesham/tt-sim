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

        diagnostics = diagnostics_from_env()
        device = Device(cycles_per_poll=args.cycles_per_poll, diagnostics=diagnostics)
        for translated, unified in TENSIX_COORD_MAP.items():
            fabric.register(translated, TensixCore(device, unified))
        for translated, unified in DRAM_COORD_MAP.items():
            fabric.register(translated, DramCore(device, unified))
        enabled = enabled_diagnostic_names(diagnostics)
        print(
            f"[server] tt-sim Wormhole ready "
            f"(tensix={list(TENSIX_COORD_MAP)}, dram={list(DRAM_COORD_MAP)}, "
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
