"""Entry point: ``python -m driver.blackhole.server``.

UMD spawns ``run.sh`` with ``NNG_SOCKET_ADDR`` set; the server binds there and
drives a tt-sim Blackhole over the wire, sharing all the protocol/transport
machinery in :mod:`tt_sim.bridge`. Only the device factory and coordinate maps
are Blackhole-specific (see ``bh_device`` / ``coords``).

This is the minimal single-tile bring-up; see ``docs/plans/blackhole-support.md``
for what is not yet modelled (full DRAM/eth grid, coordinate-translation table).
"""

import argparse
import contextlib
import os
import sys

from tt_sim.bridge import (
    DramCore,
    Fabric,
    TensixCore,
    TraceWriter,
    Transport,
    diagnostics_from_env,
    enabled_diagnostic_names,
    install_worker_guards,
)

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP, default_tensix_coords


def _parse_tensix_pool(env):
    """Physical worker coords to pre-construct (TT_SIM_TENSIX_COORDS / _CORES)."""
    coords_raw = env.get("TT_SIM_TENSIX_COORDS")
    cores_raw = env.get("TT_SIM_TENSIX_CORES")
    if coords_raw is not None and cores_raw is not None:
        raise SystemExit("set only one of TT_SIM_TENSIX_COORDS or TT_SIM_TENSIX_CORES")
    if cores_raw is not None:
        try:
            return default_tensix_coords(int(cores_raw.strip()))
        except ValueError as e:
            raise SystemExit(f"TT_SIM_TENSIX_CORES: {e}") from None

    # Default: the first functional worker the profile targets, (1, 2).
    raw = coords_raw if coords_raw is not None else "1-2"
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
    ap = argparse.ArgumentParser(prog="driver.blackhole.server")
    ap.add_argument(
        "--addr", default=None, help="nng IPC address (default: $NNG_SOCKET_ADDR)"
    )
    ap.add_argument("--log-protocol", action="store_true")
    ap.add_argument("--mock-tensix", action="store_true", help="every core is NullCore")
    ap.add_argument("--cycles-per-poll", type=int, default=100, metavar="N")
    ap.add_argument("--record", metavar="FILE", default=None)
    # Inert marker: the test scripts cannot reach this process by pid (UMD
    # spawns run.sh detached), so they stamp their run tag into our command
    # line and match on it when cleaning up. See driver/sim_procs.sh.
    ap.add_argument("--run-tag", metavar="TAG", default=None)
    args = ap.parse_args(argv)

    addr = args.addr or os.environ.get("NNG_SOCKET_ADDR")
    if not addr:
        print("error: NNG_SOCKET_ADDR not set and --addr not provided", file=sys.stderr)
        return 2

    print(f"[server] listening on {addr}", file=sys.stderr, flush=True)

    fabric = Fabric()
    device = None
    if not args.mock_tensix:
        tensix_pool = _parse_tensix_pool(os.environ)
        diagnostics = diagnostics_from_env()
        device = make_device(
            cycles_per_poll=args.cycles_per_poll, diagnostics=diagnostics
        )
        for translated, tile_coord in DRAM_COORD_MAP.items():
            fabric.register(translated, DramCore(device, tile_coord))
        for physical in tensix_pool:
            device.ensure_tensix_tile(physical)
            fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
        # Same "worker isn't materialised" guards the Wormhole server installs:
        # without them traffic to (or a kernel launch on) a worker outside
        # TT_SIM_TENSIX_COORDS is silently NullCore-swallowed.
        install_worker_guards(fabric, tensix_pool, TENSIX_COORD_MAP)
        enabled = enabled_diagnostic_names(diagnostics)
        print(
            f"[server] tt-sim Blackhole ready (tensix={tensix_pool}, "
            f"dram={list(DRAM_COORD_MAP)}, cycles_per_poll={args.cycles_per_poll})",
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
                transport.trace_writer = stack.enter_context(TraceWriter(args.record))
            transport.serve(fabric)
    finally:
        if device is not None:
            device.tt_device.shutdown()

    print(
        f"[server] shutdown after {transport.msg_count} messages",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
