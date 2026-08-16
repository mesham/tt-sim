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
    LazyTensixPool,
    TensixCore,
    TraceWriter,
    Transport,
    compute_grid,
    diagnostics_from_env,
    enabled_diagnostic_names,
    host_not_stranded,
    install_convention_guard,
    install_worker_guards,
    link_contention_summary,
    profiler_flush_summary,
)
from tt_sim.network.noc_translation import translation_source

from .bh_device import make_device
from .coords import (
    CLUSTER_DESCRIPTOR_PATH,
    DEFAULT_COMPUTE_GRID,
    DRAM_COORD_MAP,
    TENSIX_COORD_MAP,
    default_tensix_coords,
    wire_conventions,
)


def _parse_tensix_pool(env):
    """``(coords, pinned)``: workers to pre-build, and whether the user pinned
    that set.

    Same contract as the Wormhole server's — see its docstring. With neither
    env var set the default worker ``(1, 2)`` is built up front and everything
    else is materialised on demand.
    """
    coords_raw = env.get("TT_SIM_TENSIX_COORDS")
    cores_raw = env.get("TT_SIM_TENSIX_CORES")
    if coords_raw is not None and cores_raw is not None:
        raise SystemExit("set only one of TT_SIM_TENSIX_COORDS or TT_SIM_TENSIX_CORES")
    if cores_raw is not None:
        try:
            return default_tensix_coords(int(cores_raw.strip())), True
        except ValueError as e:
            raise SystemExit(f"TT_SIM_TENSIX_CORES: {e}") from None

    # Default: the first functional worker the profile targets, (1, 2).
    if coords_raw is None:
        return [(1, 2)], False

    raw = coords_raw
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
    return pool, True


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
    lazy_pool = None
    if not args.mock_tensix:
        tensix_pool, pinned = _parse_tensix_pool(os.environ)
        diagnostics = diagnostics_from_env()
        translated, why = translation_source(os.environ)
        device = make_device(
            cycles_per_poll=args.cycles_per_poll,
            diagnostics=diagnostics,
            noc_translation=translated,
        )
        alias, translated_only, untranslated_only = wire_conventions()
        install_convention_guard(
            fabric,
            translated=translated,
            wire_alias=alias if translated else {},
            foreign_coords=untranslated_only if translated else translated_only,
            reason=why,
            descriptor_hint=CLUSTER_DESCRIPTOR_PATH,
            wire_addr=addr,
        )
        for wire_coord, tile_coord in DRAM_COORD_MAP.items():
            fabric.register(wire_coord, DramCore(device, tile_coord))
        if pinned:
            for physical in tensix_pool:
                device.ensure_tensix_tile(physical)
                fabric.register(
                    physical, TensixCore(device, TENSIX_COORD_MAP[physical])
                )
        else:
            # Nothing pinned: build the default worker and let the program ask
            # for the rest — see ``tt_sim.bridge.materialise``.
            lazy_pool = LazyTensixPool(
                fabric,
                device,
                TENSIX_COORD_MAP,
                eager=tensix_pool,
                noc_alias=alias if translated else None,
            )
        # Same "worker isn't materialised" guards the Wormhole server installs:
        # without them traffic to (or a kernel launch on) a worker outside
        # TT_SIM_TENSIX_COORDS is silently NullCore-swallowed.
        install_worker_guards(
            fabric,
            tensix_pool,
            TENSIX_COORD_MAP,
            wire_addr=addr,
            lazy=not pinned,
        )
        enabled = enabled_diagnostic_names(diagnostics)
        grid = compute_grid(DEFAULT_COMPUTE_GRID)
        how = "pinned" if pinned else "on demand"
        print(
            f"[server] tt-sim Blackhole ready (tensix={tensix_pool} ({how}), "
            f"dram={list(DRAM_COORD_MAP)}, "
            f"compute_grid={grid[0]}x{grid[1]}, "
            f"noc_translation={'on' if translated else 'off'} ({why}), "
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
                transport.trace_writer = stack.enter_context(TraceWriter(args.record))
            # A simulator-side exception here would otherwise leave the host
            # blocked in recv for ever behind a traceback nobody sees.
            with host_not_stranded(addr):
                transport.serve(fabric)
    finally:
        if device is not None:
            device.tt_device.shutdown()

    extra = ""
    if lazy_pool is not None:
        extra = (
            f", {len(lazy_pool.materialised)} tensix materialised "
            f"({len(lazy_pool.on_demand)} on demand)"
        )
    links = link_contention_summary(device)
    if links:
        extra += f", {links}"
    flush = profiler_flush_summary(device)
    if flush:
        extra += f", {flush}"
    print(
        f"[server] shutdown after {transport.msg_count} messages{extra}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
