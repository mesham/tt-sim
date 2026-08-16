"""Entry point: ``python -m driver.wormhole.server``.

UMD spawns ``run.sh`` with ``NNG_SOCKET_ADDR`` already set in the environment;
the server binds there and immediately ack's with an EXIT message.
"""

import argparse
import contextlib
import os
import sys

from tt_sim.bridge import (
    Fabric,
    TraceWriter,
    Transport,
    host_not_stranded,
    link_contention_summary,
    profiler_flush_summary,
)


def _parse_tensix_pool(env):
    """Return ``(coords, pinned)``: the workers to pre-build, and whether the
    user pinned that set.

    With **neither** env var set the answer is ``([(1, 1)], False)``: the
    historical single-tile default is still built up front — so the single-tile
    path is unchanged down to the cycle — but ``pinned=False`` lets the server
    install a :class:`~tt_sim.bridge.LazyTensixPool` that materialises any
    other worker the program turns out to need. No coordinates to work out, and
    no tax for workers a program never launches on.

    Two ways to pin the set explicitly, in precedence order:

    - ``TT_SIM_TENSIX_COORDS``: comma-separated ``x-y`` physical NoC coords,
      e.g. ``"1-1,2-1"`` — exactly these workers exist, and no others ever will.
    - ``TT_SIM_TENSIX_CORES``: a bare count ``N`` — the N workers tt-metal will
      actually launch on (``coords.default_tensix_coords``: column-major over
      its compute grid), so you can drive by core *count* without naming coords.

    Both pin, because the replay guards and the cost-model gate want a fixed,
    reproducible grid. Setting both is an error (ambiguous). Coords are
    validated against the SoC-descriptor-derived ``TENSIX_COORD_MAP`` so typos
    surface immediately rather than as silent zero-fills at runtime.
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
            return default_tensix_coords(n), True
        except ValueError as e:
            raise SystemExit(f"TT_SIM_TENSIX_CORES: {e}") from None

    if coords_raw is None:
        return [(1, 1)], False

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
    ap.add_argument(
        "--run-tag",
        metavar="TAG",
        default=None,
        help=(
            "inert marker stamped into our command line by a test script, so it "
            "can clean up exactly the servers it started (driver/sim_procs.sh)"
        ),
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
    lazy_pool = None
    if not args.mock_tensix:
        # Late import so --mock-tensix avoids the (slow) Wormhole construction.
        from tt_sim.bridge import (
            DramCore,
            EthCore,
            LazyTensixPool,
            TensixCore,
            diagnostics_from_env,
            enabled_diagnostic_names,
            install_convention_guard,
            install_worker_guards,
        )
        from tt_sim.network.noc_translation import translation_source

        from .coords import (
            CLUSTER_DESCRIPTOR_PATH,
            DRAM_COORD_MAP,
            ETH_COORD_MAP,
            TENSIX_COORD_MAP,
            wire_conventions,
        )
        from .wh_device import make_device

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
        for wire_coord, unified in DRAM_COORD_MAP.items():
            fabric.register(wire_coord, DramCore(device, unified))
        for wire_coord, unified in ETH_COORD_MAP.items():
            fabric.register(wire_coord, EthCore(device, unified))
        if pinned:
            # The user named the set; build exactly it and nothing else. Every
            # other worker coord falls through to NullCore, as it always has.
            for physical in tensix_pool:
                unified = TENSIX_COORD_MAP[physical]
                device.ensure_tensix_tile(physical)
                fabric.register(physical, TensixCore(device, unified))
        else:
            # Nothing pinned: build the default worker up front and let the
            # program ask for the rest. See ``tt_sim.bridge.materialise`` — a
            # worker appears when the host writes to it after releasing it,
            # when a kernel launches on it, or when a peer sends it NoC
            # traffic, whichever comes first.
            lazy_pool = LazyTensixPool(
                fabric,
                device,
                TENSIX_COORD_MAP,
                eager=tensix_pool,
                noc_alias=alias if translated else None,
            )

        # Surface the most common "silent zero-fill" config bug: host traffic
        # (or worse, a kernel launch) addressing a functional_worker that
        # wasn't pre-built via TT_SIM_TENSIX_COORDS. Shared with the Blackhole
        # server so both architectures shout about it — see
        # ``tt_sim.bridge.install_worker_guards``.
        install_worker_guards(
            fabric,
            tensix_pool,
            TENSIX_COORD_MAP,
            wire_addr=addr,
            lazy=not pinned,
        )

        enabled = enabled_diagnostic_names(diagnostics)
        from tt_sim.bridge import compute_grid

        from .coords import DEFAULT_COMPUTE_GRID

        grid = compute_grid(DEFAULT_COMPUTE_GRID)
        how = "pinned" if pinned else "on demand"
        print(
            f"[server] tt-sim Wormhole ready "
            f"(tensix={tensix_pool} ({how}), dram={list(DRAM_COORD_MAP)}, "
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
                tracer = stack.enter_context(TraceWriter(args.record))
                transport.trace_writer = tracer
                print(
                    f"[server] recording trace to {args.record}",
                    file=sys.stderr,
                    flush=True,
                )

            # A simulator-side exception here would otherwise leave the host
            # blocked in recv for ever behind a traceback nobody sees.
            with host_not_stranded(addr):
                transport.serve(fabric)
    finally:
        # Join per-tile worker threads spawned by MultiTileClock so they
        # don't outlive the process on graceful exit or Ctrl-C. Safe even
        # when --mock-tensix is set (no Wormhole was built).
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
