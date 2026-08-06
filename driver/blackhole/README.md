# Blackhole driver

A Blackhole bring-up of tt-sim, sharing all the tt-metal wire-bridge machinery
with the Wormhole driver (`tt_sim/bridge`) — only the device factory, SoC
descriptor and coordinate maps here are Blackhole-specific. See the top-level
`docs/plans/blackhole-support.md` for the full multi-arch port.

## Layout

- `soc_descriptor.yaml` — tt-metal's `blackhole_140_arch.yaml` (17×12 grid, 8 DRAM
  channels, 140 workers).
- `run.sh` — what UMD spawns; execs `python -m driver.blackhole.server`.
- `server/` — thin entry point onto `tt_sim.bridge`:
  - `bh_device.py` — the `Blackhole` device factory + `make_device`.
  - `coords.py` — coordinate maps. Because Blackhole tiles are keyed by their
    **physical NoC coord** (not a "unified" band), the tensix/dram maps are
    identity; the complex NIU coordinate-translation table lives in hardware/UMD,
    below the wire, so it never enters the bridge.
  - `__main__.py` — parses args, builds the fabric, registers cores, serves.
- `bringup.py` — a standalone (no tt-metal) demo: constructs the device and does
  a real DRAM→L1 **NoC read** with Blackhole's HI-register address encoding.

## Running

Standalone bring-up demo:

```bash
python3 -m driver.blackhole.bringup
```

Against tt-metal — build and run the shared examples, pointing the simulator env
var here instead of at Wormhole:

```bash
export TT_METAL_SIMULATOR="$PWD/driver/blackhole"
export TT_METAL_SLOW_DISPATCH_MODE=1
```

The full build + run flow (CMake, the run environment, the whole-suite runner
`tests/run_examples.sh`, and the offline replay guards) is documented in
**[examples/README.md](../../examples/README.md)** — the same doc both arches share.
The only Blackhole difference is the worker coordinate convention: logical `(0,0)`
maps to physical `(1,2)` (vs `(1,1)` on Wormhole), so single-tile examples use
`TT_SIM_TENSIX_COORDS=1-2` and `nine` uses `1-2,2-2`. `TT_SIM_TENSIX_CORES` and the
`TT_SIM_DIAG_*` diagnostics work exactly as for Wormhole (handled by the shared bridge).

## Scope / status

**Working end-to-end.** All bundled examples (`one`–`nine`, `loopback`) build and
run live against `driver/blackhole` and validate their own results — including the
`six` bf16 matmul (matrix engine + K-accumulation) and the two-tile `nine` (cross-tile
NoC + semaphore). Each has a socket-free replay guard in `server/*_replay_test.py`, and
Wormhole stays byte-identical throughout. See `docs/plans/blackhole-support.md` for the
full port history.

Those guards cover three families now: the shared `examples/` tree, the
`optests/` differential programs, and — since the upstream sweep
(`docs/upstream-examples-status.md`) — four of tt-metal's own
`programming_examples/`: `noc_tile_transfer` (2 workers, semaphore-gated NoC
tile hand-off, asserting the program's own `Result = 14`), `vecadd_sharding`
(4 workers, L1-sharded buffers driven by a compute-only kernel),
`pad_multi_core` and `shard_data_rm` (4 workers, page-granular interleaved and
row-major sharded data movement, neither with a self-check of its own).
Recapture their traces with `driver/tests/capture_upstream_traces.sh`.

Modelled: device construction, the 17×12 NoC grid + NoC-1 mirror, Blackhole's NoC
HI-register address encoding, ISA-faithful DST addressing, all **8 DRAM channels**,
baby-core kernel execution across BRISC/NCRISC/TRISC, and the Tensix coprocessor
(unpackers, matrix FPU, SFPU superset, packer).

Not yet modelled (see the plan): the full worker grid (tiles are materialised on
demand; the default is a single Tensix), Blackhole eth tiles (2 RV cores / 512 KB L1),
and ttsim-Blackhole differential testing.
