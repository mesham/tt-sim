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

Against tt-metal (like the Wormhole flow, but point the simulator env var here):

```bash
export TT_METAL_SIMULATOR="$PWD/driver/blackhole"
export TT_METAL_SLOW_DISPATCH_MODE=1
# ... then run a tt-metal program as in ../wormhole/README.md
```

`TT_SIM_TENSIX_COORDS` / `TT_SIM_TENSIX_CORES` and the `TT_SIM_DIAG_*` diagnostics
work exactly as for Wormhole (they are handled by the shared bridge).

## Scope / status

This is a **single Tensix + single DRAM** bring-up. Modelled and verified
in-process: device construction, the 17×12 NoC grid and NoC-1 mirror, Blackhole's
NoC HI-register address encoding, ISA-faithful DST addressing, the SFPU superset
comparison/mul ops, and WRITE/READ routed end-to-end through the bridge into the
Blackhole device.

Not yet modelled (see the plan): the full DRAM/eth grid (only channel-0 DRAM is
backed; other DRAM + eth fall to `NullCore`), Blackhole eth tiles (2 RV cores /
512 KB L1), baby-core kernel execution, and a live tt-metal run — the last needs
a Blackhole-built tt-metal.
