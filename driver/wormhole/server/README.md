# tt-sim ⇄ tt-metal wire bridge

A Python server that wraps mesham's tt-sim `Wormhole` behind the
`tt_SimulationDevice` IPC protocol used by tt-metal's UMD layer. With the
server running, tt-metal can drive the in-process simulator as if it were a
real device — kernel binaries are loaded over the wire, BRISC executes, and
the host reads results back.

## How tt-metal launches it

When `TT_METAL_SIMULATOR=<dir>` is set, UMD's `tt_SimulationDevice`
constructor:

1. Builds an nng `pair1` dialer at
   `ipc:///tmp/<user>_<MM-DD-HH:MM:SS>_nng_ipc` and exports that path as
   `NNG_SOCKET_ADDR`.
2. Spawns `<dir>/run.sh` via `uv_spawn` with `UV_PROCESS_DETACHED`, inheriting
   `stdout`/`stderr`.
3. Connects with retry; expects the simulator to send a one-shot
   `DEVICE_COMMAND::EXIT` message as its "I'm alive" handshake.

After that handshake every message is one flatbuffer (`DeviceRequestResponse`)
with one of six commands: `WRITE`, `READ`, `ALL_TENSIX_RESET_DEASSERT`,
`ALL_TENSIX_RESET_ASSERT`, `START` (unused), `EXIT`. `WRITE`/`RESET_*` are
fire-and-forget; `READ` requires exactly one response.

## Usage with tt-metal

```bash
export TT_METAL_SIMULATOR=/home/nick/projects/riscv/tt-sim/driver/wormhole
<your tt-metal program>
```

`run.sh` in the parent directory does the rest — it sets `PYTHONPATH` and
execs `python3 -m driver.wormhole.server --log-protocol`.

## Standalone usage (no tt-metal)

```bash
# Terminal 1 — bind the listener at an explicit address.
export NNG_SOCKET_ADDR=ipc:///tmp/ttsim.sock
python3 -m driver.wormhole.server --log-protocol

# Terminal 2 — drive it with the replay tool.
export NNG_SOCKET_ADDR=ipc:///tmp/ttsim.sock
python3 driver/wormhole/replay.py traces/some.trace
```

## CLI flags

| Flag | Purpose |
| --- | --- |
| `--addr ADDR` | Listen address (default: `$NNG_SOCKET_ADDR`). |
| `--log-protocol` | Print every wire message to stderr. |
| `--mock-tensix` | Skip building a tt-sim Wormhole. Every core is `NullCore` (writes swallowed, reads return zeros). Useful for transport regressions and for matching the phase-1 zero-stub. |
| `--cycles-per-poll N` | Run `wormhole.run(N)` after every wire message once any BRISC is out of reset (default 100). Tune this if tt-metal's poll budget expires before BRISC reaches a "done" state, or if the simulator is unnecessarily slow. |
| `--record FILE` | Append every host→sim message (and READ reply) to FILE in the trace format. Replayable with `replay.py`. |

When UMD spawns `run.sh`, the same flags can be set via env vars (UMD inherits
the parent env). `run.sh` translates these into CLI args:

| Env var | Flag equivalent |
| --- | --- |
| `TT_SIM_RECORD=<path>` | `--record <path>` |
| `TT_SIM_LOG_PROTOCOL=1` | `--log-protocol` |
| `TT_SIM_MOCK_TENSIX=1` | `--mock-tensix` |
| `TT_SIM_CYCLES_PER_POLL=N` | `--cycles-per-poll N` |

## Package layout

```
driver/wormhole/
├── run.sh                       # UMD entry point
├── replay.py                    # standalone trace replayer (client)
├── soc_descriptor.yaml          # passed to UMD; copy of tt-metal's wormhole_b0_80_arch.yaml
└── server/
    ├── __main__.py              # CLI + wiring
    ├── transport.py             # nng + flatbuffer dispatch loop
    ├── protocol.py              # build_msg / parse — bytes at the boundary
    ├── fabric.py                # coord → Core dispatch, lazy NullCore
    ├── cores.py                 # NullCore / TensixCore / DramCore
    ├── coords.py                # translated ↔ unified coord map
    ├── device.py                # Wormhole wrapper + cycle pumping + reset tracking
    ├── trace.py                 # line-oriented trace writer + parser
    ├── regen_flatbuf.sh         # regenerate Python bindings from the .fbs schema
    ├── _flatbuf/                # committed flatc-generated bindings
    ├── smoke_test.py            # transport ↔ NullCore round-trip
    ├── replay_smoke_test.py     # record + replay pipeline
    ├── tensix_smoke_test.py     # TensixCore + DramCore + cycle pump
    └── traces/                  # captured tt-metal traces (regression fixtures)
```

## Trace format

One host→sim message per line, ASCII, whitespace-separated:

```
<CMD> core=<x>,<y> addr=0x<hex> size=<n> data=<hex|-> [reply=<hex|->]
```

`CMD` is the textual name (`WRITE`, `READ`, ...). `data=` is the host's
payload bytes in lowercase hex (`-` for empty). `reply=` is only present on
`READ` lines and captures what the server returned at record time. Comment
lines (`#`) and blank lines are skipped.

Sample (recorded against `--mock-tensix`):

```
WRITE core=1,1 addr=0x2010 size=4 data=01020304
READ core=1,1 addr=0x2a0 size=4 data=- reply=00000000
RESET_DEASSERT core=1,1 addr=0x0 size=0 data=-
EXIT core=0,0 addr=0x0 size=0 data=-
```

## Capturing a real tt-metal trace

UMD spawns `run.sh` itself, so capture is driven by env vars that `run.sh`
forwards to the server:

```bash
# Real-Wormhole capture — replies reflect actual simulator state, so this
# trace verifies end-to-end correctness on replay.
TT_SIM_RECORD=$(pwd)/driver/wormhole/server/traces/one.trace \
TT_METAL_SIMULATOR=$(pwd)/driver/wormhole \
    <your tt-metal program>

# NullCore capture — all replies are zeros; verifies only transport + init.
TT_SIM_MOCK_TENSIX=1 \
TT_SIM_RECORD=$(pwd)/driver/wormhole/server/traces/one_mocktensix.trace \
TT_METAL_SIMULATOR=$(pwd)/driver/wormhole \
    <your tt-metal program>
```

Then replay against any future server build to check regressions:

```bash
NNG_SOCKET_ADDR=ipc:///tmp/replay.sock python3 -m driver.wormhole.server \
    --addr ipc:///tmp/replay.sock &
python3 driver/wormhole/replay.py driver/wormhole/server/traces/one.trace
```

Traces are server-mode specific: a real-Wormhole trace replays cleanly only
against another real-Wormhole server (or one whose state matches), and a
`--mock-tensix` trace replays cleanly only against `--mock-tensix`.

## Regenerating flatbuffer bindings

The schema lives in tt-metal at
`tt_metal/third_party/umd/device/simulation/tt_simulation_device.fbs`. The
generated Python bindings under `server/_flatbuf/` are committed, but to
rebuild them after a tt-metal schema bump:

```bash
./driver/wormhole/server/regen_flatbuf.sh
# or, against a different schema:
./driver/wormhole/server/regen_flatbuf.sh /path/to/tt_simulation_device.fbs
```

The script post-patches `DeviceRequestResponse.py` so its inline import of
`tt_vcs_core` becomes a relative import (`from .tt_vcs_core import …`).

## Smoke tests

Three Python scripts under `server/` exercise the stack incrementally; each
runs both ends of a pair1 socket in-process:

```bash
PYTHONPATH=. python3 -m driver.wormhole.server.smoke_test
# transport ↔ NullCore round trip (Phase 1)

PYTHONPATH=. python3 -m driver.wormhole.server.replay_smoke_test
# record + replay pipeline (Phase 2)

PYTHONPATH=. python3 -m driver.wormhole.server.tensix_smoke_test
# Wormhole + TensixCore + DramCore + cycle pump (Phase 3)

PYTHONPATH=. python3 -m driver.wormhole.server.one_replay_test
# replay the captured tt-metal "one" trace; verifies every recorded
# reply (including the kernel's result buffer at (0,11):0x360)
# reproduces bit-for-bit. Skips cleanly if traces/one.trace is absent.
```

## Stub-out / extension points

Search for `TODO(future)` in the package; the current ones:

- `coords.py` — derive the translated↔unified map from `soc_descriptor.yaml`.
- `cores.py:TensixCore.deassert_reset` — fan reset deassertion out to
  NCRISC/TRISC0/1/2 keyed off the launch message's `enables` field.
- `cores.py:DramCore` — extend to all 18 DRAM channels enumerated in the SoC
  descriptor.
- `fabric.py` — multi-Tensix expansion (needs `Wormhole.__init__` to
  instantiate more tiles).
- `transport.py` — `START` command handler (cmd=4); currently log-and-skip.
