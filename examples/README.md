# tt-sim examples

Real tt-metal programs used to exercise the simulator end-to-end. Each example is
an ordinary tt-metal host program that you **build against a local tt-metal
checkout and run exactly the way you would on hardware** — the only difference is
that `TT_METAL_SIMULATOR` points UMD at tt-sim (`driver/wormhole` or
`driver/blackhole`) instead of a real chip. Every program validates its own
result on the host and exits non-zero on mismatch, so the examples double as a
test suite.

The sources are **architecture-agnostic**: the same binary runs on either arch
depending on `TT_METAL_SIMULATOR` (kernels JIT for the target). Only the worker
coordinate convention differs — see [Running](#running) below.

```
examples/
├── <name>/src/          one example: <name>.cpp + CMakeLists.txt + kernels/
├── examples_test.py     Wormhole live runner (build + run each, assert pass)
└── README.md            this file
```

## Requirements

- A built tt-metal checkout that exports the `TT-Metalium` CMake package — i.e.
  `<tt-metal>/<build>/lib/cmake/tt-metalium/tt-metalium-config.cmake` exists (true
  for a normal `build/` or `build_Release/`). Point `TT_METAL_RUNTIME_ROOT` (or the
  older `TT_METAL_HOME`) at that checkout.
- `cmake` (≥ 3.22) and `clang++-17` on `PATH`.
- The tt-sim repo on `PYTHONPATH` and the venv's `python` for the simulator server
  (`source /home/nick/projects/riscv/venv/bin/activate` sets these up).

## Building an example

Each example is a self-contained CMake project. Build it from its own `src/`
directory:

```bash
cd examples/two/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# -> build/two
```

`CMakeLists.txt` locates tt-metal via `find_package(TT-Metalium)` (honouring
`TT_METAL_RUNTIME_ROOT`), so no hand-rolled include paths or flags are needed.
The build is arch-independent — you build once and choose the arch at run time.

## Running

Running an example *is* running tt-metal; the only switch that redirects it from
silicon to tt-sim is `TT_METAL_SIMULATOR`. Run the binary **from its `src/`
directory** (the host program refers to its kernels by the relative path
`kernels/...`, resolved against the CWD).

| Variable | Purpose |
| --- | --- |
| `TT_METAL_SIMULATOR` | **The switch.** Path to `driver/wormhole` or `driver/blackhole`; UMD spawns that arch's `run.sh` as the device. |
| `TT_METAL_RUNTIME_ROOT` | tt-metal checkout (CMake build + runtime kernel/firmware lookup). `TT_METAL_HOME` is accepted as a fallback. |
| `TT_METAL_SLOW_DISPATCH_MODE=1` | Forces `EnqueueProgram` to fall back to `detail::LaunchProgram` — the only launch path the simulator models. |
| `LD_LIBRARY_PATH` | Must include `<tt-metal>/<build>/lib` so the binary finds `libtt_metal.so` etc. |
| `TT_SIM_TENSIX_COORDS` | Physical worker tile(s) to materialise (see coords below). |

**Worker coordinates differ per arch.** A program's logical core `(col,row)` maps
to a physical NoC coord; that mapping is `+ (1,1)` on Wormhole and `+ (1,2)` on
Blackhole. Every bundled example runs on the single default tile except `nine`
and `pipestall`, which bridge a CB across two tiles:

| | single-tile examples | `nine` / `pipestall` (two-tile) |
| --- | --- | --- |
| **Wormhole** | `1-1` | `1-1,2-1` |
| **Blackhole** | `1-2` | `1-2,2-2` |

Example — run `two` on the Blackhole sim (in a **normal shell**, not a sandbox
that would kill the spawned server):

```bash
REPO=/home/nick/projects/riscv/tt-sim
export TT_METAL_HOME=/path/to/tt-metal
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"
export TT_METAL_SIMULATOR="$REPO/driver/blackhole"   # or "$REPO/driver/wormhole"
export TT_METAL_SLOW_DISPATCH_MODE=1

cd "$REPO/examples/two/src"
TT_SIM_TENSIX_COORDS=1-2 ./build/two
# ...tt-metal / UMD log lines...
# Completed successfully on the device, with 100 elements
```

Success prints `Completed successfully on the device`. A hang usually means the
kernel hit a Tensix/SFPU op the simulator hasn't modelled yet (the server aborts,
the host then waits on it) rather than slowness.

The full runbook — including driving the upstream `programming_examples/` and the
diagnostics knobs — is in
[`docs/running-tt-metal-on-the-simulator.md`](../docs/running-tt-metal-on-the-simulator.md).

## Running the whole suite

Build + run every example and get a pass/fail summary:

```bash
# Blackhole (bash; also records traces with --record):
TT_METAL_HOME=/path/to/tt-metal ../driver/blackhole/tests/run_examples.sh

# Wormhole (python; standalone or under pytest):
python3 -m examples.examples_test
pytest examples/examples_test.py -v
```

Both build each example as needed and run it against their arch's simulator with
the right coords. `run_examples.sh --record` (Blackhole) and
[`../driver/wormhole/tests/capture_traces.sh`](../driver/wormhole/tests/capture_traces.sh)
(Wormhole) additionally record a wire trace per example into the arch's
`server/traces/`.

## Offline regression guards (no live sim needed)

Each recorded trace backs a socket-free replay test that reruns the captured wire
conversation through the fabric directly — a fast, dependency-free CI guard that
needs neither the tt-metal toolchain nor the live server:

- **Blackhole** — `driver/blackhole/server/<name>_replay_test.py`: pumps the
  kernel to completion and checks the DRAM result values.
- **Wormhole** — `driver/wormhole/server/examples_replay_test.py`: replays every
  example trace and asserts each host READ reply reproduces bit-for-bit.

Recapture the traces after a tt-metal bump with the `--record` / `capture_traces.sh`
commands above.

## The examples

Each `<name>/src/` is a host program (`<name>.cpp`), a `CMakeLists.txt`, and a
`kernels/` tree.

* **one** — BRISC only: read two vectors from DRAM, add on the RISC-V core, write back.
* **two** — like one, but NCRISC writes the result back, so a circular buffer bridges BRISC and NCRISC.
* **three** — like two, but chunked (loops over fixed-size tiles through a single-page CB).
* **four** — the matrix unit (FPU) does the elementwise add via the ELWADD path (Int8 in, Int32 out).
* **four-fp** — like four, but Float32 in/out.
* **five** — like four, but the vector unit (SFPU) performs the add (Int32).
* **five-fp** — floating-point variant of five.
* **six** — single-core 128³ bf16 matmul on the matrix unit, validated against a CPU golden by Pearson correlation (bf16 + HiFi4 isn't bit-exact).
* **eight** — elementwise add on BRISC only, issuing its two DRAM reads with distinct NoC transaction IDs and barriering on them out of order.
* **nine** — two-tile: a producer tile runs reader+compute+sender, a consumer tile runs the writer, with a CB bridged across tiles over the NoC (needs the two-tile coords above).
* **pipestall** — two-tile, and the only example whose *point* is timing: `nine` plus a reverse credit semaphore, so the producer's Tensix backs up behind the consumer core. Three environment knobs (`PIPESTALL_DELAY`, `PIPESTALL_CREDITS`, `PIPESTALL_OUT_DEPTH`) set how long the producer's unpacker legitimately blocks. It is the workload the per-unit stall detector's threshold is calibrated against — see `tt_sim/device/deadlock.py`.
* **loopback** — Int32 copy DRAM→DRAM through the TRISC/pack path (`copy_tile` → `pack_tile`), chunked.

## Writing a new example

Create `examples/<name>/src/` with:

1. `<name>.cpp` — a tt-metal host program. Include `<tt-metalium/host_api.hpp>` /
   `<tt-metalium/device.hpp>` / `<tt-metalium/tt_metal.hpp>`, launch with
   `tt::tt_metal::detail::LaunchProgram` (slow dispatch is the only modelled path),
   validate on the host, and `return` non-zero on mismatch.
2. `CMakeLists.txt` — copy one from an existing example; only the `project`/target
   name changes.
3. `kernels/` — data-movement and compute kernels. For this tt-metal release:
   data-movement kernels do **not** include `dataflow_api.h` (force-included);
   compute kernels include `api/compute/<name>.h` and use a plain
   `void kernel_main() { ... }` entry point.

Then add it (with its per-arch coords) to `examples_test.py` /
`run_examples.sh` / `capture_traces.sh` to fold it into the suites.
