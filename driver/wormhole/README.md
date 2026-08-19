# Wormhole simulator examples

This directory holds a set of example tt-metal programs that run against the Wormhole
simulator. Each example is a real tt-metal host program: it is **built against a local
tt-metal checkout and run exactly the way you would run it on hardware — only the
device is the simulator instead of silicon.** tt-metal's UMD has a "simulation" chip
backend; when `TT_METAL_SIMULATOR` points at this directory the host binary talks to
tt-sim over a socket in place of a real chip. Each program validates its own results on
the host and exits non-zero on mismatch, so the examples double as a test suite.

> **Why not run the examples "directly" any more?** Earlier revisions shipped a
> standalone Python driver per example (`one/one.py`, a hand-written `parameters.json`,
> pre-extracted kernel `.bin` files) that drove the simulator without tt-metal. That
> flow was pinned to one tt-metal release (its L1 memory map was baked into a JSON) and
> was not tenable across tt-metal versions. It has been removed. The examples now build
> against whatever tt-metal you point them at, and the release-specific layout comes
> from the host binary over the wire.

## Layout

```
driver/wormhole/
├── tests/
│   └── capture_traces.sh   build + run every shared example, record its wire trace
├── server/                 the wire-bridge server UMD talks to (run.sh spawns it)
│   ├── examples_replay_test.py  byte-identical offline replay of every example trace
│   ├── offline_replay_test.py   byte-identical replay of one.trace (the pinned baseline)
│   ├── sfpumath_replay_test.py  optests/sfpumath vs the ttsim-Wormhole golden
│   └── traces/             one recorded wire trace per example (+ optests/sfpumath)
├── run.sh                  UMD entry point → `python -m driver.wormhole.server`
├── soc_descriptor.yaml     the Wormhole worker/DRAM grid UMD sees
├── replay.py               wire-trace replayer (used by server regression tests)
└── docs/                   profiling walkthrough

The arch-agnostic example *sources* live in examples/ (shared with
Blackhole); the Wormhole live runner is examples/examples_test.py.
```

> **Building and running the examples** is the same across both simulators and is
> documented once in **[examples/README.md](../../examples/README.md)** (CMake
> build, the run environment, per-arch coordinates, the suite runners, and the
> offline replay guards). This file covers the **Wormhole-driver specifics** on
> top of that: the environment switch, structured tracing, per-component
> diagnostics, and deadlock detection.

## Requirements

- A built tt-metal checkout that exports the `TT-Metalium` CMake package (i.e. a
  `.../<build>/lib/cmake/tt-metalium/tt-metalium-config.cmake` exists — true for a
  normal `build/` or `build_Release/`).
- `cmake` and `clang++-17` on `PATH`.

## Environment — driving execution to the simulator

Running an example is ordinary tt-metal: the *only* thing that redirects it from silicon
to tt-sim is `TT_METAL_SIMULATOR`. When it points at this directory, UMD's simulation
backend spawns `run.sh` (which starts the tt-sim server) and the host binary talks to it
over a socket. The variables that matter:

| Variable | Purpose |
| --- | --- |
| `TT_METAL_SIMULATOR` | **The switch.** Path to this `driver/wormhole/` directory; UMD spawns its `run.sh` as the device instead of opening real hardware. |
| `TT_METAL_RUNTIME_ROOT` | tt-metal checkout. Used by CMake to build the example and by the host binary at runtime to locate its kernels/firmware. (`TT_METAL_HOME`, the older name, is accepted as a fallback.) |
| `TT_METAL_SLOW_DISPATCH_MODE=1` | Forces `EnqueueProgram` to fall back to `detail::LaunchProgram` — the only launch path the simulator models. |
| `LD_LIBRARY_PATH` | Must include `<tt-metal>/<build>/lib` so the host binary finds `libtt_metal.so` etc. |
| `TT_SIM_TENSIX_COORDS` | **Optional.** Pins the worker tiles to exactly these, e.g. `1-1` or `1-1,2-1`. Unset, tt-sim materialises whatever the program turns out to use. |
| `TT_METAL_MOCK_CLUSTER_DESC_PATH` | **Optional.** Point it at `driver/wormhole/cluster_descriptor.yaml` to run with **NoC coordinate translation** enabled, the configuration real cards ship in. Read by UMD (it decides which coordinates the host puts on the wire) *and*, through the environment the simulator inherits, by tt-sim itself — so the two ends cannot disagree. Forgetting it against a translated server is a loud error, not a wrong answer. See [§1.4 of the runbook](../../docs/running-tt-metal-on-the-simulator.md). |

The project venv sets `TT_METAL_RUNTIME_ROOT`, `TT_METAL_SIMULATOR`,
`TT_METAL_SLOW_DISPATCH_MODE=1` and `LD_LIBRARY_PATH` for you. Without it, export them
yourself:

```bash
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal
export TT_METAL_SIMULATOR="$(git rev-parse --show-toplevel)/driver/wormhole"
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$TT_METAL_RUNTIME_ROOT/build/lib:$LD_LIBRARY_PATH"
```

See [docs/running-tt-metal-on-the-simulator.md](../../docs/running-tt-metal-on-the-simulator.md)
for the full runbook, including running the upstream `programming_examples/` and the
diagnostics knobs below.

## Getting started

Activate the venv (which sets `TT_METAL_RUNTIME_ROOT`) and build one example with CMake,
then run the binary — no extra flags needed:

```bash
source /path/to/venv/bin/activate

cd examples/one/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/one
# ...tt-metal / UMD log lines...
# Completed successfully on the device, with 100 elements
```

Run the binary **from its `src/` directory** — the host program refers to its kernels
by the relative path `kernels/...`, which tt-metal resolves against the current
directory. Multi-tile programs need no configuration: tt-sim materialises the worker
tiles a program launches on (and any a peer sends NoC traffic to) as it discovers them —
see the runbook above. `TT_SIM_TENSIX_COORDS=<physical coords>` still *pins* the set
exactly, which is what the replay guards want and what you want if you are deliberately
starving a program of cores.

## Running the examples as a test suite

[`examples/examples_test.py`](../../examples/examples_test.py) builds every example and runs it
against the simulator, asserting each exits 0 and prints its success line. It follows the
repo's `*_test.py` convention, so it runs either standalone or under pytest, and skips
cleanly when the tt-metal build environment is absent:

```bash
python3 -m examples.examples_test   # prints PASS/FAIL per example
pytest examples/examples_test.py -v  # same, under pytest
```

That harness needs the built tt-metal toolchain (it compiles and launches each
example live). For a fast, dependency-free CI guard there is also **offline
replay**: [`server/examples_replay_test.py`](server/examples_replay_test.py)
replays a recorded wire trace per example (`server/traces/<name>.trace`) and
asserts every host READ reply reproduces bit-for-bit — the Wormhole analogue of
Blackhole's `server/*_replay_test.py`. Recapture the traces after a tt-metal
bump with [`tests/capture_traces.sh`](tests/capture_traces.sh).

Some guards are not examples but `optests/` programs, replayed the same way and
checked on **values** against a frozen ttsim-Wormhole dump
(`traces/<name>.expected`) so the vendor reference stays pinned with no oracle
in the loop: [`server/sfpumath_replay_test.py`](server/sfpumath_replay_test.py)
(FP32 SFPU: recip / tanh / sigmoid) and
[`server/softplus_replay_test.py`](server/softplus_replay_test.py) (upstream's
`sfpu_eltwise_chain` — exp → SFPU add → log on a bfloat16 Dst, with the ones
tile built on-device by 16-bit BRISC stores). Recapture either with
`TT_SIM_ARCH=wormhole TT_SIM_RECORD=driver/wormhole/server/traces/<name>.trace
./optests/diff.sh <name>` (and refresh `.expected` from that run's oracle
output) — never to make the test pass, only when the trace itself is stale.

A third family replays tt-metal's **upstream** `programming_examples/`:
[`server/noc_tile_transfer_replay_test.py`](server/noc_tile_transfer_replay_test.py)
moves one `uint16` tile DRAM → core (1,1) L1 → NoC → core (1,2) L1 → DRAM
behind a cross-core semaphore, and asserts the program's own verdict
(`Result = 14 : Expected = 14`) on all 1024 elements. Blackhole carries the
same guard plus three more (`vecadd_sharding`, `pad_multi_core`,
`shard_data_rm`). Recapture any of them with
[`../tests/capture_upstream_traces.sh`](../tests/capture_upstream_traces.sh),
which holds the binary names, worker coordinates and success lines.

## The examples

The sources and a one-line description of each live in
[examples/README.md](../../examples/README.md#the-examples). Of the bundled set only
`nine` launches on two tiles, and it needs nothing said about it — the second tile
appears when the program launches on it. If you do pin the set by hand, `nine` wants
`TT_SIM_TENSIX_COORDS=1-1,2-1`; a bare `TT_SIM_TENSIX_CORES=2` pins `1-1,1-2` instead,
so `nine`'s launch on `2-1` hits an unmaterialised tile and the server names it and
stops the run.

All eleven examples currently pass. The Tensix coprocessor in the simulator is still
incomplete, though, so a future example may exercise a gap the simulator hasn't modelled;
a compute gap crashes the simulator server, which then stops the host too rather than
leaving it blocked in `recv` (`tt_sim/bridge/hostlink.py`), so an unmodelled op shows up
as a prompt signal-15 exit with the simulator's traceback beside it, not as a hang. The
`examples_test.py` output tells you which examples pass on your build.

> A former `seven` (a Python-only two-tile smoke test) has moved to
> [`server/multi_tensix_test.py`](server/multi_tensix_test.py); the real two-tile
> firmware+kernel path it stood in for is now exercised end-to-end by `nine`.

## Writing your own example

The `<name>.cpp` / `CMakeLists.txt` / `kernels/` recipe (and the tt-metal kernel
conventions this release uses) is in
[examples/README.md](../../examples/README.md#writing-a-new-example). To fold a new
example into the **Wormhole** suites, add `(name, coords)` to `EXAMPLES` in
`examples/examples_test.py` (`coords` = the physical tiles it launches on, e.g. `"1-1"`)
and to `tests/capture_traces.sh`.

> **Firmware.** tt-metal runs firmware on each Tensix tile before launching a kernel; in
> this flow the host binary streams those firmware binaries to the device over the wire,
> so nothing needs to be staged here. Only the direct launch path
> (`tt::tt_metal::detail::LaunchProgram`) is modelled — the command-queue/fast-dispatch
> flow is not.

## Structured tracing and profiling

Beyond the human-readable diagnostics below, the simulator ships nine `TT_SIM_TRACE_*`
env vars that produce machine-readable output (JSONL, Perfetto, Spike-compatible
commitlogs, Parquet, Cachegrind, LCOV, invariant violations, state dumps). All work in
this tt-metal-driven flow — UMD inherits the env, `run.sh` inherits it, the simulator
inherits it. Quick taster, from `examples/one/src/` after building:

```bash
export TT_METAL_SIMULATOR=$HOME/tt-sim/driver/wormhole
TT_SIM_TRACE_PERFETTO=/tmp/run.json.gz ./build/one
# Drag /tmp/run.json.gz onto https://ui.perfetto.dev
```

A full walkthrough — what each output is and which downstream tool reads it — lives in
[docs/profiling.md](docs/profiling.md). Developer-side docs (event schema, adding a
writer) are in [tt_sim/trace/README.md](../../tt_sim/trace/README.md); design history and
what isn't yet modelled is in [ROADMAP §H](../../ROADMAP.md).

## Enabling diagnostics

Because UMD owns the entry point (it spawns `run.sh`), per-component diagnostics are
exposed through `TT_SIM_DIAG_*` environment variables. All default to off; set any to a
truthy value (`1`, `true`, `yes`, `on`, case-insensitive) before running the tt-metal
program. Output goes to stderr alongside the server log.

Per baby-RISC-V core (instruction trace):

| Env var | Field |
| --- | --- |
| `TT_SIM_DIAG_BRISC`  | `brisc_diagnostics` |
| `TT_SIM_DIAG_NCRISC` | `ncrisc_diagnostics` |
| `TT_SIM_DIAG_TRISC0` | `trisc0_diagnostics` |
| `TT_SIM_DIAG_TRISC1` | `trisc1_diagnostics` |
| `TT_SIM_DIAG_TRISC2` | `trisc2_diagnostics` |

Per NoC (transaction trace):

| Env var | Field |
| --- | --- |
| `TT_SIM_DIAG_NOC0` | `noc0_diagnostics` |
| `TT_SIM_DIAG_NOC1` | `noc1_diagnostics` |

Tensix coprocessor:

| Env var | Field |
| --- | --- |
| `TT_SIM_DIAG_CO_ISSUED` | `issued_instructions` |
| `TT_SIM_DIAG_CO_CONFIG` | `configurations_set` |
| `TT_SIM_DIAG_CO_UNPACK` | `unpacking` |
| `TT_SIM_DIAG_CO_PACK`   | `packing` |
| `TT_SIM_DIAG_CO_FPU`    | `fpu_calculations` |
| `TT_SIM_DIAG_CO_SFPU`   | `sfpu_calculations` |
| `TT_SIM_DIAG_CO_THCON`  | `thcon` |

Aggregates (fan out to several individual flags): `TT_SIM_DIAG_TRISC` (all three TRISCs),
`TT_SIM_DIAG_NOC` (both NoCs), `TT_SIM_DIAG_CO` (every `CO_*` flag), `TT_SIM_DIAG_ALL`
(every flag). Individual vars win over aggregates, so `TT_SIM_DIAG_ALL=1
TT_SIM_DIAG_NCRISC=0` means "everything except NCRISC". `TT_SIM_DIAG_CO_SFPU=1` is
particularly handy when a compute result is wrong. The server prints a one-line summary
of the enabled flags at startup.

```bash
# Trace BRISC instructions only:
TT_SIM_DIAG_BRISC=1 ./build/one
```

## Deadlock detection

Some kernels or firmware can wedge — a NoC read with no matching response, a circular
buffer counter that never advances, a Tensix backend instruction that never completes, a
mailbox read with nothing on the other end. Because the host polls the go-signal mailbox
in a tight loop, such a wedge shows up as a silent hang.

The simulator runs a progress watchdog by default. If nothing observable has changed for
a configured number of cycles (default 50000), a multi-line `[DEADLOCK]` block is printed
to stderr describing what it can see (each out-of-reset core's PC, NoC counters, the
coprocessor frontends/backends, and unknown-instruction counts) and names the responsible
component. It warns and keeps running, re-printing once per window, so a long stall is
visible without flooding output. It is dormant while every baby core is in soft reset
(the normal state before firmware launch).

The signature is **sampled** once every `threshold / 8` cycles rather than taken every
cycle — taking it walks every tile, core, NIU and Tensix thread, which on a large grid
cost more than the rest of the simulator put together. A stall that clears the threshold
is then confirmed on 64 consecutive cycles before anything is printed, so a loop that
merely aliases with the sampling interval is not reported. The practical consequence is
that a report arrives up to `threshold / 8 + 64` cycles later than it used to
(50,000–56,314 with the defaults); nothing that used to be detected stops being detected.

| Env var | Effect |
| --- | --- |
| `TT_SIM_DEADLOCK` | Set falsy (`0`/`false`/`no`/`off`) to disable the watchdog. On by default. |
| `TT_SIM_DEADLOCK_THRESHOLD` | Cycles of no observable progress before a warning fires (default `50000`). Raise it if a long compute loop trips a false positive; lower it to surface stalls sooner — the sampling interval is a fixed fraction of it, so a low threshold also samples more often, down to every cycle. |
| `TT_SIM_UNIT_STALL` | Set falsy to disable the per-unit checks (`[UNIT STALL]` **and** `[UNIT WEDGED]`). On by default. |
| `TT_SIM_UNIT_STALL_THRESHOLD` | Consecutive cycles one Tensix backend unit may stay blocked on a single latched instruction before a `[UNIT STALL …]` hint fires (default `10000`). No effect on `[UNIT WEDGED]`, which is not a cycle count. |

The second pair is a *different* check with a different question. The watchdog above
asks "did anything change anywhere"; a wedged Tensix unit answers yes, because nothing
back-pressures the baby RISC-V cores on Tensix instruction issue and the kernel behind
the blocked unit runs to completion regardless (ROADMAP.md, "Unpacker dvalid deadlock").
Any unit exposing `blocked_on()` — today the two unpackers — is picked up at the
watchdog's sampling cadence and then counted **per cycle**, so what reaches the
threshold is one unbroken run and not a loop that happened to be blocked at every
sample point. The default of 10,000 is 2.8x the longest legitimate blocked run measured
anywhere in the tree (3,528 cycles, Wormhole `sfpumath`, with and without the cost
model); zero of the 41 in-tree workloads produce a report.

`[UNIT STALL]` is a **hint** and cannot be anything else — a correct cross-core
pipeline blocks linearly in its downstream consumer's cost with no ceiling
(`examples/pipestall`), so no threshold separates it from a wedge. It says so, prints
once per unit per waited-on instruction, and names the two lines that do settle it:
`[UNIT STALL CLEARED]` (the unit recovered — deep pipeline, nothing wrong) and
`[UNIT WEDGED]`. **`[UNIT WEDGED]` is the authoritative signal**: it reports a unit still
blocked once *every baby core on its tile is in soft reset*, which is a proof rather than
a heuristic — handing a Src bank back takes an instruction and no thread can issue one
from reset — so it has no threshold, fires on reproductions far too short for any cycle
count, and has never fired on an in-tree workload. Full write-up:
[`docs/running-tt-metal-on-the-simulator.md`](../../docs/running-tt-metal-on-the-simulator.md) §4.5.

## NoC alignment checking

Real silicon requires the source and destination addresses of a NoC transfer to
agree in their low bits — a *congruence* requirement, because the NIU rotates the
payload by the shared low-address bits rather than byte-shifting it. Violations
are `UndefinedBehavior`: the transfer is skewed or dropped, it does not fault, so
an unchecked simulator quietly returns wrong data.

tt-sim enforces the rules that
[`WormholeB0/NoC/Alignment.md`](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/Alignment.md)
states and that the vendor reference simulator flags as `UndefinedBehavior`:

| Path | Requirement |
| --- | --- |
| write, L1 → anywhere (incl. multicast) | `(src % 16) == (dst % 16)` |
| read, DRAM → L1 | `(src % 32) == (dst % 32)` on Wormhole, `% 64` on Blackhole |
| read, L1 → L1 | `(src % 16) == (dst % 16)` |
| read, MMIO → L1 | `(src % 4) == (dst % 4)` |

A violation raises `NoCAlignmentError` naming the access path, both addresses and
the required alignment, and stops. The per-arch DRAM modulus lives in
`ArchProfile.noc_dram_read_congruence` (32 Wormhole / 64 Blackhole), matching each
architecture's NoC byte-enable span.

| Env var | Effect |
| --- | --- |
| `TT_SIM_DISABLE_ALIGNMENT_CHECKS` | Set truthy (`1`/`true`/`yes`/`on`) to turn alignment checking off. Off by default, i.e. **checking is on**. |

### Multicast rectangle corner order

A broadcast names its destinations as a `Start` corner and an `End` corner, and
which corner goes in which field depends on the NoC: each NoC's coordinates
increment along its own direction of data flow (NoC 0 rightwards/downwards, NoC 1
leftwards/upwards), so the corners are always written `Start ≤ End` *in that
NoC's own coordinates*. Coordinate translation gives both NoCs one shared range
that increments with NoC 0's flow, so — in the ISA docs' words — "when performing
broadcasts, StartX ↔ EndX need to be swapped by software, and likewise
StartY ↔ EndY" ([`WormholeB0/NoC/Coordinates.md`](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/Coordinates.md)).

| Coordinate space | NoC 0 | NoC 1 |
| --- | --- | --- |
| translated (what tt-metal puts on the wire) | `Start ≤ End` | `Start ≥ End` |
| raw NoC coordinates | `Start ≤ End` | `Start ≤ End` |

tt-metal's `Device::get_noc_multicast_encoding` performs exactly that swap for
`noc_index == 1`, and its watcher enforces the same table. Getting it wrong is
silent on silicon in the worst way: the span wraps the torus rather than being
empty, so the packet reaches tiles the kernel never counted in `num_dests` and
`noc_async_write_barrier` never retires. A violation raises
`MulticastOrderError` naming the NoC, both corners, the required order and the
axis that broke it.

| Env var | Effect |
| --- | --- |
| `TT_SIM_DISABLE_MULTICAST_ORDER_CHECKS` | Set truthy (`1`/`true`/`yes`/`on`) to turn multicast corner-order checking off. Off by default, i.e. **checking is on**. |

## Tensix performance counters

`RISCV_DEBUG_REG_PERF_CNT_*` is modelled (`tt_sim/misc/perf_counters.py`), so a
tt-metal program built with `TT_METAL_PROFILE_PERF_COUNTERS=<bitmask>` programs,
starts, stops and reads the counters exactly as it does on silicon. Bit 5 (`32`)
selects the `INSTRN_THREAD` bank, which is the one tt-sim sources; the other
banks answer their registers but decline their counters.

Counters reported from a quantity tt-sim tracks: `THREAD_STALLS_{0,1,2}`,
`THREAD_INSTRUCTIONS_{0,1,2}`, `WAITING_FOR_NONZERO_SEM_{0,1,2}`,
`WAITING_FOR_NONFULL_SEM_{0,1,2}`, `WAITING_FOR_SRC{A,B}_VALID` and
`WAITING_FOR_SRC{A,B}_CLEAR`, plus `ref_cnt` on every bank. Anything else reads
back as `0` **and prints a warning naming the counter** — the counters are a
functional register model (`vendor_source`), and none of them feeds the cost
model or charges a cycle.

| Env var | Effect |
| --- | --- |
| `TT_SIM_STRICT_PERF_COUNTERS` | Set truthy to make an unmodelled performance-counter read raise instead of warning. |
| `TT_SIM_PERMISSIVE_TILE_CTRL` | Set truthy to downgrade an unmodelled `RISCV_DEBUG_REG` read from a raise back to a one-shot warning returning `0`. Off by default, i.e. **unmodelled reads raise**. |
