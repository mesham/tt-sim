# perfbench — cycle-cost measurement programs

A third program tree, alongside `examples/` (functional, arch-agnostic tt-metal
programs) and `optests/` (differential op tests against the vendor reference
simulator). These are **timing** programs: real tt-metal binaries whose output
is device cycle counts, built so that the *same binary* runs on silicon and
against tt-sim and the two can be diffed.

```
perfbench/
├── run.sh              simulator-side runner (arch, coords, venv, cost model)
├── tensixbench/src/    per-instruction Tensix cycle costs
└── riscvbench/src/     the baby RISC-V front end: issue rate, `.ttinsn` push
                        cost, branch cost, instruction fetch
```

The two are complements, and the second exists because of the first's headline
result: `tensixbench` measures what a Tensix unit costs, and found that against
tt-sim **every** probe of **every** unit reads exactly 1.000 cycles because
nothing back-pressures the core that issued it. `riscvbench` measures that core.

| | tensixbench | riscvbench |
| --- | --- | --- |
| **Run it on hardware** | [`tensixbench/README.md`](tensixbench/README.md) | [`riscvbench/README.md`](riscvbench/README.md) |
| **Why it is shaped this way** | [`../docs/plans/tensix-cost-benchmark.md`](../docs/plans/tensix-cost-benchmark.md) | [`../docs/plans/riscv-front-end-benchmark.md`](../docs/plans/riscv-front-end-benchmark.md) |
| **Analyse the results** | `python3 -m tt_sim.perf.tensix_bench_sweep --measured <csv>` | `python3 -m tt_sim.perf.riscv_bench_sweep --measured <csv>` |

Against the simulator:

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh tensixbench -- --blocks 2 --iters 4
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh riscvbench  -- --blocks 2
```

`TT_SIM_ARCH=wormhole` picks the other simulator, `TT_SIM_COST_MODEL=1` turns
the cycle cost model on. Keep the burst sizes small — the simulator runs a few
tens of thousands of cycles per second, where hardware runs a billion. Run
phase B one fidelity at a time (`--phase b --iters 1 --fidelities HiFi2`); two
in the same process do not finish against the simulator.

Every number these programs report is a **slope** over several instruction
counts, so kernel launch, timer overhead and loop setup cancel exactly. No
single absolute measurement is ever reported as a cost, and nothing here writes
to the cost tables.

Both refuse to let a null pass as a confirmation, but they do it differently, and
`riscvbench`'s way is the one worth copying: a run in which every probe reads
exactly `1.000` is simultaneously the expected answer and the signature of a
benchmark that measured nothing, so `riscvbench` singles out four probes with a
documented cost above one cycle, checks them, and refuses the run in those words
if they read 1.000 too.
