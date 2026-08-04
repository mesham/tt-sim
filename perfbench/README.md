# perfbench — cycle-cost measurement programs

A third program tree, alongside `examples/` (functional, arch-agnostic tt-metal
programs) and `optests/` (differential op tests against the vendor reference
simulator). These are **timing** programs: real tt-metal binaries whose output
is device cycle counts, built so that the *same binary* runs on silicon and
against tt-sim and the two can be diffed.

```
perfbench/
├── run.sh              simulator-side runner (arch, coords, venv, cost model)
└── tensixbench/src/    per-instruction Tensix cycle costs
```

| | |
| --- | --- |
| **Run it on hardware** | [`tensixbench/README.md`](tensixbench/README.md) — build, run, what to send back |
| **Why it is shaped this way** | [`../docs/plans/tensix-cost-benchmark.md`](../docs/plans/tensix-cost-benchmark.md) |
| **Analyse the results** | `python3 -m tt_sim.perf.tensix_bench_sweep --measured <csv>` |

Against the simulator:

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh tensixbench -- --blocks 2 --iters 4
```

`TT_SIM_ARCH=wormhole` picks the other simulator, `TT_SIM_COST_MODEL=1` turns
the cycle cost model on. Keep the burst sizes small — the simulator runs a few
tens of thousands of cycles per second, where hardware runs a billion.

Every number these programs report is a **slope** over several instruction
counts, so kernel launch, timer overhead and loop setup cancel exactly. No
single absolute measurement is ever reported as a cost, and nothing here writes
to the cost tables.
