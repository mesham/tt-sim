# Simulator for Tenstorrent architecture

This is a software simulator for the Tenstorrent architecture in large part based upon the [Tenstorrent ISA Documentation](https://github.com/tenstorrent/tt-isa-documentation/tree/main). It implements **Wormhole** and **Blackhole**, both selected by an architecture profile (`tt_sim/arch/`) rather than a fork, so the device, NoC, tiles and the tt-metal wire bridge are all parameterised by architecture. Both run real tt-metal programs unmodified, on the full default worker grid — 72 workers on Wormhole, 130 on Blackhole. Note that it is not cycle accurate (or intended to be). The idea was to create a software simulator from the public documentation, so developers can experiment with writing TT-Metal code without needing physical hardware.

## Contents

- [Installation](#installation)
- [Using tt-sim as a simulator for tt-metal](#using-tt-sim-as-a-simulator-for-tt-metal)
- [Repository overview](#repository-overview)
- [What the timing model claims, and what it does not](#what-the-timing-model-claims-and-what-it-does-not)
- [Key parts of the simulator](#key-parts-of-the-simulator)

Profiling a kernel: [`driver/wormhole/docs/profiling.md`](driver/wormhole/docs/profiling.md)
walks through every output the simulator can produce.
Building tooling on those outputs: [`docs/trace-schema.md`](docs/trace-schema.md)
is the stable, versioned schema — field meanings, units and what you may
rely on.

## Installation

tt-sim is pure Python and requires Python ≥ 3.10. Clone the repository and install it in
editable mode, which pulls in its dependencies (numpy, pyyaml, pynng, flatbuffers,
pyarrow, pyelftools):

```bash
git clone https://github.com/mesham/tt-sim.git
cd tt-sim
pip install -e .           # add [dev] for the ruff + pytest tooling: pip install -e .[dev]
```

This installs the `tt_sim` simulator library. The example drivers live under `driver/`
and are run from the repository root, so keep the repo root on your `PYTHONPATH` when
invoking them directly (the tt-metal flow below wires this up for you automatically):

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Using tt-sim as a simulator for tt-metal

tt-metal's UMD has a "simulation" chip backend: point `TT_METAL_SIMULATOR` at the
[`driver/wormhole`](driver/wormhole) directory and UMD launches tt-sim in place of real
silicon, so an ordinary tt-metal program runs against the simulator over a socket — no
code changes, exactly as it would run on hardware. The minimal setup:

```bash
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal          # your built tt-metal checkout
export TT_METAL_SIMULATOR="$PWD/driver/wormhole"        # the switch: route execution to tt-sim
export TT_METAL_SLOW_DISPATCH_MODE=1                    # the only launch path the sim models
export LD_LIBRARY_PATH="$TT_METAL_RUNTIME_ROOT/build/lib:$LD_LIBRARY_PATH"

# then run any tt-metal program, e.g. an upstream example:
cd "$TT_METAL_RUNTIME_ROOT/build/programming_examples"
./metal_example_add_2_integers_in_compute
```

A set of ready-to-run example programs, plus a build-and-run test suite, lives under
[`examples`](examples). For the full walkthrough — building the
examples, the `TT_SIM_TENSIX_COORDS` grid-pinning knob, diagnostics, structured tracing and
the deadlock watchdog — see **[`driver/wormhole/README.md`](driver/wormhole/README.md)**.

## Repository overview

The simulator implementation is in the [tt_sim](https://github.com/mesham/tt-sim/tree/main/tt_sim) directory, with the [driver](https://github.com/mesham/tt-sim/tree/main/driver) directory providing a range of examples that illustrate running the simulator. These are individually documented, but to summarise:

* [wormhole](https://github.com/mesham/tt-sim/tree/main/driver/wormhole) is where you likely want to go to. It provides an implementation of a Wormhole, currently with one DRAM tile and one Tensix block (although this is easy to expand, although will likely be slow!) It holds a set of example tt-metal programs that are built against a local tt-metal checkout and run against the simulator — exactly as they would run on hardware, only the device is the simulator. Each validates its own results, so the examples double as a test suite. See its README for how to build and run them.
* [blackhole](https://github.com/mesham/tt-sim/tree/main/driver/blackhole) is the Blackhole driver: the same wire bridge and machinery as wormhole (all shared in [`tt_sim/bridge`](https://github.com/mesham/tt-sim/tree/main/tt_sim/bridge)), pointed at a Blackhole device instead. Point `TT_METAL_SIMULATOR` at `driver/blackhole` to route execution there. Workers are materialised on demand by the wire bridge, so a program gets exactly the grid it launches on, up to the full 130; see its README.
* [simple](https://github.com/mesham/tt-sim/tree/main/driver/simple) are very basic examples, demonstrating the memory subsystem and running codes on a vanilla RV32IM CPU.

### What the timing model claims, and what it does not

The simulator carries a cost model of **documented provenance**: every cycle
figure is traceable to the ISA documentation or a vendor source, there are
**zero un-sourced estimates** (asserted by test), and it has been checked
against real silicon. Stated precisely:

> Runs real tt-metal programs unmodified on both architectures, on the full
> default grid, with a timing model of documented provenance, corroborated
> against silicon at the **slope** and **launch** level on both parts, and
> **by mechanism** for NoC-bound work on both parts.

Four caveats belong with that sentence rather than under it, because three of
them are permanent:

- **It is a floor.** Every bound is charged at its low end, by rule, so it
  under-predicts wherever hardware sits above a documented minimum.
- **Mechanism-level checking is one leg of three.** The RV-bound leg is
  Blackhole-only (Wormhole documents no CSRs, so there is no retired-instruction
  count) and currently fails its bar; the Tensix mechanism leg cannot be built
  at all from the counters the hardware exposes.
- **That failure has a single, sourced cause** — integer divide, where the
  documentation gives only a 6–33 cycle range and no algorithm, so a
  magnitude-dependent term cannot be sourced at any provenance.
- **Energy is ranking-level only**, and barely beats a model that knows nothing
  but the cycle count. Absolute joules are out of reach of the instrument.

The full evidence, including what was proven *impossible* rather than merely
left undone, is in [`ROADMAP.md`](ROADMAP.md).

Optional diagnostic information can be provided by each RISC-V baby core, the NoC, the Tensix co-processor and memory, enabling tracing of the execution of a program. This is currently at the instruction and architectural state level, but could be enhanced in the future to provide feedback to developers around potential code bottlenecks or other issues.

This is written in Python, mainly to make it easy for people to hackaround and experiment with things. If you want to add some functionality, or fix a bug, then please feel free to go ahead and raise a PR. This is still work in progress, so also please raise issues etc as you find them!

## Key parts of the simulator

There are a few of key components which are worth highlighting:

* [tt_device.Wormhole](https://github.com/mesham/tt-sim/blob/93da242e8a1a26160afaca43b0772bebc88b9171/tt_sim/device/tt_device.py#L111) creates the Wormhole, currently with a single DRAM tile and a tensix tile. Here you can see the NoC coordinates specified of each, and also the booleans provided to _TensixTile_ are whether to report diagnostic information (the first five for each RISC-V baby core, then next two for each NoC and then the separate Tensix co-processor choices). In the [wormhole](https://github.com/mesham/tt-sim/tree/main/driver/wormhole) tt-metal flow these are set from the `TT_SIM_DIAG_*` environment variables (see that directory's README).
* [tt_device.TensixTile](https://github.com/mesham/tt-sim/blob/93da242e8a1a26160afaca43b0772bebc88b9171/tt_sim/device/tt_device.py#L186) plumbs everything together within a Tensix tile, setting all the memory addresses and ranges for each individual component. These are all based on the ISA documentation [memory map](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/README.md). 
* [tt_noc](https://github.com/mesham/tt-sim/blob/main/tt_sim/network/tt_noc.py) is the implementation of the NoC, it is not yet complete with all the functionality but is sufficient to communicate between tiles and, for example, read and write between DRAM and the Tensix tile.
* [rv](https://github.com/mesham/tt-sim/tree/main/tt_sim/pe/rv) provides the RV32IM implementation with [.ttinsn extension](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/PushTensixInstruction.md#ttinsn-instruction-set-extension). This is fairly self explanatory, providing a pluggable approach to combining ISAs. The [BabyRISCV](https://github.com/mesham/tt-sim/blob/main/tt_sim/pe/rv/babyriscv.py) ties this together for the baby RISC-V cores in the Tensix tile, for instance determining the initial PC value after a soft reset as per [here](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/SoftReset.md).
* [tensix](https://github.com/mesham/tt-sim/tree/main/tt_sim/pe/tensix) is the implementation of the Tensix coprocessor as per [here](https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor). This is not fully complete, but enough to run a range of codes that use the matrix, vector and scalar units (and all the associated unit implementation required to enable this). 
