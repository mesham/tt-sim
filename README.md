# Simulator for Tenstorrent architecture

This is a software simulator for the Tenstorrent architecture in large part based upon the [Tenstorrent ISA Documentation](https://github.com/tenstorrent/tt-isa-documentation/tree/main). It currently provides an (incomplete) implementation of the Wormhole, but has been developed so that it can be easily enhanced and support a range of future architectures too. Note that it is not cycle accurate (or intended to be), although this could be added in the future. Instead, the idea was to try and create a software similator from the public documentation to enable developers to experiment with writing TT-Metal code without needing to run on physical hardware.

Optional diagnostic information can be provided by each RISC-V baby core, the NoC, the Tensix co-processor and memory, enabling tracing of the execution of a program. This is currently at the instruction and architectural state level, but could be enhanced in the future to provide feedback to developers around potential code bottlenecks or other issues.

This is written in Python, mainly to make it easy for people to hackaround and experiment with things. If you want to add some functionality, or fix a bug, then please feel free to go ahead and raise a PR. 

## Getting started

The simulator implementation is in the [tt_sim](https://github.com/mesham/tt-sim/tree/main/tt_sim) directory, with the [driver](https://github.com/mesham/tt-sim/tree/main/driver) directory providing a range of examples that illustrate running the simulator. These are individually documented, but to summarise:

* [simple](https://github.com/mesham/tt-sim/tree/main/driver/simple) are very basic examples, demonstrating the memory subsystem and running codes on a vanilla RV32IM CPU.
* [wormhole](https://github.com/mesham/tt-sim/tree/main/driver/wormhole) provides an implementation of a Wormhole, currently with one DRAM tile and one Tensix block (although this is easy to expand, although will likely be slow!) There is some abstraction of TT-Metal (currently v0.62.2 is assumed) provided here too, along with the firmware and example codes that will launch and run on the simulator. This also contains instructions around how to build you own binaries, via TT-Metal, for the simulator.

## Key parts of the simulator

There are a few of key components which are worth highlighting:

* [tt_device.Wormhole](https://github.com/mesham/tt-sim/blob/93da242e8a1a26160afaca43b0772bebc88b9171/tt_sim/device/tt_device.py#L111) creates the Wormhole, currently with a single DRAM tile and a tensix tile. Here you can see the NoC coordinates specified of each, and also the booleans provided to _TensixTile_ are whether to report diagnostic information (the first five for each RISC-V baby core, then next two for each NoC and then the separate Tensix co-processor choices). 
* [tt_device.TensixTile](https://github.com/mesham/tt-sim/blob/93da242e8a1a26160afaca43b0772bebc88b9171/tt_sim/device/tt_device.py#L186) plumbs everything together within a Tensix tile, setting all the memory addresses and ranges for each individual component. These are all based on the ISA documentation [memory map](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/README.md). 
* [tt_noc](https://github.com/mesham/tt-sim/blob/main/tt_sim/network/tt_noc.py) is the implementation of the NoC, it is not yet complete with all the functionality but is sufficient to communicate between tiles and, for example, read and write between DRAM and the Tensix tile.
* [rv](https://github.com/mesham/tt-sim/tree/main/tt_sim/pe/rv) provides the RV32IM implementation with [.ttinsn extension](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/PushTensixInstruction.md#ttinsn-instruction-set-extension). This is fairly self explanatory, providing a pluggable approach to combining ISAs. The [BabyRISCV](https://github.com/mesham/tt-sim/blob/main/tt_sim/pe/rv/babyriscv.py) ties this together for the baby RISC-V cores in the Tensix tile, for instance determining the initial PC value after a soft reset as per [here](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/SoftReset.md).
* [tensix](https://github.com/mesham/tt-sim/tree/main/tt_sim/pe/tensix) is the implementation of the Tensix coprocessor as per [here](https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor). This is not fully complete, but enough to run a range of codes that use the matrix, vector and scalar units (and all the associated unit implementation required to enable this). 
