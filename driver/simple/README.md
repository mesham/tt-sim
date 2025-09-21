# Simple examples

These examples demonstrate very basic parts of the simulator, but were useful during development and kept here as they demonstrate flexibility. You first need to add the root of the repository to your _PYTHONPATH_

```bash
export PYTHONPATH=~/tt-sim:$PYTHONPATH
```

Then from within each example directory execute the code, e.g. _python3 ex1.py_ . These all use assertions to check the results and ensure correctness, so are fairly useful as a basic set of tests too.

* [one](https://github.com/mesham/tt-sim/tree/main/driver/simple/ex1) sets up different memories and maps them to ranges in a memory space. Then accesses these different addresses ensuring that data is routed to the correct memory space.
* [two](https://github.com/mesham/tt-sim/tree/main/driver/simple/ex2) runs a very simple binary on the RV32I core which sets a value into a specific memory address. The C code is extremely simple, and it doesn't include any set up or linker script, but tests some basic functionality of the RV32I core.
* [three](https://github.com/mesham/tt-sim/tree/main/driver/simple/ex3) is a little more complex, this is adding integers based on a list of input values. The C code has been compiled with a linker script and assembly to set things up, a makefile is included which builds the binary from the C code.
* [four](https://github.com/mesham/tt-sim/tree/main/driver/simple/ex4) similar to three, but tests RV32IM (specifically the M extension) by a slightly more complex calculation
* [five](https://github.com/mesham/tt-sim/tree/main/driver/simple/ex5) similar to four, but separates out a global (externally visible) memory space from a local (private) CPU only memory space. This is important for the Tenstorrent architecture as each core has it's own mappings.
