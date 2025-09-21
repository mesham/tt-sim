# Wormhole simulator driver

This is the Wormhole driver, and probably what people will use the most. There is the Wormhole setup itself provided, along with a range of code examples that can be run and modified. You can also build your own kernels and run them too. 

## Getting started

You first need to add the root of the repository to your _PYTHONPATH_

```bash
export PYTHONPATH=~/tt-sim:$PYTHONPATH
```

Then from within this _wormhole_ directory you can run an example (note it takes a few seconds to run, this is startup overhead parsing YAML configuration files):

```bash
~/tt-sim/driver/wormhole $ python3 one/one.py
Example one completed successfully
```

If you look into the [one/one.py](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/one.py) file you will see that this simple example sets things up, launches the firmware to initialise the cores, writes some numbers to the DRAM, runs the kernel, and then checks that the resulting value is the element wise sum of the input values. This is using the RISC-V BRISC core to do the addition in this first example.

The source code of this example is [here](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/kernels/dataflow/read_kernel.cpp) and this has been built into the _brisc_kernel.bin_ file in the [one example directory](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/one). The [parameters.json](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/parameters.json) file provides the values for device mailboxes, runtime arguments, circular buffers etc. Normally this is provided by TT-Metal via the host executable at runtime, but we don't yet have integration with TT-Metal so is provided manually.

>**NOTE:**  
> For these examples you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.
