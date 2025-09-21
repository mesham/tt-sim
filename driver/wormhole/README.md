# Wormhole simulator driver

This is the Wormhole driver, and probably what people will use the most. There is the Wormhole setup itself provided, along with a range of code examples that can be run and modified. You can also build your own kernels and run them too. 

## Getting started

You first need to add the root of the repository to your _PYTHONPATH_

```bash
export PYTHONPATH=~/tt-sim:$PYTHONPATH
```

Then within this directory you can run an example (note it takes a few seconds to run, this is startup overhead parsing YAML configuration files):

```bash
~/tt-sim/driver/wormhole $ python3 one/one.py
Example one completed successfully
```

If you look into the [one/one.py](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/one.py) file you will see that this simple example sets things up, launches the firmware to initialise the cores, writes some numbers to the DRAM, runs the kernel, and then checks that the resulting value is the element wise sum of the input values. This is using the RISC-V BRISC core to do the addition in this first example.
