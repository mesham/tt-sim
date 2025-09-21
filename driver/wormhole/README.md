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

### Exploring more

The simulator runs until it detects completiong (this is done [here](https://github.com/mesham/tt-sim/blob/9280e5e935f83de0565876e9e57c5367fa77d80b/driver/wormhole/wormhole_driver.py#L79)). The nice thing about Python is that it is trivial to script up exploring and manipulating state, during testing I have found it very useful to run for _n_ cycles only and then through the objects grab out values of registers and memory addresse. 

In [tt_device.Wormhole](https://github.com/mesham/tt-sim/blob/9280e5e935f83de0565876e9e57c5367fa77d80b/tt_sim/device/tt_device.py#L117) you can set diagnostic choices. For example, change the first _False_ to _True_ and _issued_instructions=False_ to _issued_instructions=True_ a few lines down. If you rerun you will see the BRISC Babycore providing diagnostic information about every instruction it executes and the Tensix co-processor reporting instructions that it has issued to the backend (whilst this first example does not use the Tensix co-processor the firmware still issues some instructions at about 1700 cycles in to set it up). For example, looking about half way you will see something like:

```bash
[0-> 1690][0x478c] jalr zero, 0x0(ra)    # jump to 0x39b8
[0-> 1691][0x39b8] addi a5, zero, 0x1f    # a5 = zero + 0x1f
[0-> 1692][0x39bc] sw a5, 0x274(s0)    # mem[0xffef0274] = a5
[0-> 1693][0x39c0] lui a5, 0x10180000    # a5 = 0x10180000
[0-> 1694][0x39c4] sw a5, 0x0(s7)    # mem[0xffe40000] = a5
Issued ZEROACC to MATH from thread 0
[0-> 1695][0x39c8] lui a5, 0x8a003000    # a5 = 0x8a003000
[0-> 1696][0x39cc] addi a5, a5, 0xa    # a5 = a5 + 0xa
[0-> 1697][0x39d0] sw a5, 0x0(s7)    # mem[0xffe40000] = a5
Issued SFPENCC to SFPU from thread 0
[0-> 1698][0x39d4] lui a5, 0x2000000    # a5 = 0x2000000
[0-> 1699][0x39d8] sw a5, 0x0(s7)    # mem[0xffe40000] = a5
Issued NOP to NONE from thread 0
[0-> 1700][0x39dc] lui a5, 0x7100c000    # a5 = 0x7100c000
[0-> 1701][0x39e0] addi a5, a5, -0x80    # a5 = a5 + -0x80
[0-> 1702][0x39e4] sw a5, 0x0(s7)    # mem[0xffe40000] = a5
Issued SFPLOADI to SFPU from thread 0
[0-> 1703][0x39e8] lui a5, 0x91000000    # a5 = 0x91000000
[0-> 1704][0x39ec] addi a5, a5, 0xb0    # a5 = a5 + 0xb0
[0-> 1705][0x39f0] sw a5, 0x0(s7)    # mem[0xffe40000] = a5
Issued SFPCONFIG to SFPU from thread 0
[0-> 1706][0x39f4] lw a5, 0xc(s0)    # a5 = mem[0xffef000c]
```

Taking the first line as an example, _[0-> 1690]_ denotes this is Baby RISC-V core 0 (BRISC) at cycle number 1690. _[0x478c]_ is the value of the PC (i.e. the address of the instruction being executed), with the instruction itself and some meta data. The _Issued_ messages are from diagnostics reported by the Tensix coprocessor, here for example the firmware is issuing some instructions to the MATH and vector unit to set them up,
