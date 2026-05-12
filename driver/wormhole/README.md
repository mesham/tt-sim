# Wormhole simulator driver

This is the Wormhole driver, and probably what people will use the most. There is the Wormhole setup itself provided, along with a range of code examples that can be run and modified. You can also build your own kernels and run them too. 

## Provided examples

All the examples are run in the same way as illustrated here for the first one (see [Getting Started](#getting-started)), executed from this base _wormhole_ directory due to the paths being relative to here.

* [one](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/one) is what we explore in the README furtheron here, this is using BRISC only to read data from DRAM, perform addition, and write results back to DRAM.
* [two](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/two) is similar to the first example, but uses NCRISC to write results back to DRAM. This therefore involves a circular buffer between BRISC and NCRISC.
* [three](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/three) is similar to the second example, but is chunking up data and operating upon individual chunks passing these then to NCRISC.
* [loopback](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/loopback) brings in TRISC cores and the Tensix unit to copy data from the CBs into a segment of _dst_ and then copy this out to another CB. This tests the unpackers and packers, and all the associated functionality.
* [four](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/four) uses the matrix unit in the Tensix coprocessor to do the addition of examples one to three, via the _ELWADD_ instruction. In the Tensix co-processor data is unpacked to _srcA_ and _srcB_, then element wise addition is undertaken and results are in _dst_ which are then packed to L1.
* [four-fp](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/four-fp) is the same as the fourth one, but uses FP32 as the input and output type instead, with the FPU computing with FP16.
* [five](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/five) similar to the fourth example, but uses the vector unit (SFPU) to do the element wise addition. In the Tensix coprocessor data is unpacked to different rows of _dst_, the SFPU executes the _SFPIADD_ instruction storing results to rows of _dst_, with the packer then copying these into L1.
* [five-fp](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/five-fp) similar to the fifth example, but uses floating point numbers. The _base_ example uses FP32 throughout (the SFPU computes with this), there is also a version where the SFPU computes with TF32 and FP16.

## Getting started

You first need to add the root of the repository to your _PYTHONPATH_

```bash
export PYTHONPATH=~/tt-sim:$PYTHONPATH
```

Then from within this _wormhole_ directory you can run an example (note it takes a few seconds to run, this is startup overhead parsing YAML configuration files):

```bash
~/tt-sim/driver/wormhole $ python3 one/one.py
--> Launching and running firmware
    --> Done, device is ready
--> Launching and running kernel
    --> Done
Example one completed successfully
```

If you look into the [one/one.py](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/one.py) file you will see that this simple example sets things up, launches the firmware to initialise the cores, writes some numbers to the DRAM, runs the kernel, and then checks that the resulting value is the element wise sum of the input values. This is using the RISC-V BRISC core to do the addition in this first example.

The source code of this example is [here](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/kernels/dataflow/read_kernel.cpp) and this has been built into the _brisc_kernel.bin_ file in the [one example directory](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/one). The [parameters.json](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/parameters.json) file provides the values for device mailboxes, runtime arguments, circular buffers etc. Normally this is provided by tt-metal via the host executable at runtime, but we don't yet have integration with tt-metal so is provided manually.

>**NOTE:**  
> For these examples you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.

### Exploring more

The simulator runs until it detects completiong (this is done [here](https://github.com/mesham/tt-sim/blob/9280e5e935f83de0565876e9e57c5367fa77d80b/driver/wormhole/wormhole_driver.py#L79)). The nice thing about Python is that it is trivial to script up exploring and manipulating state, during testing I have found it very useful to run for _n_ cycles only and then through the objects grab out values of registers and memory addresse. 

In each example you can set diagnostic settings, these are off by default (turning all of them on results in lots of output!) For example, change _brisc_diagnostics=False_ to _brisc_diagnostics=True_ and _issued_instructions=False_ to _issued_instructions=True_ a few lines higher up in _one.py_. If you rerun you will see the BRISC babycore providing diagnostic information about every instruction it executes and the Tensix co-processor reporting instructions that it has issued to the backend (whilst this first example does not use the Tensix co-processor the firmware still issues some instructions at about 1700 cycles in to set it up). For example, looking about half way you will see something like:

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

Taking the first line as an example, _[0-> 1690]_ denotes this is Baby RISC-V core 0 (BRISC) at cycle number 1690. _[0x478c]_ is the value of the PC (i.e. the address of the instruction being executed), with the instruction itself and some meta data. The _Issued_ messages are from diagnostics reported by the Tensix coprocessor, here for example the firmware is issuing some instructions to the MATH and vector unit to set them up. You can enable other diagnostics via the boolean values, for example _configurations_set_ reports all values set in the Tensix configuration unit. As an aside, it is fine to omit these diagnostic settings (it just assumes they are all off), but they are included in each example explicitly for convenience.

### Structured tracing (JSONL and Perfetto)

The diagnostics above are stderr text for human reading. For machine-readable output suitable for downstream tooling, set one or both env vars:

```bash
TT_SIM_TRACE=/tmp/run.jsonl python3 one/one.py            # JSONL — every event, one per line
TT_SIM_TRACE_PERFETTO=/tmp/run.json.gz python3 one/one.py  # Chrome Trace Event Format for ui.perfetto.dev
```

The JSONL form streams every event (instruction retirement, Tensix dispatch, NoC request/response, mem accesses, sync points, kernel lifecycle) into a file plus a `*.ids.json` sidecar mapping `unit_id` tuples back to `(chip_id, core_y, core_x, unit)`. The Perfetto form writes a zoomable timeline file you can drag onto [ui.perfetto.dev](https://ui.perfetto.dev) — NoC transactions render as arrows between source and destination NUIs (flow events), and lifecycle events appear as global markers. Both writers can be enabled simultaneously and subscribe independently.

Full schema, design principles, and canned Perfetto SQL queries live in [tt_sim/trace/README.md](../../tt_sim/trace/README.md). Phases 1–2 of the broader tracing plan in [ROADMAP §H](../../ROADMAP.md) are now complete; later phases add Spike-compatible commitlog, Parquet, Cachegrind, and LCOV writers as additional consumers of the same bus.

### Firmware and kernel launching

Before launching kernels tt-metal runs firmware on each of the Tensix tiles which reset them, set up the different components etc. The [firmware](https://github.com/mesham/tt-sim/tree/main/driver/wormhole/firmware) directory contains these binaries which are taken unmodified from that generated by tt-metal when a kernel is launched. These are all built from tt-metal source code in [tt_metal/hw/firmware](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/firmware). 

It should be highlighted that tt-metal has two flows when launching kernels, a direct approach and a command queue approach. We currently only support the simpler, direct, approach. By contrast the command queue approach uses two Tensix tiles to marshal and control the execution of kernels across other Tensix tiles, and for simplicity we have avoided supporting this so far. Therefore kernels need to be launched via tt-metal using the _tt:tt_metal::detail::LaunchProgram_ API call.

### Enabling diagnostics in the tt-metal flow

The diagnostic flags shown above (_brisc_diagnostics=True_, _issued_instructions=True_, etc.) are constructed in Python at the top of each standalone example. When tt-metal launches the simulator the user does not own the entry point — UMD spawns _run.sh_ — so the same controls are exposed via _TT_SIM_DIAG_*_ environment variables. All default to off; set any of them to a truthy value (_1_, _true_, _yes_, _on_, case-insensitive) to turn the matching diagnostic on. Output goes to stderr, alongside the other server log messages.

Per baby-RISC-V core (instruction trace):

| Env var | Field |
| --- | --- |
| _TT_SIM_DIAG_BRISC_  | _brisc_diagnostics_ |
| _TT_SIM_DIAG_NCRISC_ | _ncrisc_diagnostics_ |
| _TT_SIM_DIAG_TRISC0_ | _trisc0_diagnostics_ |
| _TT_SIM_DIAG_TRISC1_ | _trisc1_diagnostics_ |
| _TT_SIM_DIAG_TRISC2_ | _trisc2_diagnostics_ |

Per NoC (transaction trace):

| Env var | Field |
| --- | --- |
| _TT_SIM_DIAG_NOC0_ | _noc0_diagnostics_ |
| _TT_SIM_DIAG_NOC1_ | _noc1_diagnostics_ |

Tensix coprocessor:

| Env var | Field |
| --- | --- |
| _TT_SIM_DIAG_CO_ISSUED_ | _issued_instructions_ |
| _TT_SIM_DIAG_CO_CONFIG_ | _configurations_set_ |
| _TT_SIM_DIAG_CO_UNPACK_ | _unpacking_ |
| _TT_SIM_DIAG_CO_PACK_   | _packing_ |
| _TT_SIM_DIAG_CO_FPU_    | _fpu_calculations_ |
| _TT_SIM_DIAG_CO_SFPU_   | _sfpu_calculations_ |
| _TT_SIM_DIAG_CO_THCON_  | _thcon_ |

Aggregates (shortcuts that fan out to several of the individual flags above):

| Env var | Expands to |
| --- | --- |
| _TT_SIM_DIAG_TRISC_ | TRISC0 + TRISC1 + TRISC2 |
| _TT_SIM_DIAG_NOC_   | NOC0 + NOC1 |
| _TT_SIM_DIAG_CO_    | every CO\_\* flag |
| _TT_SIM_DIAG_ALL_   | every individual flag |

Individual vars win over aggregates when both are present, so you can use an aggregate to turn things on broadly and then mask specific flags back off. Examples:

```bash
# Trace BRISC instructions only.
TT_SIM_DIAG_BRISC=1 \
TT_METAL_SIMULATOR=$(pwd)/driver/wormhole \
    <your tt-metal program>

# Everything on except NCRISC.
TT_SIM_DIAG_ALL=1 TT_SIM_DIAG_NCRISC=0 \
TT_METAL_SIMULATOR=$(pwd)/driver/wormhole \
    <your tt-metal program>
```

The server prints a one-line summary at startup (e.g. _[server] diagnostics: BRISC, CO_FPU_, or _none_ when nothing is enabled) so you can confirm the variables were picked up.

### Deadlock detection

Some kernels and bits of firmware can wedge — a NoC read with no matching response, a circular buffer counter that never gets bumped, a Tensix backend instruction that never completes, a mailbox read with nothing on the other end. Because each driver script polls a go-signal mailbox via `wormhole.run(100)` in a tight loop until it sees _RUN_MSG_DONE_, a wedge of this kind shows up as a silent hang of the Python process with no indication of what went wrong.

The simulator runs a per-cycle progress watchdog by default. If nothing observable has changed for a configured number of cycles (default 50000), a multi-line _[DEADLOCK]_ block is printed to stderr describing what it can see. The watchdog does not stop the simulation — it simply warns and keeps running, and re-prints once per window for as long as the stall continues, so a long stall is visible without flooding output. It is dormant while every BabyRISC-V core is in soft reset (the normal state before firmware launch).

The signals that the watchdog tracks each cycle are:

* The program counter of every BabyRISC-V core that is out of reset (frozen, oscillating, or moving through a wider range).
* The NoC counters on every NUI of the Tensix tile and the DRAM tiles (read requests sent and outstanding, write requests outgoing).
* The Tensix coprocessor's three frontends and the busy-state of its backend on each thread.
* The unknown-instruction count on each core (in case it has jumped into data).

When it fires, the report names the responsible component. A frozen-PC stall (most often a `MemoryStall` on a mailbox or `ttsync` wait gate) looks like:

```
[DEADLOCK cycle=50000] no observable progress for 50000 cycles
  BRISC: frozen at 0x4b80 (memory stall or `j .`)
  (TT_SIM_DEADLOCK=0 to disable, TT_SIM_DEADLOCK_THRESHOLD=N to tune window)
```

A NoC stall where a read request was issued but the response never arrived looks like:

```
[DEADLOCK cycle=50000] no observable progress for 50000 cycles
  NCRISC: oscillating in [0x12010, 0x12018] (3 unique PCs) — likely polling
  NoC tile=(18, 18) nui=0: 1 read(s), 0 write(s) outstanding (1 unresolved)
  (TT_SIM_DEADLOCK=0 to disable, TT_SIM_DEADLOCK_THRESHOLD=N to tune window)
```

The two env vars below control the watchdog. Both apply equally to the standalone _python3 one/one.py_-style flow and to the tt-metal server flow, so the same controls work in both places.

| Env var | Effect |
| --- | --- |
| _TT_SIM_DEADLOCK_ | Set to a falsy value (_0_, _false_, _no_, _off_) to disable the watchdog entirely. On by default. |
| _TT_SIM_DEADLOCK_THRESHOLD_ | Integer number of cycles of no observable progress before a warning fires. Defaults to _50000_; raise it if a long-running compute loop trips a false positive, lower it to surface stalls sooner. |

## Building your own kernels to run

The simulator is not currently integrated with tt-metal, although that could be fairly easy to do in the future, so this is a fairly manual (albeit rather simple) process. Before we explain how to do this, it's useful to understand tt-metal support provided here.

### tt-metal abstraction

In [tt_metal.py](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/tt_metal.py) we abstract specifics around host and device integration at the low level. The most important part of this is the memory map in L1, which is understood by the host and device, and how these communicate with each other and between RISC-V cores within a Tensix unit. Whilst we currently support release 0.62.2 of tt-metal, we have abstracted specifics around the memory map into [tt_metal_0.62.2.json](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/tt_metal_0.62.2.json) which should make it easier to update to future releases. It should be stressed that this is entirely agnostic to the hardware itself and is very much a software thing, hence providing it here rather than in the core simulator.

When a driver script loads it will read in the JSON configuration file and use this to determine the memory layout on the device, as well as some constants that are understood by the firmware. If you look at one of the example parameter files (e.g. [one/parameters.json](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/parameters.json)) you will see that the keys in the parameters file of _kernel_config_ and _go_message_ match those of _kernel_config_msg_t_ and _go_msg_t_ in _tt_metal_0.62.2.json_. That's really a major point of the _tt_metal.py_ script here, it calculates the mapping between the input parameter values and the location where these need to be stored within L1 memory of the Tensix tile. 

There are also a few other things, such as setting up mappings in L1 between DRAM banks and tiles for both NoCs etc, but these are fairly simple. The [wormhole_driver.py](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/wormhole_driver.py) script is then a high level support script that will launch and run the firmware in one method, and the kernel in another method. This really just saves lots of duplicate code between the examples.

### Building via tt-metal

Therefore to program the simulator with your own kernels you need to provide your own driver script (e.g. [one/one.py](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/one/one.py)) which is problem specific and should be fairly simple based on what is already provided. You also need to provide a _parameters.json_ and the kernel binary files and for that we need to invoke tt-metal.

We have forked tt-metal [here](https://github.com/mesham/tt-metal) providing release version 0.62.2 but also with some additional diagnostic information reported when kernels are run, and we will use that information to prepare the _parameters.json_ file and grab the binaries.

Therefore, as normal run your host code to invoke the JIT builder, there is a lot of output but if you work up from the bottom you care about the kernel specific output:

```bash
~/tt-example/$ ./my_host
.....
==== 4 runtime arguments ==== 
Arg 0: 0x20
Arg 1: 0x420
Arg 2: 0x100
Arg 3: 0x40
Write RTA to 0x7190
==== 3 runtime arguments ==== 
Arg 0: 0x820
Arg 1: 0x100
Arg 2: 0x40
Write RTA to 0x71a0
==== 2 runtime arguments ==== 
Arg 0: 0x100
Arg 1: 0x40
Write RTA to 0x71ac
Allocate CB 256, 0x185a0
Allocate CB 256, 0x185a0
Allocate CB 256, 0x185a0
Write binary /home/user/.cache/tt-metal-cache/d279aa36be/4098/kernels/read_kernel/10818148161270680775//brisc/brisc.elf of length 0x3fc to addr 0x71f0
Launch
Write binary /home/user/.cache/tt-metal-cache/d279aa36be/4098/kernels/write_kernel/6534090200371868679//ncrisc/ncrisc.elf of length 0x358 to addr 0x75f0
Launch
Write binary /home/user/.cache/tt-metal-cache/d279aa36be/4098/kernels/compute_kernel/3362008294476569189//trisc0/trisc0.elf of length 0x71c to addr 0x7950
Write binary /home/user/.cache/tt-metal-cache/d279aa36be/4098/kernels/compute_kernel/3362008294476569189//trisc1/trisc1.elf of length 0x448 to addr 0x8070
Write binary /home/user/.cache/tt-metal-cache/d279aa36be/4098/kernels/compute_kernel/3362008294476569189//trisc2/trisc2.elf of length 0x538 to addr 0x84c0
=== CB on core ===
CB 0 0x0: 0x185a0
CB 0 0x1: 0x100
CB 0 0x2: 0x1
CB 0 0x3: 0x100
=== CB on core ===
CB 1 0x4: 0x186a0
CB 1 0x5: 0x100
CB 1 0x6: 0x1
CB 1 0x7: 0x100
=== CB on core ===
CB 2 0x8: 0x187a0
CB 2 0x9: 0x100
CB 2 0xa: 0x1
CB 2 0xb: 0x100
Write CB config to addr 0x71c0 size=48
watcher_kernel_ids: 0x4 0x5 0x6
---- Launching '(x=18,y=18)' 0x20 ------ 
[0x0] watcher_kernel_ids: 0x4 0x5 0x6
[0x6] ncrisc_kernel_size16: 0x36
[0x8] kernel_config_base: 0x7190 0x3f520 0x73d0
[0x14] sem_offset: 0x30 0x0 0x0
[0x1a] local_cb_offset: 0x30
[0x1c] remote_cb_offset: 0x60
[0x1e] rta_offset 0: 0x0 0x30
[0x22] rta_offset 1: 0x10 0x30
[0x26] rta_offset 2: 0x1c 0x30
[0x2a] mode: 0x1
[0x2c] kernel_text_offset: 0x60 0x460 0x7c0 0xee0 0x1330
[0x40] local_cb_mask: 0x7
[0x44] brisc_noc_id: 0x0
[0x45] brisc_noc_mode: 0x0
[0x46] min_remote_cb_start_index: 0x20
[0x47] exit_erisc_kernel: 0x0
[0x48] host_assigned_id: 0x0
[0x4c] sub_device_origin_x: 0x0
[0x4d] sub_device_origin_y: 0x0
[0x4e] enables: 0x7
[0x4f] preload: 0x0
-> [0x2a0] GO: dispatch_message_offset: 0x1 master_x: 0x0 master_y: 0x0 signal: 0x80
Sent go message
~/tt-example/$
```

There is quite a bit here, but if we start from the top you can see it provides the location of each kernel elf file, its size and location that it is written to in memory, then the runtime arguments (for BRISC, NCRISC, and then the TRISCs), the CB configuration settings, and then the mailbox and go message. The values shown above are the same as in [five/parameters.json](https://github.com/mesham/tt-sim/blob/main/driver/wormhole/five/parameters.json) so taking this as an example, create your own  _parameters.json_ file based upon the specific values that have been reported.

You will note that the kernel files reported above are _elf_, whereas we need _bin_ files. In-fact when launching kernels tt-metal will extract the binary from the elf, so we need to do the same but a little more manually. When tt-metal builds it downloads the GCC RV32 toolchain, so you already have this, and we can use _objcopy_ to extract the binary from the elf. First change into the directory holding the elf and then execute _objcopy_

```bash
~/tt-example/$ cd /home/user/.cache/tt-metal-cache/d279aa36be/4098/kernels/read_kernel/10818148161270680775//brisc
.../$ ~/tt-metal/runtime/sfpi/compiler/bin/riscv32-tt-elf-objcopy -I elf32-little brisc.elf -O binary brisc.bin
```

You need to do this for each of the elf files, brisc, ncrisc, trisc0, trisc1 and trisc2. Note that I am assuming your tt-metal install is at the top level of your home directory here, this might need to be tweaked depending on where you have located it.

Once you have done these steps you should be able to run your kernel on the simulator.
