# Loopback example

This example reads data from DDR via BRISC, then uses a circular buffer to send to TRISC which unpack to _dst_ via _copy_tile_ (the second segment of _dst_ rows). Then these are packed into L1 memory and a circular buffer sends them to NCRISC which copies to DRAM. This tests running data (without manipulation) through the cores and the Tensix co-processor and was important to get working correctly before other examples which undertake compute on the data within the Tensix co-processor.

```bash
~/tt-sim/driver/wormhole $ python3 loopback/loopback.py
--> Launching and running firmware
    --> Done, device is ready
--> Launching and running kernel
    --> Done
Loopback example completed successfully
```

>**NOTE:**  
> For this example you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.

## Contents

* [src](src) is the source code for the host and device kernels
* [binaries](binaries) are the binaries (built from the device kernels in [src](src)) that will be deployed to the simulator
