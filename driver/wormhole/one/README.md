# Example one

This example is very simple and uses BRISC only to read data from DRAM, perform addition, and write results back to DRAM. But crucially it tests and demonstrates quite a few subsystems of the simulator. The script will check that the computed values are correct.

```bash
~/tt-sim/driver/wormhole $ python3 one/one.py
--> Launching and running firmware
    --> Done, device is ready
--> Launching and running kernel
    --> Done
Example one completed successfully
```

>**NOTE:**  
> For this example you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.

## Contents

* [src](src) is the source code for the host and device kernels
* [binaries](binaries) are the binaries (built from the device kernels in [src](src)) that will be deployed to the simulator
