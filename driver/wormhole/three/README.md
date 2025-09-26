# Example three

This example chunks up data and operating upon individual chunks. Each chunk is first read by BRISC which performs element wise addition and uses a circular buffer to send the results for the chunk to NCRISC which write these back to DRAM. When moving onto the Tensix co-processor it is crucial to work in chunks, so this ensure that that is supported by the simulator. The script will check the computed values are correct.

```bash
~/tt-sim/driver/wormhole $ python3 three/three.py
--> Launching and running firmware
    --> Done, device is ready
--> Launching and running kernel
    --> Done
Example three completed successfully
```

>**NOTE:**  
> For this example you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.

## Contents

* [src](src) is the source code for the host and device kernels
* [binaries](binaries) are the binaries (built from the device kernels in [src](src)) that will be deployed to the simulator
