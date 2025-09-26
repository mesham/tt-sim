# Example four

This example performs element wise addition of (int8) via uses the matrix unit in the Tensix coprocessor using the ELWADD instruction. In the Tensix co-processor data is unpacked to srcA and srcB, then element wise addition is undertaken and results are stored in dst (in int32) which are then packed to L1. As per previous examples, BRISC loads DRAM data into L1, NCRISC writes results from L1 back to DRAM. The script will check the computed values are correct.

```bash
~/tt-sim/driver/wormhole $ python3 four/four.py
--> Launching and running firmware
    --> Done, device is ready
--> Launching and running kernel
    --> Done
Example four completed successfully
```

>**NOTE:**  
> For this example you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.

## Contents

* [src](src) is the source code for the host and device kernels
* [binaries](binaries) are the binaries (built from the device kernels in [src](src)) that will be deployed to the simulator
