# Example four floating point version

This example performs element wise addition of FP16 via uses the matrix unit in the Tensix coprocessor using the ELWADD instruction. In the Tensix co-processor FP32 data is unpacked to srcA and srcB and converted to FP16, then element wise addition is undertaken and results are stored in dst (in FP16) which are then converted to FP32 and packed to L1. As per previous examples, BRISC loads DRAM data into L1, NCRISC writes results from L1 back to DRAM. The script will check the computed values are correct.

```bash
~/tt-sim/driver/wormhole $ python3 four-fp/four-fp.py
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
