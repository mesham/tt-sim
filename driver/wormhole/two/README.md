# Example one

This example reads data using BRISC, then performs addition and uses a circular buffer to send the results to NCRISC which write these back to DRAM. This therefore involves testing circular buffers and deploying on BRISC and NCRISC. The script will check the computed values are correct.

```bash
~/tt-sim/driver/wormhole $ python3 two/two.py
--> Launching and running firmware
    --> Done, device is ready
--> Launching and running kernel
    --> Done
Example two completed successfully
```

>**NOTE:**  
> For this example you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.

## Contents

* [src](src) is the source code for the host and device kernels
* [binaries](binaries) are the binaries (built from the device kernels in [src](src)) that will be deployed to the simulator
