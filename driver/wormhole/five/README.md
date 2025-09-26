# Example five

This example uses the vector unit (SFPU) to do the element wise addition of int32. In the Tensix coprocessor data is unpacked to different rows of dst, the SFPU executes the SFPIADD instruction storing results to rows of dst, with the packer then copying these into L1.

```bash
~/tt-sim/driver/wormhole $ python3 five/five.py
--> Launching and running firmware
    --> Done, device is ready
--> Launching and running kernel
    --> Done
Example five completed successfully
```

>**NOTE:**  
> For this example you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.

## Contents

* [src](src) is the source code for the host and device kernels
* [binaries](binaries) are the binaries (built from the device kernels in [src](src)) that will be deployed to the simulator
