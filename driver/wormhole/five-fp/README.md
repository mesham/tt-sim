# Example five floating point version

This example uses the vector unit (SFPU) to do the element wise addition of floating point. In the Tensix coprocessor data is unpacked to different rows of dst, the SFPU executes the SFPADD instruction storing results to rows of dst, with the packer then copying these into L1.

There are a few different versions included, and it is interesting to see the different set of instructions issued based upon the datatype. The base version here is working in FP32 throughout, where FP32 data is unpacked into dst, computed upon and then FP32 results are stored in dst and then the packer stores these into L1.

```bash
~/tt-sim/driver/wormhole $ python3 five-fp/five-fp.py
--> Launching and running firmware
    --> Done, device is ready
--> Launching and running kernel
    --> Done
Example five completed successfully
```

>**NOTE:**  
> For this example you should run from the _wormhole_ directory due to paths to binaries and parameter files being relative from here in configuration scripts.

## Contents

* [src](src) is the source code for the host and device kernels (FP32)
* [binaries](binaries) are the binaries (built from the device kernels in [src](src)) that will be deployed to the simulator (FP32)
* [fp16](fp16) uses FP32 on the host, but the unpacker converts this into FP32 and stores it in srcA. The matrix unit then copies from srcA into dst, with the SFPU working on FP16, with results stored in dst and then converted back to FP32 by the packer. Note this is the default without _fp32_dest_acc_en_ (use 32-bit dst) and _unpack_to_dest_mode_ (or unpack to src for this second argument) provided as compute kernel arguments.
* [tf32](tf32) uses FP32 on the host, but the unpacker converts into TF32 (19 bit) and stores it in srcA. The matrix unit then copies from srcA into dst, storing as FP32 in dst (but as it has been TF32, then only 10 bits of the mantissa have been preserved) with the SFPU working on this FP32 number. Results are stored in dst as FP32 and then written by the packer to L1. This behaviour is adopted when the argument _fp32_dest_acc_en_ (use 32-bit dst) is provided to the compute kernel, but _unpack_to_dest_mode_ is not provided (or unpack to src which is the default is selected).
