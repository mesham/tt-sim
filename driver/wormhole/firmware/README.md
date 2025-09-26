# Tensix firmware

These firmware binaries are generated from source code [here](https://github.com/tenstorrent/tt-metal/tree/v0.62.2/tt_metal/hw/firmware/src/tt-1xx) and are always deployed when the Tenstorrent device begins to be used (e.g. not on poweron, but instead when it is going to be used by a new set of kernels). The firmware is always deployed to the simulator before a kernel is run, as it will set up all the subsystems and then has the logic to wait for kernel deployment and interfacing with the host.
