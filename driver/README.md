# Driver for the simulator

The benefit of Python is that we can very easily write scripts that drive parts of the simulator, making it trivial to experiment with different parts. There are two groups of drivers provided here:

* [simple](https://github.com/mesham/tt-sim/tree/main/driver/simple) which is a range of examples driving individual components, building up to codes running on the RV32IM CPU. Helped to develop and test some of the foundational building blocks.
* [wormhole](https://github.com/mesham/tt-sim/tree/main/driver/wormhole) which is a simple Wormhole (one DRAM tile, one Tensix tile) and a range of code kernels that will run on this. There is also a simple TT-Metal wrapper provided (release 0.62.2 based) that abstracts communication between the host and device that is needed for setup and kernel execution.
