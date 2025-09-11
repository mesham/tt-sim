# Testing separate memory spaces

This tests separate device and PE local memory spaces. To compile the included C code use the same makefile, linker script and startup assembly from ex3.

>> IMPORTANT: You need to change `march=rv32i` to `march=rv32im` in the makefile to compile for the `m` extension ISA