# Testing RV32IM

This tests RV32IM (specifically support for the M extension). To compile the included C code use the same makefile, linker script and startup assembly from ex3.

>> IMPORTANT: You need to change `march=rv32i` to `march=rv32im` in the makefile to compile for the `m` extension ISA