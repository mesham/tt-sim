# Example two

This will run a binary under RV32, note that the binary is very simple as there is no set up or teardown, along with no linker script

To build:

```console
riscv32-unknown-elf-gcc -O0 -o main main.c -ffreestanding -nostdlib --entry main
riscv32-unknown-elf-objcopy -O binary main main.bin
```