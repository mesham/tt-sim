CC=riscv32-unknown-elf-gcc
OBJCOPY=riscv32-unknown-elf-objcopy

CFLAGS=-march=rv32i -mabi=ilp32 -Wall -ffunction-sections -fdata-sections -nostartfiles -mno-fdiv -mstrict-align -mbranch-cost=10 -Wl,--gc-sections -ffp-contract=off -MMD -MP
LFLAGS=-T linker.ld
LIBS=-lm -lc -lgcc

all:
	${CC} ${CFLAGS} -c main.c -o main.o
	${CC} ${CFLAGS} -c crt0.S -o crt0.o
	${CC} ${CFLAGS} ${LFLAGS} main.o crt0.o ${LIBS} -o main.elf
	${OBJCOPY} -I elf32-little main.elf -j .text -j .rodata -j .data -O binary main.bin
