// ================================================================================ //
// NEORV32 CPU - crt0.S Start-Up Code                                               //
// -------------------------------------------------------------------------------- //
// The NEORV32 RISC-V Processor - https://github.com/stnolting/neorv32              //
// Copyright (c) NEORV32 contributors.                                              //
// Copyright (c) 2020 - 2025 Stephan Nolting. All rights reserved.                  //
// Licensed under the BSD-3-Clause license, see LICENSE for details.                //
// SPDX-License-Identifier: BSD-3-Clause                                            //
// ================================================================================ //

.file "crt0.S"
.section .text.crt0
.balign 4
.global _start
.global __crt0_entry
.global __crt0_main_exit

_start:
__crt0_entry:
.cfi_startproc
.cfi_undefined ra

// ************************************************************************************************
// Register setup.
// ************************************************************************************************
.option push
.option norelax
  //csrr  x1, mhartid                     // get ID of this core

  la    x4, __crt0_ram_last             // last address of RAM, stack pointer (sp) starts here
  andi  x2, x4, 0xfffffff0              // align stack to 16-bytes according to the RISC-V ABI (#1021)
  la    x3, __global_pointer            // global pointer "gp"

  //li    x5, 0x00001800                  // mstatus.mpp = machine-mode
  //csrw  mstatus, x5
  //la    x6, __crt0_trap                 // configure early-boot trap handler
  //csrw  mtvec, x6
  //csrw  mie, zero                       // disable all interrupt sources

  la    x7,  __crt0_copy_data_src_begin // .data: start of copy-source (in .rodata)
  la    x8,  __crt0_copy_data_dst_begin // .data: start of actual data region
  la    x9,  __crt0_copy_data_dst_end   // .data: end of actual data region
  la    x10, __crt0_bss_start           // .bss: start address
  la    x11, __crt0_bss_end             // .bss: end address (not part of bss)
.option pop

  // initialize remaining registers
  addi  x12, zero, 0
  addi  x13, zero, 0
  addi  x14, zero, 0
  addi  x15, zero, 0
  addi  x16, zero, 0
  addi  x17, zero, 0
  addi  x18, zero, 0
  addi  x19, zero, 0
  addi  x20, zero, 0
  addi  x21, zero, 0
  addi  x22, zero, 0
  addi  x23, zero, 0
  addi  x24, zero, 0
  addi  x25, zero, 0
  addi  x26, zero, 0
  addi  x27, zero, 0
  addi  x28, zero, 0
  addi  x29, zero, 0
  addi  x30, zero, 0
  addi  x31, zero, 0

// ************************************************************************************************
// Copy .data section from ROM to RAM.
// ************************************************************************************************
  beq   x7, x8, __crt0_data_copy_end // __crt0_copy_data_src_begin = __crt0_copy_data_dst_begin

__crt0_data_copy:
  bge   x8, x9,  __crt0_data_copy_end
  lw    x15, 0(x7)
  sw    x15, 0(x8)
  addi  x7, x7, 4          // word-wise operations; section begins and ends on word boundary
  addi  x8, x8, 4
  j     __crt0_data_copy

__crt0_data_copy_end:


// ************************************************************************************************
// Clear .bss section.
// ************************************************************************************************
__crt0_bss_clear:
  bge   x10, x11, __crt0_bss_clear_end
  sw    zero, 0(x10)
  addi  x10, x10, 4        // word-wise operations; section begins and ends on word boundary
  j     __crt0_bss_clear

__crt0_bss_clear_end:


// ************************************************************************************************
// Call constructors (not supported for bootloader).
// WARNING! Constructors do not preserve any registers on the stack.
// ************************************************************************************************
  la    x8, __init_array_start
  la    x9, __init_array_end

__crt0_constructors:
  bge   x8, x9, __crt0_constructors_end
  lw    x1, 0(x8)
  jalr  x1, 0(x1)  // call constructor function; put return address in ra
  addi  x8, x8, 4
  j     __crt0_constructors

__crt0_constructors_end:


// ************************************************************************************************
// Setup arguments and call main function.
// ************************************************************************************************
  la    x12, main             // primary core's (core0) entry point (#1169)
__crt0_main_entry:
  la    x3, __global_pointer  // re-initialize global pointer "gp" (to prevent a race condition during SMP boot)
  fence                       // synchronize loads/stores
  fence.i                     // synchronize instruction fetch

  //csrw  mstatus, x5           // re-initialize
  addi  x10, zero, 0          // x10 = a0 = argc = 0
  addi  x11, zero, 0          // x11 = a1 = argv = 0
  jalr  x1, x12               // call actual main function; put return address in ra

__crt0_main_exit:             // main's "return" and "exit" will arrive here
  //csrw  mie, zero             // disable all interrupt sources
  //csrw  mscratch, x10         // backup main's return code to mscratch (for debugger or destructors)
  //la    x10, __crt0_trap      // re-install the crt0 trap handler
  //csrw  mtvec, x10


// ************************************************************************************************
// Go to endless sleep mode. This is the end if no interrupts are enabled.
// ************************************************************************************************
__crt0_halt:
  ebreak  // jump to debug-mode if an external debugger is connected

__crt0_sleep:
  //wfi
  j __crt0_sleep

.cfi_endproc
.end
