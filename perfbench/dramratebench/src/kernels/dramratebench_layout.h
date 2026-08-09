// Shared between the host program (dramratebench.cpp) and the reader kernel.
//
// Every result carries the reader's own physical NoC coordinate, the DRAM
// coordinate it was told to read from, and the tag word it actually found
// there, so a row proves which endpoint it measured rather than asserting it.
// That last part is not decoration: this benchmark's whole question is which
// DRAM channel N readers landed on, and a row that merely *claims* a channel
// is indistinguishable from a row that missed.
#pragma once

#define DRAMRATEBENCH_MAGIC 0x44524231u  // "DRB1"; bump on any layout change

// ---------------------------------------------------------------------------
// Result word indices, per participating reader core.
// ---------------------------------------------------------------------------
#define DRAMRATEBENCH_R_MAGIC 0
#define DRAMRATEBENCH_R_POINT 1     // index of this point within the run
#define DRAMRATEBENCH_R_INDEX 2     // which reader this is, 0..num_readers-1
#define DRAMRATEBENCH_R_T0 3        // wall clock entering the timed region
#define DRAMRATEBENCH_R_T1 4        // wall clock leaving it
#define DRAMRATEBENCH_R_CYCLES 5    // t1 - t0, computed on device (wrap-safe)
#define DRAMRATEBENCH_R_NODE_X 6    // from NOC_NODE_ID, this reader's own coord
#define DRAMRATEBENCH_R_NODE_Y 7
#define DRAMRATEBENCH_R_DRAM_X 8  // the DRAM endpoint this reader was pointed at
#define DRAMRATEBENCH_R_DRAM_Y 9
#define DRAMRATEBENCH_R_NUM_TX 10
#define DRAMRATEBENCH_R_TX_BYTES 11
// The addressing proof. The host writes a distinct 32-bit tag at the base of
// every bank's slice; the kernel reads it back BEFORE the timed region and
// stamps what it saw. `_TAG_OK` is 1 only when that matched what the host said
// to expect. A sweep that walks off the addressed grid, a bank offset applied
// to the wrong core, or a DRAM channel index that is not a bank id all show up
// here as a mismatch instead of as a plausible-looking rate.
#define DRAMRATEBENCH_R_TAG_SEEN 12
#define DRAMRATEBENCH_R_TAG_OK 13
// How many times this reader went round the arrival spin before every other
// reader had checked in. Zero for the LAST reader to arrive and for a
// single-reader point, so "did anybody wait" is `max > 0` over the point --
// which is the only evidence that the N bursts really did overlap. Without it,
// N readers running one after another would produce exactly the flat aggregate
// this benchmark is looking for, from an experiment that never happened.
#define DRAMRATEBENCH_R_BARRIER_SPINS 14
#define DRAMRATEBENCH_R_NUM_READERS 15
#define DRAMRATEBENCH_R_WORDS 16

// ---------------------------------------------------------------------------
// Runtime argument indices for kernels/dataflow/dram_reader.cpp.
// ---------------------------------------------------------------------------
#define DRAMRATEBENCH_A_RESULTS 0
#define DRAMRATEBENCH_A_ARRIVE 1      // base of the arrival array (num_readers words)
#define DRAMRATEBENCH_A_LAND 2        // base of the local landing arena
#define DRAMRATEBENCH_A_LAND_BYTES 3  // how much of it a read may land in
#define DRAMRATEBENCH_A_POINT 4
#define DRAMRATEBENCH_A_INDEX 5        // this reader's slot in the arrival array
#define DRAMRATEBENCH_A_NUM_READERS 6  // how many readers are in this point
#define DRAMRATEBENCH_A_NUM_TX 7
#define DRAMRATEBENCH_A_TX_BYTES 8
#define DRAMRATEBENCH_A_DRAM_X 9    // physical NoC coord of this reader's bank
#define DRAMRATEBENCH_A_DRAM_Y 10
#define DRAMRATEBENCH_A_DRAM_ADDR 11  // bank-local base of this reader's slice
#define DRAMRATEBENCH_A_SLICE_BYTES 12  // how far into the slice a read may go
#define DRAMRATEBENCH_A_TAG 13          // the tag word the host wrote at _DRAM_ADDR
#define DRAMRATEBENCH_A_PEERS 14  // 2 * num_readers words follow, (x, y) pairs

// The arrival array is written with plain NoC writes and polled locally, NOT
// with `noc_semaphore_inc`. An atomic increment is the idiomatic tt-metal
// barrier, but it is one more device mechanism between this experiment and its
// answer, and this benchmark has to run against tt-sim as well as a card. A
// one-word-per-reader array written with `noc_async_write` and polled until
// every word is non-zero needs only the read/write path the measurement itself
// needs, is idempotent, and does not care in what order the writes land.
#define DRAMRATEBENCH_ARRIVED 1u

// Bytes between one reader's arrival slot and the next. SIXTEEN, not four, and
// this is an alignment requirement rather than padding. A NoC write out of L1
// must be CONGRUENT in its low 4 bits -- `(src % 16) == (dst % 16)`, per
// `WormholeB0/NoC/Alignment.md`, where a violation is UndefinedBehavior on
// hardware and raises `NoCAlignmentError` in tt-sim. The source word is a fixed
// 16-aligned address, so with 4-byte slots reader 1 would write to `base + 4`
// and violate it; reader 2 to `base + 8`, and so on. Every reader past 0 would
// be silently dropped on a card and would kill the simulator's server outright.
// With a 16-byte stride every slot is congruent to the source by construction.
#define DRAMRATEBENCH_SLOT_STRIDE 16

// A reader that spins here forever would hang the card session, so the poll is
// bounded. On expiry the kernel proceeds anyway and its spin count saturates at
// this value, which the host reports as a barrier failure rather than as a
// measurement. Generous: it is untimed, and it must not expire on a simulator
// running at a few tens of thousands of cycles per second.
#define DRAMRATEBENCH_BARRIER_LIMIT 40000000u

// ---------------------------------------------------------------------------
// Registers the kernel reads. Spelled out rather than relying on a
// force-include, exactly as nocbench and nocreadbench do for the wall clock.
// ---------------------------------------------------------------------------
#ifdef RISCV_DEBUG_REG_WALL_CLOCK_L
#define DRAMRATEBENCH_WALL_CLOCK_L RISCV_DEBUG_REG_WALL_CLOCK_L
#else
#define DRAMRATEBENCH_WALL_CLOCK_L 0xFFB121F0u
#endif
