// Shared between the host program (dramratebench.cpp) and the two kernels.
//
// Every result carries the reader's own physical NoC coordinate, the DRAM
// coordinate it was told to read from, and the tag word it actually found
// there, so a row proves which endpoint it measured rather than asserting it.
// That last part is not decoration: this benchmark's whole question is which
// DRAM channel N readers landed on, and a row that merely *claims* a channel
// is indistinguishable from a row that missed.
//
// TWO DIRECTIONS SINCE 2026-08-17, and the write one is verified differently.
// A read that landed on the wrong endpoint comes back with the wrong bytes and
// the reader itself can say so. A WRITE that landed on the wrong endpoint says
// nothing at all -- it completes, the barrier retires, the rate is plausible,
// and the only trace of it is bytes sitting somewhere the plan never named. So
// a write point is graded by the HOST, in both directions:
//
//   * every writer's target block must hold that writer's witness word
//     (the positive half: the bytes went where the plan said), and
//   * every block the point did NOT target must still hold POISON
//     (the negative half: no bytes went anywhere else).
//
// See DRAMRATEBENCH_POISON and DRAMRATEBENCH_WITNESS below for what each of
// those two catches and, just as importantly, what neither can.
#pragma once

// "DRB2"; bump on any layout change. Bumped from DRB1 on 2026-08-17 when the
// write direction added result words 16-17 and argument 14. Nothing reads this
// value as a gate -- `tt_sim.perf.dram_rate_sweep` deliberately carries it as
// metadata and never compares it, so that a bumped magic cannot make a banked
// dataset unreadable -- and every DRB1 column keeps its DRB1 index above.
#define DRAMRATEBENCH_MAGIC 0x44524232u

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
//
// A WRITER fills these from a read-back of its own target block, taken AFTER
// the timed region and after the write barrier. That proves the writes were
// accepted and landed -- a dropped or mis-congruent write leaves POISON here --
// but it CANNOT prove they landed where the plan said, because it re-reads
// through the very coordinate and offset the write used. Misdirection is the
// host's check, not this one, and the two are reported in separate columns for
// exactly that reason.
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
// The witness word this participant wrote, echoed back so the host grades the
// write against what the DEVICE believed it was writing rather than against
// what the host believed it had told it to write. Those are the same number
// until a runtime argument goes astray, which is the case worth catching.
// Zero on a reader.
#define DRAMRATEBENCH_R_WITNESS 16
#define DRAMRATEBENCH_R_DIR 17  // DRAMRATEBENCH_DIR_READ or _DIR_WRITE
#define DRAMRATEBENCH_R_WORDS 18

// ---------------------------------------------------------------------------
// Runtime argument indices, shared by dram_reader.cpp and dram_writer.cpp.
// One block for both kernels: the arguments a direction does not use are
// still set (to 0), so a point's argument layout cannot depend on which
// kernel is about to consume it.
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
#define DRAMRATEBENCH_A_WITNESS 14  // writers only: the word to fill the arena with
#define DRAMRATEBENCH_A_PEERS 15  // 2 * num_readers words follow, (x, y) pairs

// ---------------------------------------------------------------------------
// Direction. Not a kernel argument -- it selects which kernel is built -- but
// stamped into every result so a row says which of the two it is without
// anyone having to trust the arm name in the CSV next to it.
// ---------------------------------------------------------------------------
#define DRAMRATEBENCH_DIR_READ 0
#define DRAMRATEBENCH_DIR_WRITE 1

// ---------------------------------------------------------------------------
// The write direction's addressing proof, which is TWO checks and needs to be.
// ---------------------------------------------------------------------------
// The host writes POISON over every block of the write region before EVERY
// write point -- not once at start-up. Re-poisoning is what keeps the check
// falsifiable across repeats: after repeat 0 the region already holds correct
// witnesses, so a repeat-1 point whose writes were all silently dropped would
// read back exactly the right words and pass. This benchmark has been bitten
// once by a check that could not fail (the 2026-08-09 card run tagged slice 0
// only, so `tags_ok` could not pass and clean data was thrown away); a check
// that cannot fail is the same defect with the sign flipped, and it does not
// throw data away, it certifies it.
#define DRAMRATEBENCH_POISON 0xBADDBADDu

// The word a writer aimed at bank `b`, slice `s` fills its arena with, and
// therefore the word every one of its timed writes carries. A FUNCTION OF THE
// TARGET, not of the writer: where a point has more writers than slices, two
// writers share a target block and must agree about what belongs in it.
//
// Disjoint from DRAMRATEBENCH_POISON and from the read direction's
// `0xDB000000 | bank` tags by construction, so no readback can be ambiguous
// about which of the three it is looking at.
#define DRAMRATEBENCH_WITNESS(bank, slice) \
    (0xDB800000u | (((bank) & 0xFFFu) << 8) | ((slice) & 0xFFu))

// Bytes of witness the host reads back per block, and bytes the reader's tag
// check reads. Thirty-two rather than four: a DRAM-to-L1 read must be
// congruent in its low 5 bits on Wormhole and its low 6 on Blackhole, and 32 is
// the smallest size that is safe on both from a pair of aligned bases.
#define DRAMRATEBENCH_TAG_BYTES 32

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
