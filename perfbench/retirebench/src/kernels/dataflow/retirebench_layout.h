// retirebench -- the layout the kernel writes and the host reads.
//
// Included by BOTH sides (kernels/dataflow/rv_zones.cpp and retirebench.cpp),
// so the zone table exists once. A zone's identity is its INDEX here: the
// artefact records names, and a reader that matched on position instead would
// silently compare two different mechanisms if this table ever grew a row.
// Append only.
#pragma once

#define RETIREBENCH_MAGIC 0x52564D31u  // "RVM1"
#define RETIREBENCH_VERSION 1u

// ---------------------------------------------------------------------------
// The result buffer, in 32-bit words.
//
// Every value is the LOW half of a 64-bit counter. Two reads bracketing a zone
// are subtracted with unsigned arithmetic, so a single wrap of the low half is
// handled exactly; a delta of 2^32 or more is not, and cannot occur here (the
// whole kernel is ~25 k cycles). The high halves are deliberately NOT read:
// each CSR access is a retirement barrier while cfg0's DisCsrSync is clear,
// which it is at reset and in tt-metal's init, so doubling the reads per marker
// would double the instrument's own cost for a range nothing needs.
// ---------------------------------------------------------------------------
#define RETIREBENCH_HDR_MAGIC 0
#define RETIREBENCH_HDR_VERSION 1
#define RETIREBENCH_HDR_NUM_ZONES 2
#define RETIREBENCH_HDR_SCALE 3
#define RETIREBENCH_HDR_WINDOW_C0 4
#define RETIREBENCH_HDR_WINDOW_C1 5
#define RETIREBENCH_HDR_WINDOW_I0 6
#define RETIREBENCH_HDR_WINDOW_I1 7
#define RETIREBENCH_HDR_WORDS 8

// Per zone: mcycle before, mcycle after, minstret before, minstret after.
#define RETIREBENCH_ZONE_WORDS 4
#define RETIREBENCH_Z_C0 0
#define RETIREBENCH_Z_C1 1
#define RETIREBENCH_Z_I0 2
#define RETIREBENCH_Z_I1 3

// ---------------------------------------------------------------------------
// The zones.
//
// Each is a loop of `reps` iterations over a body of `unroll` instructions,
// built so that ONE RISC-V mechanism dominates it. The mechanism is a property
// of the program's construction, not of a hardware counter -- see the README's
// "hardware-attributed versus structural" section, which is the one thing a
// reader of this benchmark must not get wrong.
//
// `reps` is chosen so that every zone lands near 6000 cycles ON A CARD, using
// the per-instruction costs perfbench/riscvbench measured on Blackhole silicon
// (2026-08-05, recorded in tt_sim/pe/rv/cost.py's docstring):
//
//     addi 1.00   mul indep 0.999   mul dep 1.985   divu 0x12345678/3 33.001
//     L1 chase 8.098   L1 indep load 1.742   spread L1 store ~5
//
// Equal-sized zones are a design requirement, not a tidiness preference: E_int
// and E_total are both denominated in the whole span, so a zone allowed to
// dominate the span would let its own error dominate both numbers and the
// partition would stop being a decomposition.
//
// WHY 6000 AND NOT 2000, which is what a 1000-cycle floor and a ~30-cycle
// marker cost would otherwise suggest. `div_large` is the constraint, and it is
// the one zone whose two sides are KNOWN to disagree before the program is run:
// the divide's cost is a documented data dependence ("between six and 33 cycles
// ... dependent upon the magnitude of the dividend"), tt-sim charges the
// documented floor of 6 for every operand, and riscvbench read 33.001 on
// silicon at a 29-bit dividend. So the zone must be long enough that the
// SIMULATOR side -- the short one, at 6 cycles a divide -- still clears the
// 1000-cycle floor, which puts the card side near 6300; and every other zone
// must then be sized alongside it, or that one known gap would be most of
// `E_total` and the leg would be a test of a thing already measured rather than
// a decomposition. At these sizes it is ~7 % of the span: visible in the
// interior, not in command of the total.
//
// `div_small` is deliberately shorter than the rest (~3000 simulator cycles).
// Its card cost is the one number in this table that is NOT measured -- the
// docs give the 6-33 band and no function within it, and no session has read a
// 12-bit dividend -- so it is the one zone whose card side could come back
// several times its simulator side unannounced. Sizing it half-length bounds
// what an unknown can do to E_total without hiding it.
// ---------------------------------------------------------------------------
#define RETIREBENCH_Z_MARKER_NULL 0
#define RETIREBENCH_Z_ALU_DEP 1
#define RETIREBENCH_Z_ALU_IND 2
#define RETIREBENCH_Z_MUL_DEP 3
#define RETIREBENCH_Z_MUL_IND 4
#define RETIREBENCH_Z_DIV_SMALL 5
#define RETIREBENCH_Z_DIV_LARGE 6
#define RETIREBENCH_Z_LOAD_DEP 7
#define RETIREBENCH_Z_LOAD_IND 8
#define RETIREBENCH_Z_STORE_SPREAD 9
#define RETIREBENCH_Z_BRANCH_NT 10
#define RETIREBENCH_Z_BRANCH_T 11
#define RETIREBENCH_NUM_ZONES 12

// Base repetition counts, before `--scale`. Zone 0 runs its body zero times:
// it is the calibration zone and measures the two-marker cost alone.
#define RETIREBENCH_REPS_MARKER_NULL 0
#define RETIREBENCH_REPS_ALU_DEP 94
#define RETIREBENCH_REPS_ALU_IND 94
#define RETIREBENCH_REPS_MUL_DEP 47
#define RETIREBENCH_REPS_MUL_IND 94
#define RETIREBENCH_REPS_DIV_SMALL 31
#define RETIREBENCH_REPS_DIV_LARGE 13
#define RETIREBENCH_REPS_LOAD_DEP 12
#define RETIREBENCH_REPS_LOAD_IND 54
#define RETIREBENCH_REPS_STORE_SPREAD 19
#define RETIREBENCH_REPS_BRANCH_NT 47
#define RETIREBENCH_REPS_BRANCH_T 47

// ---------------------------------------------------------------------------
// The scratch region the memory zones work on. Two disjoint blocks, because
// the store zone would otherwise overwrite the pointer chase's ring and the
// load zone that follows it would chase into whatever the stores left.
// ---------------------------------------------------------------------------
#define RETIREBENCH_CHASE_NODES 64
#define RETIREBENCH_CHASE_STRIDE 16
#define RETIREBENCH_CHASE_BYTES (RETIREBENCH_CHASE_NODES * RETIREBENCH_CHASE_STRIDE)
// The 16-byte-aligned blocks the independent-load and spread-store zones rotate
// through, at chase_bytes .. chase_bytes + 127.
//
// EIGHT of them, not four, and that is the whole reason this block is 128 bytes.
// Blackhole publishes an L0 data cache of "a mere 64 bytes: 4 lines of 16 bytes
// each". A load zone that rotates through exactly four 16-byte lines therefore
// has a working set of exactly the published capacity, and whether those four
// stay resident depends on associativity, indexing and replacement -- none of
// which the docs publish. The zone's own name would then rest on an unpublished
// property: it would measure the L0 hit path on a machine that keeps them and
// the miss path on one that does not, and the first simulator run did exactly
// that (1.001 cycles/instruction against the 1.742 riscvbench read on silicon
// at the same four addresses). Eight lines exceed the published capacity under
// ANY organisation, so the zone measures the mechanism it is named for on both
// sides. The resident case is deliberately NOT given a zone of its own: a zone
// whose meaning depends on unpublished cache organisation is a zone whose label
// cannot be defended.
#define RETIREBENCH_DATA_OFFSET RETIREBENCH_CHASE_BYTES
#define RETIREBENCH_DATA_LINES 8
#define RETIREBENCH_DATA_BYTES (RETIREBENCH_DATA_LINES * 16)
#define RETIREBENCH_SCRATCH_BYTES (RETIREBENCH_CHASE_BYTES + RETIREBENCH_DATA_BYTES)

// One word past the zone table: the kernel's dead-code sink. It holds a sum of
// every operand the zone bodies touched, so the optimiser cannot fold one of
// them to a constant and quietly change which instruction sequence ran.
#define RETIREBENCH_SINK_WORD \
    (RETIREBENCH_HDR_WORDS + RETIREBENCH_NUM_ZONES * RETIREBENCH_ZONE_WORDS)

#define RETIREBENCH_RESULT_WORDS (RETIREBENCH_SINK_WORD + 1)
