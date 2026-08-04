// Phase B of the Tensix cycle-cost benchmark: the fidelity-phase claim.
//
// docs/plans/cost-model.md derives the single most load-bearing number in the
// MATH table this way: one MVMUL is eight rows of SrcB against a 16x16 SrcA, so
// a 32x32x32 tile matmul is 16 MVMULs *per fidelity phase*
// (`fidelity_phases.mvmuls_per_tile`), and the fidelity multiplier is carried by
// the instruction stream rather than by a longer instruction -- so an MVMUL
// costs 1 cycle at every phase and a tile matmul costs 16 / 32 / 48 / 64 cycles
// at LoFi / HiFi2 / HiFi3 / HiFi4.
//
// WHAT IS ALREADY SETTLED, STATICALLY. The first Blackhole run of this kernel
// reported LoFi/HiFi2/HiFi4 within 0.2 cycles of each other where 16 and 32 were
// predicted. Before believing that, the generated code was checked. Disassembling
// the three JIT-built TRISC ELFs (tt-metal's cache, `riscv-tt-elf-objdump -d`)
// shows the fidelity setting reaching the math thread exactly as documented:
//
//   * trisc0 (unpack) and trisc2 (pack) are BYTE-IDENTICAL across all three
//     fidelities -- so nothing outside the math thread changes, and the
//     difference really is the clean measurement it is claimed to be;
//   * trisc1 (math) differs in exactly three places: the ADDR_MOD_5 fidelity
//     increment (`ttsetc16 33,3072` at LoFi vs `33,11264` at HiFi2/HiFi4), the
//     final MVMUL of the recorded sequence (`ttmvmul 1,0,5,0` vs `0,0,5,0`), and
//     the `ckernel_template` inner-loop count stored to the MOP config: 1 at
//     LoFi, `li a3,2` at HiFi2, `li a3,4` at HiFi4;
//   * the recorded sequence is `ttreplay 16,16,0,1` followed by exactly SIXTEEN
//     `ttmvmul` instructions -- i.e. `mvmuls_per_tile = 16` is confirmed by
//     construction, and the MOP expands it 1x / 2x / 4x.
//
// So the fidelity setting propagates, and a tile matmul really is 16 / 32 / 64
// MVMULs at LoFi / HiFi2 / HiFi4. What the null result means is therefore NOT
// "fidelity is ignored" but "this loop was not limited by the math unit".
//
// WHY THE FIRST SHAPE COULD NOT SEE IT. Those MVMULs are emitted by the Tensix
// MOP expander, not by the RISC-V core: the math thread pushes ONE `MOP`
// instruction and the expansion happens inside the coprocessor. A thread's wall
// clock therefore only charges for them if the Tensix backs pressure up into the
// instruction FIFO. The first shape did a `cb_wait_front` and a `cb_pop_front`
// per operand per matmul -- about 81 cycles of RISC-V semaphore work per call,
// measured -- which is more than even HiFi4's 64 MVMULs, so the coprocessor
// never fell behind and the fidelity difference stayed invisible.
//
// WHAT THIS SHAPE DOES INSTEAD. It blocks the loop: one `cb_wait_front` and one
// `cb_pop_front` per operand covers TTBENCH_MM_BLOCK matmuls. That divides the
// circular-buffer cost per call by the block factor and leaves the per-call
// RISC-V work at a handful of instructions, which is the only way the Tensix
// side can become the limit. It is still the "wait, matmul, pop, pack" shape
// that `examples/six` uses -- deliberately, because the alternative (hoist the
// wait out entirely and never pop) deadlocks tt-sim's Src-bank handshake, which
// is recorded in docs/plans/tensix-cost-benchmark.md and not chased here.
//
// The measurement that is clean is still the DIFFERENCE between fidelities.
// Raising math fidelity changes the MVMUL count and nothing else -- proved above
// from the disassembly, not assumed -- so
//
//     slope(HiFi2) - slope(LoFi)  should be 16 cycles
//     slope(HiFi4) - slope(HiFi2) should be 32 cycles
//
// IF THE UNPACKER IS SLOWER THAN THE MATH the differences are still zero, and
// that is a real result rather than a broken one: it says a bf16 tile matmul is
// unpack-bound at every fidelity the hardware offers. The host prints the two
// deltas next to the unpack thread's own slope so the two cases can be told
// apart, and does NOT fail the run for a null -- an unpack-bound composite is a
// finding, and phase A must not be invalidated by it.
//
// THE PACK THREAD IS NOT MEASURED. `cb_wait_front`/`cb_pop_front` are UNPACK-only
// and `matmul_tiles` is UNPACK+MATH, so the pack thread's copy of the inner loop
// is empty by construction and times ~1 cycle regardless of the iteration count.
// It used to be recorded anyway, which made it fail the linearity and
// monotonicity checks and took the whole run's verdict down with it. It still
// runs the loop (it owns the dst handshake and the output pack) but writes no
// timing, and the host emits no phase B row for thread 2.
//
// The operand tiles are never written by anyone. Values are irrelevant to
// timing and leaving them uninitialised is what lets phase B run with no DRAM
// buffer and no golden.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

#include "bench_layout.h"

namespace {

inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    return p[0];
}

}  // namespace

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_iters = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    uint32_t thread = 2;
#if defined(TRISC_UNPACK)
    thread = 0;
#elif defined(TRISC_MATH)
    thread = 1;
#endif

    volatile tt_l1_ptr uint32_t* base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
    if (thread == 1) {
        base[TTBENCH_HDR_MAGIC] = TTBENCH_MAGIC;
        base[TTBENCH_HDR_UNROLL] = 1;  // one matmul_tiles iteration, not a block
        base[TTBENCH_HDR_NUM_PROBES] = 1;
        base[TTBENCH_HDR_NUM_POINTS] = TTBENCH_MM_NUM_POINTS;
        base[TTBENCH_HDR_BASE_BLOCKS] = base_iters;
        base[TTBENCH_HDR_ACTIVE_MASK] = 0x3;  // unpack + math; pack is not timed
        base[TTBENCH_HDR_PROBE_MASK] = 0x1;
    }
    volatile tt_l1_ptr uint32_t* out =
        base + TTBENCH_HDR_WORDS + thread * TTBENCH_NUM_PROBES * TTBENCH_NUM_POINTS;

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t p = 0; p < TTBENCH_MM_NUM_POINTS; p++) {
        const uint32_t iters = base_iters * (p + 1);
        acquire_dst();
        const uint32_t t0 = wall_clock_lo();
        // Blocked so the circular-buffer bookkeeping is amortised over
        // TTBENCH_MM_BLOCK matmuls rather than paid per matmul. `iters` need not
        // be a multiple of the block; the last group is short.
        for (uint32_t done = 0; done < iters;) {
            uint32_t blk = iters - done;
            if (blk > TTBENCH_MM_BLOCK) {
                blk = TTBENCH_MM_BLOCK;
            }
            cb_wait_front(cb_in0, blk);
            cb_wait_front(cb_in1, blk);
            for (uint32_t j = 0; j < blk; j++) {
                matmul_tiles(cb_in0, cb_in1, j, j, 0);
            }
            cb_pop_front(cb_in0, blk);
            cb_pop_front(cb_in1, blk);
            done += blk;
        }
        if (thread != 2) {
            out[p] = wall_clock_lo() - t0;
        }
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        release_dst();
    }
}
