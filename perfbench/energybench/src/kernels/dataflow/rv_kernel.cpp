// energybench arm `rv` -- the RISC-V-heavy arm.
//
// A dependent integer chain on one baby RISC-V core: one MUL and three ALU ops
// per iteration, no memory traffic in the loop, no Tensix instruction, no NoC.
// The chain is dependent on purpose so the loop cannot be reassociated away,
// and the result is stored once at the end so the whole thing cannot be dead-
// code eliminated.
//
// What this arm contributes to the activity vector is ``instr_retired`` on
// BRISC and essentially nothing else.

void kernel_main() {
    uint32_t inner = get_arg_val<uint32_t>(0);
    uint32_t sink_addr = get_write_ptr(tt::CBIndex::c_0);

    uint32_t a = 0x12345678u;
    uint32_t b = 0x9E3779B9u;
    uint32_t acc = 1u;

    for (uint32_t i = 0; i < inner; i++) {
        acc = acc * b + a;
        a ^= (acc >> 7);
        b += acc;
    }

    volatile uint32_t* sink = reinterpret_cast<volatile uint32_t*>(sink_addr);
    *sink = acc;
}
