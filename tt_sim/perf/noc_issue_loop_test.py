"""The issuing core's loop (``tt_sim/perf/noc_issue_loop.py``).

The claims, in the order that matters:

1. The encoders produce the instructions they say they do, checked against
   words emitted by tt-metal's own ``riscv-tt-elf-gcc`` for the same source.
2. The per-transaction store list matches the vendor headers -- the thing that
   makes this a transcription rather than an invention, and the thing that
   makes Blackhole cost one cycle more than Wormhole.
3. Run under the cost model, one iteration costs its instruction count plus the
   six-cycle load-use interlock on the ``NOC_CMD_CTRL`` poll, on both
   architectures and in both directions. This is the load-bearing claim: it was
   *predicted* from the ISA docs' load-latency table before it was measured, and
   it is what makes the sweep's use of this program a validation rather than a
   fit.
4. The cost is flat in the number of transactions and flat in transfer size --
   an issue loop, not a network effect.
5. With the cost model off, the interlock is not charged and the loop costs
   exactly its instruction count. The six cycles are the model's, not an
   artefact of the program.

Runs standalone (``python3 -m tt_sim.perf.noc_issue_loop_test``) or under
pytest.
"""

import os
from contextlib import contextmanager

import pytest

from tt_sim.perf import noc_issue_loop as loop
from tt_sim.perf.noc_dataset_sweep import MEMORY_L1, predict_timed_region

ARCHES = ("wormhole", "blackhole")


@contextmanager
def _cost_model(on):
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if on:
        os.environ["TT_SIM_COST_MODEL"] = "1"
    else:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


# -- 1. the encoders --------------------------------------------------------
#
# Reference words from ``riscv-tt-elf-gcc -O3 -march=rv32im`` on the C the
# module's docstring quotes, read out of the objdump listing. Encoders that
# quietly produce the wrong bits would make every number below meaningless
# while still looking entirely plausible.

GCC_REFERENCE = (
    (0x000017B7, loop.lui(15, 0x1), "lui a5, 0x1"),
    (0xFFB00537, loop.lui(10, 0xFFB00), "lui a0, 0xffb00"),
    (0x0007A883, loop.lw(17, 15, 0), "lw a7, 0(a5)"),
    (0x0047A303, loop.lw(6, 15, 4), "lw t1, 4(a5)"),
    (0x02872783, loop.lw(15, 14, 40), "lw a5, 40(a4)"),
    (0x00672023, loop.sw(6, 14, 0), "sw t1, 0(a4)"),
    (0x01E72E23, loop.sw(30, 14, 28), "sw t5, 28(a4)"),
    (0x2007A023, loop.sw(0, 15, 512), "sw zero, 512(a5)"),
    (0x00850513, loop.addi(10, 10, 8), "addi a0, a0, 8"),
    (0x00000613, loop.addi(12, 0, 0), "li a2, 0"),
    (0xFE079EE3, loop.bne(15, 0, -4), "bnez a5, .-4"),
    (0x0000006F, loop.jal(0, 0), "j ."),
)


@pytest.mark.parametrize("expected,got,disasm", GCC_REFERENCE)
def test_the_encoders_agree_with_the_reference_build(expected, got, disasm):
    assert got == expected, f"{disasm}: {got:#010x} != {expected:#010x}"


def test_the_loop_branch_lands_on_the_top_of_the_loop():
    """A back-edge off by one instruction still runs; it just runs wrong."""
    for arch in ARCHES:
        for is_read in (False, True):
            program = loop.issue_loop_program(arch, is_read)
            body = loop.instructions_per_transaction(arch, is_read)
            # tail: loop branch, barrier lw, barrier branch, sw DONE, spin
            branch = program[-5]
            # B-type immediate, reassembled.
            imm = (
                (((branch >> 31) & 0x1) << 12)
                | (((branch >> 7) & 0x1) << 11)
                | (((branch >> 25) & 0x3F) << 5)
                | (((branch >> 8) & 0xF) << 1)
            )
            offset = imm - 0x2000 if imm & 0x1000 else imm
            assert offset == -4 * (body - 1), (arch, is_read, offset)


# -- 2. the store list is the vendor headers' -------------------------------


def test_blackhole_writes_exactly_one_more_register_per_transaction():
    """The whole of the two architectures' one-cycle difference.

    Blackhole splits the far address across ``*_ADDR_MID`` and
    ``*_ADDR_COORDINATE`` where Wormhole writes only the latter, in both
    directions. If this ever stops being true the 22-vs-23 and 18-vs-19 pairs
    below stop meaning anything.
    """
    for is_read in (False, True):
        wormhole = loop.ISSUE_REGISTERS[("wormhole", is_read)]
        blackhole = loop.ISSUE_REGISTERS[("blackhole", is_read)]
        assert len(blackhole) == len(wormhole) + 1
        extra = {name for name, _src in blackhole} - {name for name, _src in wormhole}
        assert extra == ({"TARG_ADDR_MID"} if is_read else {"RET_ADDR_MID"})


def test_a_read_does_not_configure_ctrl_per_transaction():
    """``DM_DEDICATED_NOC`` sets the read command field once, not in the loop.

    It is a ``if constexpr (noc_mode == DM_DYNAMIC_NOC)`` store. Putting it in
    the loop would cost a cycle a real kernel does not pay.
    """
    for arch in ARCHES:
        assert "CTRL" not in {n for n, _s in loop.ISSUE_REGISTERS[(arch, True)]}
        assert "CTRL" in {n for n, _s in loop.ISSUE_REGISTERS[(arch, False)]}


def test_every_store_lands_on_a_register_the_architecture_has():
    for (arch, is_read), stores in loop.ISSUE_REGISTERS.items():
        for name, _source in stores:
            assert name in loop._REG[arch], (arch, is_read, name)


def test_the_instruction_count_is_the_published_load_use_interlock_apart():
    """The prediction, stated as arithmetic rather than as a measurement.

    Six cycles is the ">= 7" row of the RISC-V load-latency table -- the row
    naming "NoC 0 configuration and command" -- and it is charged once per
    iteration, on the ``NOC_CMD_CTRL`` poll.
    """
    for key, expected in loop.EXPECTED_CYCLES_PER_TRANSACTION.items():
        assert loop.instructions_per_transaction(*key) + 6 == expected, key


# -- 3, 4. what it costs, run ------------------------------------------------


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("is_read", [False, True])
def test_one_transaction_costs_its_instructions_plus_the_interlock(arch, is_read):
    with _cost_model(True):
        near = predict_timed_region(arch, MEMORY_L1, is_read, False, 64, 16)
        far = predict_timed_region(arch, MEMORY_L1, is_read, False, 64, 64)
    marginal = (far - near) / 48
    assert marginal == loop.EXPECTED_CYCLES_PER_TRANSACTION[(arch, is_read)]


@pytest.mark.parametrize("arch", ARCHES)
def test_the_issue_cost_is_flat_in_burst_length_and_in_size(arch):
    """An issue loop is per-transaction work; a network term is not.

    Below the size at which serialisation binds, the marginal cost of one more
    transaction must not depend on how many came before it or on how big it is.
    """
    with _cost_model(True):
        marginals = set()
        for size in (64, 256):
            cycles = {
                n: predict_timed_region(arch, MEMORY_L1, False, False, size, n)
                for n in (4, 16, 64)
            }
            marginals.add((cycles[16] - cycles[4]) / 12)
            marginals.add((cycles[64] - cycles[16]) / 48)
    assert marginals == {loop.EXPECTED_CYCLES_PER_TRANSACTION[(arch, False)]}


@pytest.mark.parametrize("arch", ARCHES)
def test_with_the_model_off_the_loop_costs_exactly_its_instructions(arch):
    """The six cycles are the cost model's, and are not baked into the program.

    With the model off every instruction retires in one cycle, so the marginal
    must fall to the instruction count exactly. If it did not, the program
    itself would be carrying a stall and the agreement with the dataset would
    be an accident.
    """
    with _cost_model(False):
        near = predict_timed_region(arch, MEMORY_L1, False, False, 64, 16)
        far = predict_timed_region(arch, MEMORY_L1, False, False, 64, 64)
    assert (far - near) / 48 == loop.instructions_per_transaction(arch, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
