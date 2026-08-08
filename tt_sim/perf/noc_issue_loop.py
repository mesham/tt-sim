"""The issuing core's per-transaction loop, as RV32IM, for the rung-2 sweep.

Why this module exists
----------------------

``tt_sim.perf.noc_dataset_sweep`` predicts tt-metal's measured NoC dataset. Its
original predictor drove the initiator's NoC command registers **from Python**
and pumped until ``NIU_MST_REQS_OUTSTANDING`` returned to zero. That models the
network and the endpoint and nothing else -- but the measurement is a
``DeviceZoneScopedN`` region wrapping *the whole issue loop plus the closing
barrier*, so a RISC-V core's own instruction stream is inside every measured
number and was outside every predicted one.

The sweep recorded the difference as a constant 77-94 cycle residual and called
it "one unmodelled issuing-core path". **It is not unmodelled.** tt-sim runs
baby RISC-V cores against a published pipeline and a published load-latency
table, and a kernel that issues a NoC transaction pays for its own stores and
polls like any other code. What was missing was that the *harness* never ran
them. This module supplies the missing program, so the prediction covers the
same region the measurement does.

Nothing here is a cost. There is no new table entry, no new provenance, and no
simulated cycle moves anywhere outside this harness: every cycle the program
costs is charged by terms that were already in ``unit_costs.yaml``.

What the program is
-------------------

A faithful reconstruction of the loop the estimator kernels time::

    for (uint32_t i = 0; i < num_of_transactions; i++) {
        noc.async_write(...);          // or async_read
    }
    noc.async_write_barrier();         // or async_read_barrier

expanded through ``ncrisc_noc_fast_write_any_len`` /
``ncrisc_noc_fast_read_any_len`` at ``noc_mode == DM_DEDICATED_NOC``,
``use_trid == false``, ``mcast == false``. Two things come out of the vendor
headers and are transcribed rather than invented:

* **which command-buffer registers are written, and in which order.** This is
  :data:`ISSUE_REGISTERS`, taken verbatim from
  ``tt_metal/hw/inc/internal/tt-1xx/{wormhole,blackhole}/noc_nonblocking_api.h``.
  The two architectures genuinely differ: Blackhole writes one register more in
  each direction, because it splits the destination address across
  ``NOC_RET_ADDR_MID`` and ``NOC_RET_ADDR_COORDINATE`` where Wormhole writes
  only the latter.
* **the per-iteration wait.** ``ncrisc_noc_fast_*_any_len`` opens with
  ``while (!noc_cmd_buf_ready(noc, cmd_buf));``, a load of ``NOC_CMD_CTRL``
  followed immediately by a branch on it. That load is in the ">= 7" row of the
  RISC-V load-latency table -- the row that names "NoC 0 configuration and
  command" -- so it costs a six-cycle load-use interlock, charged since the NIU
  register block was wired.

The rest is bookkeeping the headers also do (``noc_reads_num_issued`` and the
two non-posted write counters, which the BRISC linker script places in local
data RAM at ``0xFFB00000``) and the loop's own induction variable.

Honesty about what this is
--------------------------

**This is a reconstruction of a compiled program, not a hardware constant.**
The measured per-transaction cost is the cost of *some* compilation of the
vendor's inline function; a different compiler, or a different tt-metal
release, would move it. That is a real property of the measurement rather than
of the chip, and it is why nothing here is eligible to become a cost-table
entry.

What makes the reconstruction checkable rather than fitted is that its cost was
**predicted before it was run**, from two numbers neither of which came from
this dataset: the instruction count (16 on Wormhole, 17 on Blackhole for a
write) and the six-cycle interlock from the ISA docs' load-latency table.
16 + 6 = 22 and 17 + 6 = 23; the dataset's measured marginals are 22.0 and
23.0. The same arithmetic gives 18 and 19 for a read, and a reference build
with tt-metal's own ``riscv-tt-elf-gcc -O3`` produces exactly those four
numbers from the C above. ``noc_issue_loop_test.py`` pins all four.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# RV32I encoders. Only the seven forms this program needs.
# ---------------------------------------------------------------------------

X0 = 0


def lui(rd, imm20):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0x37


def addi(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13


def lw(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (0x2 << 12) | (rd << 7) | 0x03


def sw(rs2, rs1, imm):
    imm &= 0xFFF
    return (
        ((imm >> 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (0x2 << 12)
        | ((imm & 0x1F) << 7)
        | 0x23
    )


def bne(rs1, rs2, offset):
    """Branch if not equal; ``offset`` is a byte displacement from this insn."""
    imm = offset & 0x1FFF
    return (
        (((imm >> 12) & 0x1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (0x1 << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 0x1) << 7)
        | 0x63
    )


def jal(rd, offset):
    imm = offset & 0x1FFFFF
    return (
        (((imm >> 20) & 0x1) << 31)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 0x1) << 20)
        | (((imm >> 12) & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
    )


# ---------------------------------------------------------------------------
# Memory layout the program and its driver agree on. All in the initiator
# tile's L1, clear of the payload buffer the sweep uses at 0x30000.
# ---------------------------------------------------------------------------

PROGRAM_ADDR = 0x0
PARAMS_ADDR = 0x1000  #: six words: coord, targ, ret, len, n, ctrl
DONE_ADDR = 0x1200  #: set to 1 after the barrier retires
START_ADDR = 0x1204  #: set to 1 immediately before the timed region

#: Local data RAM, where the BRISC linker script puts ``.data``/``.bss`` and
#: therefore where tt-metal's NoC bookkeeping counters live
#: (``runtime/hw/toolchain/wormhole/firmware_brisc.ld``: ``.data 0xFFB00000``).
COUNTERS_ADDR = 0xFFB00000

NOC0_BASE = 0xFFB20000

#: ``NOC_STATUS(cnt) = NOC_REGS_START_ADDR + 0x200 + cnt * 4``; counter 16 is
#: ``NIU_MST_REQS_OUTSTANDING_ID_0``, which is what a barrier drains.
STATUS_OUTSTANDING = 0x200 + 4 * 16

# ---------------------------------------------------------------------------
# The command-buffer register offsets, per architecture.
# ---------------------------------------------------------------------------
#
# From each architecture's ``noc/noc_parameters.h``. Blackhole inserts
# NOC_AT_LEN_BE_1 at 0x24, which pushes NOC_CMD_CTRL from 0x28 to 0x40, and it
# aliases the COORDINATE registers one slot higher than Wormhole does
# (``NOC_RET_ADDR_COORDINATE = NOC_RET_ADDR_HI`` against Wormhole's
# ``= NOC_RET_ADDR_MID``).

_REG = {
    "wormhole": {
        "TARG_ADDR_LO": 0x00,
        "TARG_ADDR_COORDINATE": 0x04,  # = NOC_TARG_ADDR_MID
        "RET_ADDR_LO": 0x0C,
        "RET_ADDR_COORDINATE": 0x10,  # = NOC_RET_ADDR_MID
        "CTRL": 0x1C,
        "AT_LEN_BE": 0x20,
        "CMD_CTRL": 0x28,
    },
    "blackhole": {
        "TARG_ADDR_LO": 0x00,
        "TARG_ADDR_MID": 0x04,
        "TARG_ADDR_COORDINATE": 0x08,  # = NOC_TARG_ADDR_HI
        "RET_ADDR_LO": 0x0C,
        "RET_ADDR_MID": 0x10,
        "RET_ADDR_COORDINATE": 0x14,  # = NOC_RET_ADDR_HI
        "CTRL": 0x1C,
        "AT_LEN_BE": 0x20,
        "CMD_CTRL": 0x40,
    },
}

#: The per-iteration store list, ``(register name, value source)``, transcribed
#: from ``ncrisc_noc_fast_write`` / ``ncrisc_noc_fast_read`` in each
#: architecture's ``noc_nonblocking_api.h``. Value sources are ``coord``,
#: ``targ``, ``ret``, ``len``, ``zero`` and ``one`` (the last being
#: ``NOC_CTRL_SEND_REQ``).
#:
#: A read does *not* write ``CTRL`` in the loop: that store sits behind
#: ``if constexpr (noc_mode == DM_DYNAMIC_NOC)``, so under the dedicated-NoC
#: mode these kernels build with, the read CTRL is configured once up front.
ISSUE_REGISTERS = {
    ("wormhole", False): (
        ("CTRL", "ctrl"),
        ("TARG_ADDR_LO", "targ"),
        ("RET_ADDR_LO", "ret"),
        ("RET_ADDR_COORDINATE", "coord"),
        ("AT_LEN_BE", "len"),
        ("CMD_CTRL", "one"),
    ),
    ("blackhole", False): (
        ("CTRL", "ctrl"),
        ("TARG_ADDR_LO", "targ"),
        ("RET_ADDR_LO", "ret"),
        ("RET_ADDR_MID", "zero"),
        ("RET_ADDR_COORDINATE", "coord"),
        ("AT_LEN_BE", "len"),
        ("CMD_CTRL", "one"),
    ),
    ("wormhole", True): (
        ("RET_ADDR_LO", "ret"),
        ("TARG_ADDR_LO", "targ"),
        ("TARG_ADDR_COORDINATE", "coord"),
        ("AT_LEN_BE", "len"),
        ("CMD_CTRL", "one"),
    ),
    ("blackhole", True): (
        ("RET_ADDR_LO", "ret"),
        ("TARG_ADDR_LO", "targ"),
        ("TARG_ADDR_MID", "zero"),
        ("TARG_ADDR_COORDINATE", "coord"),
        ("AT_LEN_BE", "len"),
        ("CMD_CTRL", "one"),
    ),
}

#: How many bookkeeping counters the direction increments. A write bumps both
#: ``noc_nonposted_writes_num_issued`` and ``noc_nonposted_writes_acked``; a
#: read bumps only ``noc_reads_num_issued``.
ISSUE_COUNTERS = {False: 2, True: 1}

#: The per-transaction cost this program is expected to cost under
#: ``TT_SIM_COST_MODEL``, ``(arch, is_read) -> cycles``. Not an input to
#: anything -- it is the claim ``noc_issue_loop_test.py`` checks, and it equals
#: the loop's instruction count plus the six-cycle load-use interlock on the
#: ``NOC_CMD_CTRL`` poll.
EXPECTED_CYCLES_PER_TRANSACTION = {
    ("wormhole", False): 22,
    ("blackhole", False): 23,
    ("wormhole", True): 18,
    ("blackhole", True): 19,
}

# Registers, named the way the listing in this module's docstring reads.
_T0, _T1, _T2 = 5, 6, 7  # noc base, counter A addr, counter B addr
_A0, _A1, _A2, _A3 = 10, 11, 12, 13  # coord, targ, ret, len
_A4, _A5, _A6, _A7 = 14, 15, 16, 17  # n, ctrl, i, scratch
_S2, _S3, _S4, _S5 = 18, 19, 20, 21  # const 1, counter A, counter B, params


def instructions_per_transaction(arch, is_read):
    """Instructions in one iteration of the issue loop.

    Two for the ``NOC_CMD_CTRL`` poll, one per command-buffer store, three per
    bookkeeping counter (load, increment, store), one for the induction
    variable and one for the loop branch.
    """
    stores = len(ISSUE_REGISTERS[(arch, is_read)])
    return 2 + stores + 3 * ISSUE_COUNTERS[is_read] + 2


def issue_loop_program(arch, is_read):
    """The whole kernel, as a list of 32-bit words to load at ``PROGRAM_ADDR``.

    Reads its six parameters from :data:`PARAMS_ADDR`, raises
    :data:`START_ADDR`, issues ``n`` transactions, drains the barrier, raises
    :data:`DONE_ADDR` and spins. The driver times ``START -> DONE``, which is
    the same span ``DeviceZoneScopedN`` wraps in the estimator kernels.
    """
    reg = _REG[arch]
    stores = ISSUE_REGISTERS[(arch, is_read)]
    counters = ISSUE_COUNTERS[is_read]
    value_reg = {
        "coord": _A0,
        "targ": _A1,
        "ret": _A2,
        "len": _A3,
        "ctrl": _A5,
        "one": _S2,
        "zero": X0,
    }

    prologue = [
        lui(_T0, NOC0_BASE >> 12),
        lui(_S5, PARAMS_ADDR >> 12),
        lw(_A0, _S5, 0),  # coord
        lw(_A1, _S5, 4),  # target address
        lw(_A2, _S5, 8),  # return address
        lw(_A3, _S5, 12),  # length in bytes
        lw(_A4, _S5, 16),  # transactions per barrier
        lw(_A5, _S5, 20),  # NOC_CTRL command field
        lui(_T1, COUNTERS_ADDR >> 12),
        addi(_T2, _T1, 4),
        addi(_A6, X0, 0),  # i = 0
        addi(_S2, X0, 1),  # the constant 1 / NOC_CTRL_SEND_REQ
    ]
    if is_read:
        # DM_DEDICATED_NOC configures the read command field once, not per
        # transaction: the in-loop CTRL store is DM_DYNAMIC_NOC-only.
        prologue.append(sw(_A5, _T0, reg["CTRL"]))
    prologue.append(sw(_S2, _S5, START_ADDR - PARAMS_ADDR))

    # -- the timed region ---------------------------------------------------
    body = [
        # while (!noc_cmd_buf_ready(noc, cmd_buf));
        lw(_A7, _T0, reg["CMD_CTRL"]),
        bne(_A7, X0, -4),
    ]
    body += [sw(value_reg[src], _T0, reg[name]) for name, src in stores]
    # The bookkeeping counters, scheduled the way the reference build
    # schedules them: both loads first, then the independent induction
    # variable, then the increments, then both stores.
    body.append(lw(_S3, _T1, 0))
    if counters == 2:
        body.append(lw(_S4, _T2, 0))
    body.append(addi(_A6, _A6, 1))
    body.append(addi(_S3, _S3, 1))
    if counters == 2:
        body.append(addi(_S4, _S4, 1))
    body.append(sw(_S3, _T1, 0))
    if counters == 2:
        body.append(sw(_S4, _T2, 0))
    body.append(bne(_A4, _A6, -4 * len(body)))  # back to the poll at body[0]

    barrier = [
        lw(_A7, _T0, STATUS_OUTSTANDING),
        bne(_A7, X0, -4),
    ]
    # -- and it ends here ---------------------------------------------------

    epilogue = [
        sw(_S2, _S5, DONE_ADDR - PARAMS_ADDR),
        jal(X0, 0),  # spin
    ]
    return prologue + body + barrier + epilogue
