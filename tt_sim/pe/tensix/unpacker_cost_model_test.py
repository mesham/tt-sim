"""The unpacker driven with the cycle costs: the one non-constant charge.

Runs standalone (``python3 -m tt_sim.pe.tensix.unpacker_cost_model_test``) or
under pytest. Companion to ``backend_cost_model_test.py`` (the five
constant-cost units), ``matrix_cost_model_test.py`` (the fidelity-scaled FPU)
and ``mover_cost_model_test.py``.

The unpacker is the unit ``docs/plans/cost-model.md`` has called the clearest
case in the file of a cost that is *not* a per-opcode constant, and this pins
both halves of it:

1. **The address phase** — ``UNPACR_Regular.md``: "An ``UNPACR`` instruction
   spends at least two cycles calculating the initial input address:
   uncompressed data requires exactly two cycles". Charged at the low end of
   its ``at_least``, i.e. 2, which is *exact* for the only kind of data tt-sim
   unpacks.
2. **The data phase** — the same section: "execution proceeds in a pipelined
   fashion, with the primary bottleneck being the fetching of bytes from L1",
   at x1 = 16, x2 = 32 or x4 = 64 bytes per cycle as
   ``THCON_SEC[n].Throttle_mode`` selects. So the charge is
   ``ceil(bytes / rate)``, computed at decode from the transfer this UNPACR
   actually describes — the thing a flat lookup cannot express.

Plus the two properties that make the charge safe rather than merely present:
a **blocked** unpacker (waiting thousands of cycles for a Src bank to come
back) is charged for waiting exactly nothing, and its address phase is charged
once rather than once per re-run; and the joint 80 B/cycle ceiling across the
two unpackers stays **unconsumed**, because no source states how it divides
per transfer.
"""

import os
import pathlib
from contextlib import contextmanager

import pytest
import yaml

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixConfigurationConstants
from tt_sim.perf.model import unit_cost_model

#: The table itself, for the two assertions that are about what the *file*
#: says rather than about what the model charges (the ``at_least`` bound on the
#: address phase, and the joint ceiling being recorded as unconsumed). Read as
#: YAML rather than through ``load_costs``, because a consumer of the cost
#: tables must reach them through ``tt_sim/perf/model.py`` and nothing else —
#: ``test_the_consumers_only_reach_the_tables_through_the_model`` enforces it.
_COSTS_YAML = pathlib.Path(__file__).resolve().parent / "tensix_instruction_costs.yaml"

BF16 = 5  # DataFormat.BF16, as it appears in a tile descriptor / REG2
FP32 = 0
INT8 = 14

#: Where the tile lives in L1, and the ``REG3_Base_address`` naming it
#: (``InAddr = (Base + Offset + 1) * 16``).
IN_ADDR = 0x1000
BASE_ADDRESS = IN_ADDR // 16 - 1

#: ``UNPACR`` with every argument bit clear: unpacker 0 (bit 23), the regular
#: datum-moving form (SearchCacheFlush at bit 1 and CfgContextCntInc at 13 both
#: clear), no dvalid flip. Opcode 0x42.
UNPACR = 0x42 << 24

#: Throttle mode values, from ``UNPACR_Regular.md``: "0 means x1, 1 means x2,
#: and 2 means x4"; Blackhole adds 3 = x8 ("x8 ThrottleMode is illegal on
#: Wormhole").
THROTTLE_X1, THROTTLE_X2, THROTTLE_X4, THROTTLE_X8 = 0, 1, 2, 3

#: The published rates those select, in bytes per cycle.
RATE = {THROTTLE_X1: 16, THROTTLE_X2: 32, THROTTLE_X4: 64, THROTTLE_X8: 128}


class _L1:
    """Just enough addressable memory for the unpacker to read a tile out of."""

    def __init__(self, size=1 << 20):
        self.data = bytearray(size)

    def read(self, addr, size):
        return bytes(self.data[addr - IN_ADDR : addr - IN_ADDR + size])

    def write(self, addr, payload):
        self.data[addr - IN_ADDR : addr - IN_ADDR + len(payload)] = payload


def _set_config(backend, key, value, state_id=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


def _set_thread_config(backend, key, value, thread=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    cfg = backend.config_unit.threadConfig[thread]
    cfg[addr32] = (cfg[addr32] & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)


@contextmanager
def _unpacker(
    cost_model=True,
    blackhole=False,
    throttle=THROTTLE_X1,
    num_rows=4,
    data_format=BF16,
    tileize=0,
    ovrd_default_throttle=1,
):
    """A backend whose unpacker 0 is set up for one ``num_rows`` x 16 unpack.

    ``ovrd_default_throttle`` is the Blackhole-only
    ``THCON_SEC0_REG1_ovrd_default_throttle_mode`` bit; set (the default here)
    means the configured ``Throttle_mode`` is honoured, clear means the doc's
    "8-bit modes use x8, others use x4" defaults take over.
    """
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if cost_model:
        os.environ["TT_SIM_COST_MODEL"] = "1"
    else:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    try:
        backend = TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size if blackhole else None,
            BLACKHOLE_PROFILE.tensix_thd_state_size if blackhole else None,
            blackhole=blackhole,
        ).getBackend()
        memory = _L1()
        backend.setAddressableMemory(memory)

        descriptor = TensixConfigurationConstants.get_addr32(
            "THCON_SEC0_REG0_TileDescriptor"
        )
        backend.config_unit.setConfig(0, descriptor + 0, (16 << 16) | data_format)
        backend.config_unit.setConfig(0, descriptor + 1, (1 << 16) | 1)
        backend.config_unit.setConfig(0, descriptor + 2, 1)
        backend.config_unit.setConfig(0, descriptor + 3, 0)

        _set_config(backend, "THCON_SEC0_REG3_Base_address", BASE_ADDRESS)
        _set_config(backend, "THCON_SEC0_REG2_Out_data_format", data_format)
        _set_config(backend, "THCON_SEC0_REG2_Throttle_mode", throttle)
        _set_config(backend, "THCON_SEC0_REG2_Tileize_mode", tileize)
        if tileize:
            # A contiguous RowStride, so the walk is the plain one: the stride
            # field is in 16-byte units at shift 4.
            row_width = 32 if (blackhole and data_format != INT8) else 16
            stride = row_width * (
                1 if data_format == INT8 else (4 if data_format == FP32 else 2)
            )
            _set_config(backend, "THCON_SEC0_REG2_Shift_amount_cntx0", stride >> 4)
        if blackhole:
            _set_config(
                backend,
                "THCON_SEC0_REG1_ovrd_default_throttle_mode",
                ovrd_default_throttle,
            )
        # OutAddr must come out 64 datums (SrcA start row 4 - 4 = 0), and the
        # unpacker divides the configured byte address by the output datum
        # width, so the base scales with the format.
        _set_config(
            backend, "UNP0_ADDR_BASE_REG_1_Base", 64 * _datum_bytes(data_format)
        )
        _set_thread_config(backend, "SRCA_SET_SetOvrdWithAddr", 1)

        # ADC channel 1 X is the last input datum index (inclusive).
        backend.getADC(0).Unpacker[0].Channel[1].X = num_rows * 16 - 1
        yield backend, memory
    finally:
        TensixConfigurationConstants.use_blackhole(False)
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def _issue_and_tick(backend, cycle=1):
    """Issue one UNPACR to unpacker 0 and retire it, as the frontend would."""
    unpacker = backend.unpacker_units[0]
    assert unpacker.issueInstruction(UNPACR, 0)
    unpacker.clock_tick(cycle)
    return unpacker


def _datum_bytes(data_format):
    return {BF16: 2, FP32: 4, INT8: 1}[data_format]


def _model(arch="wormhole"):
    """The unpacker's cost model, however the env var happens to be set."""
    previous = os.environ.get("TT_SIM_COST_MODEL")
    os.environ["TT_SIM_COST_MODEL"] = "1"
    try:
        return unit_cost_model("UNPACK", arch)
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def _unpack_table():
    return yaml.safe_load(_COSTS_YAML.read_text())["units"]["UNPACK"]


# ---------------------------------------------------------------------------
# 1. Off is still off.
# ---------------------------------------------------------------------------


def test_without_the_env_var_the_unpacker_has_no_cost_model():
    with _unpacker(cost_model=False) as (backend, _memory):
        for unpacker in backend.unpacker_units:
            assert unpacker.cost_model is None
            assert unpacker.instruction_occupancy("UNPACR", 0) is None
            assert unpacker.instruction_occupancy("UNPACR_NOP", 0) is None


def test_off_means_an_unpack_retires_in_the_tick_it_was_issued():
    with _unpacker(cost_model=False) as (backend, _memory):
        unpacker = _issue_and_tick(backend)
        assert unpacker.busy_until is None
        assert not unpacker.is_occupied()
        # ...and the next one goes straight in, as it always did.
        assert unpacker.issueInstruction(UNPACR, 0)


# ---------------------------------------------------------------------------
# 2. The address phase.
# ---------------------------------------------------------------------------


def test_the_address_phase_is_the_documented_two_cycles_at_its_low_end():
    """ ">= 2 cycles ... uncompressed data requires exactly two cycles".

    The bound is charged at its low end per ``BOUND_POLICY``, and here the low
    end is not merely a floor: 2 is the *exact* figure for uncompressed data,
    which is the only kind tt-sim unpacks (``get_isUncompressed`` returns True
    unconditionally, and the compressed walk raises).
    """
    model = _model()
    assert model.occupancy("UNPACR") == 2
    assert not model.is_exact("UNPACR")  # the entry itself is an ">= 2"
    entry = _unpack_table()["instructions"]["UNPACR"]["occupancy"]
    assert entry == {"cycles": 2, "bound": "at_least"}


def test_the_address_phase_holds_the_unit_and_deliberately_not_the_thread():
    """The sentence's other two halves are both recorded gaps, for one reason.

    ``UNPACR_Regular.md`` says the address phase blocks three things: this
    unpacker (charged, as ``busy_until``), *the issuing thread* ("the issuing
    thread cannot start its next instruction"), and *every* thread's `UNPACR`
    ("nor can any other thread start an ``UNPACR``"). Only the first is
    charged. The third needs a cross-unpacker arbitration rule no source
    gives, exactly as the joint 80 B/cycle ceiling does.

    The second is sourced, and was still not charged (2026-08-11, ROADMAP item
    3, which *did* wire the Scalar Unit's identically-shaped interlock through
    ``TensixBackend.block_thread_issue``). Charging it turns three Wormhole
    value guards red -- ``softplus`` / ``tilize`` / ``untilize`` all raise
    ``SrcDvalidError`` -- and the cause is on the other side of the handshake:
    tt-sim performs an unpack and flips its Src ``AllowedClient`` in the tick
    the instruction retires, where hardware flips it at the *end* of a
    pipelined transfer. The LLK datacopy loop leans on a one-cycle margin
    between the unpack thread's ``UNPACR_NOP`` dvalid set and the math
    thread's ``SETRWC`` release, and a correctly anchored two-cycle interlock
    spends exactly that margin. So the blocker is a mechanism tt-sim does not
    have, not a number nobody published, and it is recorded here rather than
    shipped as a value regression.
    """
    with _unpacker(throttle=THROTTLE_X1, num_rows=4) as (backend, _memory):
        unpacker = _issue_and_tick(backend)
        assert unpacker.busy_until == 0 + 2 + 8  # address phase + 128 B / 16
        assert backend.thread_issue_block == [0, 0, 0]


def test_the_base_hook_declines_unpacr_so_the_handler_can_price_it():
    """``UNPACR`` is three instruction forms behind one opcode, and only the
    datum-moving one has published timing — so the flat pre-handler lookup is
    refused and the charge is computed at decode instead. ``UNPACR_NOP`` keeps
    the lookup, because its cost really is a published constant."""
    with _unpacker() as (backend, _memory):
        unpacker = backend.unpacker_units[0]
        assert unpacker.cost_model is not None
        assert unpacker.instruction_occupancy("UNPACR", 0) is None
        assert unpacker.instruction_occupancy("UNPACR_NOP", 0) == 1


def test_the_context_counter_and_flush_forms_are_charged_nothing():
    """Only ``UNPACR_Regular.md`` has a Performance section. The increment and
    flush forms move no datums and publish no timing, so they keep the
    same-cycle retire rather than borrowing the regular form's two cycles."""
    with _unpacker() as (backend, _memory):
        unpacker = backend.unpacker_units[0]
        for bit in (1, 13):  # SearchCacheFlush, CfgContextCntInc
            assert unpacker.issueInstruction(UNPACR | (1 << bit), 0)
            unpacker.clock_tick(1)
            assert unpacker.busy_until is None, bit


# ---------------------------------------------------------------------------
# 3. The data phase, at each throttle rate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("throttle", [THROTTLE_X1, THROTTLE_X2, THROTTLE_X4])
@pytest.mark.parametrize("num_rows", [1, 4, 16])
def test_the_data_phase_is_bytes_over_the_configured_throttle_rate(throttle, num_rows):
    """The whole point of the exercise: the charge follows the transfer.

    ``num_rows`` x 16 BF16 datums is ``num_rows * 32`` bytes, over 16 / 32 / 64
    bytes per cycle — so the *same* instruction costs four different numbers
    depending on two things a per-opcode table cannot see.
    """
    transfer_bytes = num_rows * 16 * 2
    expected = 2 + -(-transfer_bytes // RATE[throttle])  # address + ceil(bytes/rate)
    with _unpacker(throttle=throttle, num_rows=num_rows) as (backend, _memory):
        unpacker = _issue_and_tick(backend, cycle=1)
        # Armed from the acceptance cycle (one before the retire tick), as
        # every other unit's occupancy is.
        assert unpacker.busy_until == 0 + expected


def test_the_rates_are_the_published_sixteen_thirtytwo_sixtyfour():
    """Read off the model rather than through a backend, so the arithmetic is
    visible: "x1 speed: Up to 16 bytes per cycle from L1", x2 32, x4 64."""
    model = _model()
    for throttle, rate in ((THROTTLE_X1, 16), (THROTTLE_X2, 32), (THROTTLE_X4, 64)):
        # One byte over a whole number of cycles' worth rounds up, because a
        # partially-used cycle is still a cycle.
        assert model.unpack_data_phase_cycles(rate * 3, throttle, 2) == 3
        assert model.unpack_data_phase_cycles(rate * 3 + 1, throttle, 2) == 4
        assert model.unpack_data_phase_cycles(1, throttle, 2) == 1


def test_a_wider_datum_moves_more_bytes_and_costs_more():
    """The transfer size is datums x datum bytes, so an FP32 unpack of the same
    datum count costs twice a BF16 one at the same throttle."""
    with _unpacker(throttle=THROTTLE_X2, num_rows=4, data_format=BF16) as (be, _m):
        bf16 = _issue_and_tick(be).busy_until
    with _unpacker(throttle=THROTTLE_X2, num_rows=4, data_format=FP32) as (be, _m):
        fp32 = _issue_and_tick(be).busy_until
    # 128 B and 256 B at 32 B/cycle, plus the 2-cycle address phase each.
    assert bf16 == 2 + 4
    assert fp32 == 2 + 8


def test_tileize_forces_x4_whatever_the_config_says():
    """ "tileize always runs at x4, regardless of Throttle_mode" — the one
    forced mode tt-sim can reach, since the others (compressed data,
    UpsampleZeroes, BFP2) force modes of unpacks it rejects at decode."""
    model = _model()
    assert model.unpack_data_phase_cycles(256, THROTTLE_X1, 2) == 16  # 256/16
    assert model.unpack_data_phase_cycles(256, THROTTLE_X1, 2, tileize=True) == 4
    # ...and end to end, through a real strided-mode backend.
    with _unpacker(throttle=THROTTLE_X1, num_rows=8, tileize=1) as (backend, _memory):
        # 8 rows x 16 BF16 datums = 256 bytes, at x4 rather than the configured
        # x1: 4 cycles, not 16.
        assert _issue_and_tick(backend).busy_until == 2 + 4


def test_an_untabulated_throttle_mode_charges_no_data_phase():
    """No entry means no opinion, here as everywhere. Wormhole's page says
    outright that "x8 ThrottleMode is illegal on Wormhole", so mode 3 has no
    rate there and the unpack is charged its address phase alone rather than a
    guessed one."""
    model = _model("wormhole")
    assert model.unpack_data_phase_cycles(256, THROTTLE_X8, 2) is None
    with _unpacker(throttle=THROTTLE_X8, num_rows=4) as (backend, _memory):
        assert _issue_and_tick(backend).busy_until == 2


# ---------------------------------------------------------------------------
# 4. Blackhole's own rates, which the shared page states in its pseudocode.
# ---------------------------------------------------------------------------


def test_blackhole_adds_x8_and_wormhole_does_not_get_it():
    """ "x8 ThrottleMode is illegal on Wormhole" — so the 128 B/cycle rate lives
    in the arch override and cannot leak into the Wormhole table."""
    assert _model("blackhole").unpack_data_phase_cycles(256, THROTTLE_X8, 1) == 2
    assert _model("wormhole").unpack_data_phase_cycles(256, THROTTLE_X8, 1) is None


def test_blackhole_upgrades_x4_to_128_bytes_for_wide_datums():
    """ "ThrottleBytes = 128; // upgrade to x4 '2x'" when the architecture is
    Blackhole, the mode is x4 and ``DatumSizeBytes >= 2``. A one-byte datum
    stays at the shared 64."""
    bh = _model("blackhole")
    assert bh.unpack_data_phase_cycles(256, THROTTLE_X4, 2) == 2  # 256 / 128
    assert bh.unpack_data_phase_cycles(256, THROTTLE_X4, 4) == 2
    assert bh.unpack_data_phase_cycles(256, THROTTLE_X4, 1) == 4  # 256 / 64
    # Wormhole is untouched by any of it.
    assert _model("wormhole").unpack_data_phase_cycles(256, THROTTLE_X4, 4) == 4


def test_blackholes_default_throttle_bit_selects_x8_or_x4():
    """ "ThrottleMode = (DatumSizeBytes == 1) ? 3 : 2; // 8-bit modes use x8,
    others use x4", taken when ``THCON_SEC[0].REG1_ovrd_default_throttle_mode``
    is clear — which is what tt-metal's LLK leaves it as. The configured mode
    is ignored entirely in that case."""
    bh = _model("blackhole")
    # Configured x1, but the override bit is clear: a 2-byte datum runs at x4
    # upgraded to 128 B/cycle, a 1-byte datum at x8 = 128.
    assert (
        bh.unpack_data_phase_cycles(
            512, THROTTLE_X1, 2, default_throttle_overridden=False
        )
        == 4
    )
    assert (
        bh.unpack_data_phase_cycles(
            512, THROTTLE_X1, 1, default_throttle_overridden=False
        )
        == 4
    )
    # With the bit set the configured x1 is honoured: 512 / 16.
    assert bh.unpack_data_phase_cycles(512, THROTTLE_X1, 2) == 32


def test_the_blackhole_backend_reads_the_default_throttle_bit():
    """End to end, on a real Blackhole backend: the same UNPACR costs the x1 it
    configured with the override bit set, and the doc's default x4 "2x" without
    it."""
    with _unpacker(blackhole=True, throttle=THROTTLE_X1, num_rows=8) as (be, _m):
        assert _issue_and_tick(be).busy_until == 2 + 16  # 256 B / 16
    with _unpacker(
        blackhole=True, throttle=THROTTLE_X1, num_rows=8, ovrd_default_throttle=0
    ) as (be, _m):
        assert _issue_and_tick(be).busy_until == 2 + 2  # 256 B / 128


# ---------------------------------------------------------------------------
# 5. Blocking, which is where this could most easily have gone wrong.
# ---------------------------------------------------------------------------


def test_a_blocked_unpacker_is_waiting_not_busy():
    """The unpacker legitimately blocks for thousands of cycles waiting for the
    Matrix Unit to hand a Src bank back (worst measured in-tree: 3,528 cycles).
    That wait is the unit *waiting on somebody else*, and it is already modelled
    functionally — charging occupancy for it as well would double-count the
    stall and could wedge a unit that is making perfectly good progress.

    So: while blocked, only the address phase is held (it happens before the
    wait — "spends at least two cycles calculating the initial input address",
    then "the primary bottleneck being the fetching of bytes from L1"), and the
    per-cycle re-runs charge nothing at all.
    """
    with _unpacker(throttle=THROTTLE_X1, num_rows=4) as (backend, _memory):
        unpacker = backend.unpacker_units[0]
        backend.getSrcA(unpacker.srcBank).setAllowedClient(
            SrcRegister.SrcClient.MatrixUnit
        )
        assert unpacker.issueInstruction(UNPACR, 0)
        unpacker.clock_tick(1)
        assert unpacker.blocked
        # The address phase, once — and nothing more, however long the wait.
        assert unpacker.busy_until == 0 + 2
        for cycle in range(2, 40):
            unpacker.clock_tick(cycle)
            assert unpacker.blocked
            assert unpacker.busy_until is None, cycle
        # The bank comes back: the data phase is charged from the cycle the
        # transfer actually starts, and the address phase is NOT charged again.
        backend.getSrcA(unpacker.srcBank).setAllowedClient(
            SrcRegister.SrcClient.Unpackers
        )
        unpacker.clock_tick(40)
        assert not unpacker.blocked
        assert unpacker.busy_until == 40 + 8  # 128 B at 16 B/cycle


def test_a_blocked_unpacker_still_reports_its_thread_as_inflight():
    """The property the deadlock machinery reads. An occupancy hold must not
    make a blocked unpacker look done, and must not make a *finished* one look
    blocked: ``blocked_on`` stays the authority on which Src bank is awaited."""
    with _unpacker() as (backend, _memory):
        unpacker = backend.unpacker_units[0]
        backend.getSrcA(unpacker.srcBank).setAllowedClient(
            SrcRegister.SrcClient.MatrixUnit
        )
        assert unpacker.issueInstruction(UNPACR, 0)
        unpacker.clock_tick(1)
        assert unpacker.hasInflightInstructionsFromThread(0)
        assert unpacker.blocked_on() == (0, "UNPACR", "SrcA", 0)
        assert not unpacker.is_clock_idle()


def test_an_occupied_unpacker_refuses_the_next_unpack():
    """The back-pressure itself, which is what the charge is *for*: with the
    front-end FIFO bounded, a refused issue reaches the issuing baby core."""
    with _unpacker(throttle=THROTTLE_X1, num_rows=16) as (backend, _memory):
        unpacker = _issue_and_tick(backend, cycle=1)
        assert unpacker.busy_until == 2 + 32  # 512 B at 16 B/cycle
        assert unpacker.is_occupied()
        for thread in range(3):
            assert not unpacker.issueInstruction(UNPACR, thread)
        # The pump is told when to come back rather than spinning.
        assert unpacker.next_wake_cycle(5) == 34
        unpacker.clock_tick(34)
        assert unpacker.busy_until is None
        # Thread 1 rather than thread 0: three threads were refused while the
        # unit was held, and the freed slot rotates to the least recently
        # granted of them rather than back to the thread that had it (the
        # anti-starvation grant in ``TensixBackendUnit.issueInstruction``).
        assert unpacker.issueInstruction(UNPACR, 1)


def test_the_other_unpacker_is_untouched_by_the_first_ones_hold():
    """The under-charge this wiring knowingly leaves, pinned so it cannot be
    mistaken for an accident.

    The doc's address-phase sentence ends "nor can any other thread start an
    ``UNPACR`` instruction" — a cross-unpacker constraint — and the joint
    80 B/cycle L1 ceiling is likewise shared. Neither is modelled: the hold is
    per unit, so unpacker 1 issues freely while unpacker 0 is held. Both are
    floors (the model charges less than the hardware), which is the direction
    the bounds policy requires, and both are recorded in the table.
    """
    with _unpacker(throttle=THROTTLE_X1, num_rows=16) as (backend, _memory):
        first = _issue_and_tick(backend, cycle=1)
        assert first.is_occupied()
        second = backend.unpacker_units[1]
        assert second.issueInstruction(UNPACR | (1 << 23), 0)


def test_the_joint_ceiling_is_recorded_and_not_charged():
    """80 B/cycle is in the table at ``vendor_source`` and nothing reads it: it
    is a *shared* limit over two simultaneously streaming unpackers, and no
    source states how it divides per transfer. The 3x3 interference table is
    the same shape of fact — sustained rates, not an arbitration rule — so it
    is unconsumed too, and each unpacker is charged its own uncontended rate."""
    joint = _unpack_table()["joint_bandwidth"]
    assert joint["bytes_per_cycle"] == 80
    assert "UNCONSUMED" in joint["note"].upper()
    # The charge at x4 is the per-unpacker 64, never the joint 80.
    assert _model().unpack_data_phase_cycles(640, THROTTLE_X4, 2) == 10


def main():
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        if not marks:
            fn()
            continue
        # Two parametrize decorators at most, applied outermost-last.
        values = [
            [(mark.args[0], value) for value in mark.args[1]]
            for mark in marks
            if mark.name == "parametrize"
        ]
        for first in values[0]:
            if len(values) == 1:
                fn(**dict([first]))
            else:
                for second in values[1]:
                    fn(**dict([first, second]))
    print(
        "unpacker_cost_model_test OK: address phase 2, data phase bytes/rate at "
        "x1/x2/x4 (+ Blackhole x8 and the x4 '2x' upgrade), blocking charged "
        "nothing, joint ceiling unconsumed"
    )


if __name__ == "__main__":
    main()
