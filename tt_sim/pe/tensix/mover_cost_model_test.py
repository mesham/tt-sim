"""The Mover driven with the cycle costs: an issue cost and a transfer time.

Runs standalone (``python3 -m tt_sim.pe.tensix.mover_cost_model_test``) or under
pytest. Companion to ``unpacker_cost_model_test.py``; the two units were wired
together on 2026-08-06.

The Mover is the only unit whose cost the ISA doc splits into two *explicitly
different* quantities, in one sentence: "The thread issuing an ``XMOV``
instruction will be automatically stalled until the mover is able to _start_
work, at which point ``XMOV`` will execute in a single cycle - the mover
proceeds with the task in the background." So the Tensix table's 1 is the issue
cost, and the interesting half — how long the background task runs — is
bandwidth-derived, from the ``mover.transfer`` rates in
``tt_sim/perf/unit_costs.yaml``. Those rates are unusual in these files for
being published as *measured*, with an ideal and a contended column per
transfer kind.

What is pinned here:

* the durations, per transfer kind, against the doc's own arithmetic ("eight
  128b reads and eight 128b writes every 11 cycles i.e. 93.1 bits copied per
  cycle"; one 128b write per cycle for the memsets);
* that the **ideal** column is charged and the contended one is not — the page
  gives no rule for when contention applies, so the ideal rate is the floor;
* that the duration reaches the unit as occupancy, so a second XMOV and a
  queued TDMA command both wait for it, and ``STALLWAIT``'s mover condition
  sees the transfer as outstanding;
* and that with ``TT_SIM_COST_MODEL`` unset every one of those is inert.
"""

import os
import pathlib
from contextlib import contextmanager

import yaml

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.pe.tensix.backends.mover import MoverUnit
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixConfigurationConstants
from tt_sim.perf.model import mover_cost_model

#: The table itself, for the assertions about what the *file* records rather
#: than about what the model charges (the published bits-per-cycle figures, and
#: the contended column being present and unspent). Read as YAML rather than
#: through ``load_costs``: a consumer of the cost tables must reach them
#: through ``tt_sim/perf/model.py``, which
#: ``test_the_consumers_only_reach_the_tables_through_the_model`` enforces.
_UNIT_COSTS_YAML = (
    pathlib.Path(__file__).resolve().parents[2] / "perf" / "unit_costs.yaml"
)

#: ``XMOV``, opcode 0x40, no argument bits set.
XMOV = 0x40 << 24

#: Somewhere in L1 for both ends of a memcpy, 16-byte aligned as the config
#: fields (which are in 16-byte units) require.
SRC = 0x2000
DST = 0x4000


class _L1:
    def __init__(self, size=1 << 20):
        self.data = bytearray(size)

    def read(self, addr, size):
        return bytes(self.data[addr : addr + size])

    def write(self, addr, payload):
        self.data[addr : addr + len(payload)] = payload


def _set_config(backend, key, value, state_id=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


@contextmanager
def _mover(cost_model=True, count=None, mode=MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1):
    """A backend whose mover is configured for one ``count``-byte transfer.

    ``handle_xmov`` reads the *mode* out of ``Destination_address`` (the two
    are the same config field in tt-sim's decode), so the destination doubles
    as the mode selector and this fixture keeps the two consistent the way a
    real kernel's config would.
    """
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if cost_model:
        os.environ["TT_SIM_COST_MODEL"] = "1"
    else:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    try:
        backend = TensixCoProcessor(
            None,
            WORMHOLE_PROFILE.tensix_cfg_state_size,
            WORMHOLE_PROFILE.tensix_thd_state_size,
        ).getBackend()
        backend.setAddressableMemory(_L1())
        if count is not None:
            _set_config(backend, "THCON_SEC0_REG6_Source_address", SRC >> 4)
            _set_config(backend, "THCON_SEC0_REG6_Destination_address", int(mode))
            _set_config(backend, "THCON_SEC0_REG6_Buffer_size", count >> 4)
        yield backend
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def _model(arch="wormhole"):
    """The mover's transfer model, however the env var happens to be set."""
    previous = os.environ.get("TT_SIM_COST_MODEL")
    os.environ["TT_SIM_COST_MODEL"] = "1"
    try:
        return mover_cost_model(arch)
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def _l1_to_l1_rates():
    return yaml.safe_load(_UNIT_COSTS_YAML.read_text())["mover"]["transfer"]["l1_to_l1"]


# ---------------------------------------------------------------------------
# 1. Off is still off.
# ---------------------------------------------------------------------------


def test_without_the_env_var_the_mover_has_no_models():
    with _mover(cost_model=False, count=4096) as backend:
        mover = backend.backend_units["XMOV"]
        assert mover.cost_model is None
        assert mover.transfer_model is None
        assert mover.instruction_occupancy("XMOV", 0) is None
        assert mover.transfer_cycles(MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1, 4096) is (
            None
        )


def test_off_means_a_transfer_completes_in_the_tick_it_was_issued():
    with _mover(cost_model=False, count=4096) as backend:
        mover = backend.backend_units["XMOV"]
        assert mover.issueInstruction(XMOV, 0)
        mover.clock_tick(1)
        assert mover.busy_until is None
        assert not mover.checkForOutstandingInstructions()
        assert mover.issueInstruction(XMOV, 0)


# ---------------------------------------------------------------------------
# 2. The durations, from the published rates.
# ---------------------------------------------------------------------------


def test_the_memcpy_rate_is_the_docs_own_arithmetic():
    """ "Eight 128b reads and eight 128b writes every 11 cycles i.e. 93.1 bits
    copied per cycle" — so 1024 bits move every 11 cycles, and the charge is
    ``ceil(bits * 11 / 1024)``. 1 KiB is 8192 bits, i.e. 88 cycles."""
    model = _model()
    assert model.transfer_cycles("l1_to_l1", 1024) == 88
    assert model.transfer_cycles("l1_to_l1", 128) == 11  # exactly one period
    # ...and the recorded bits-per-cycle figure agrees with the charge to
    # within the rounding: 8192 bits / 93.1 = 87.99.
    l1 = _l1_to_l1_rates()
    assert l1["ideal"]["bits_per_cycle"] == 93.1
    assert round(1024 * 8 / l1["ideal"]["bits_per_cycle"]) == 88


def test_the_memset_rates_are_one_128_bit_write_per_cycle():
    """Both memset kinds ideal at "one 128b write per cycle", so a byte count
    costs ``ceil(bytes / 16)`` cycles."""
    model = _model()
    for kind in ("l1_memset", "non_l1_memset"):
        assert model.transfer_cycles(kind, 1024) == 64, kind
        assert model.transfer_cycles(kind, 16) == 1, kind
        assert model.transfer_cycles(kind, 17) == 2, kind


def test_a_partial_period_is_charged_fractionally_not_rounded_up_to_a_burst():
    """A 16-byte memcpy is 128 bits, an eighth of the 11-cycle period's 1024,
    so it costs 2 cycles rather than the whole 11. Rounding up to whole bursts
    would charge a small transfer a duration the doc never claims for it —
    over-charging, which the floor policy forbids."""
    assert _model().transfer_cycles("l1_to_l1", 16) == 2
    assert _model().transfer_cycles("l1_to_l1", 0) is None
    assert _model().transfer_cycles("nonesuch", 4096) is None


def test_the_contended_column_is_recorded_and_not_charged():
    """The page gives both an ideal and a contended rate and no rule for which
    applies, so the ideal one is the floor and the contended one stays
    unconsumed — 32 bits/cycle would be ~2.9x the charge, i.e. exactly the
    over-charge the bounds policy exists to prevent."""
    l1 = _l1_to_l1_rates()
    assert l1["contended"]["bits_per_cycle"] == 32
    # 1 KiB at the contended rate would be 256 cycles; the model charges 88.
    assert _model().transfer_cycles("l1_to_l1", 1024) == 88


def test_both_arches_price_a_transfer_the_same():
    """The Mover section has no ``arch_overrides`` entry: the ISA doc's Mover
    page is Wormhole's and Blackhole publishes nothing of its own, so the same
    measured rates are read for both rather than being scaled by a clock."""
    assert _model("blackhole").transfer_cycles("l1_to_l1", 1024) == 88


# ---------------------------------------------------------------------------
# 3. The charge reaching the unit.
# ---------------------------------------------------------------------------


def test_an_xmov_is_charged_its_transfer_duration_not_its_issue_cost():
    """The unit's whole point: the table's 1 is issue, the occupancy is the
    background transfer. 4 KiB memcpy = 32768 bits / 1024 per 11 cycles = 352
    cycles, armed from the acceptance cycle (one before the retire tick)."""
    with _mover(count=4096) as backend:
        mover = backend.backend_units["XMOV"]
        assert mover.instruction_occupancy("XMOV", 0) == 352
        assert mover.issueInstruction(XMOV, 0)
        mover.clock_tick(1)
        assert mover.busy_until == 0 + 352
        assert mover.is_occupied()


def test_the_issue_cost_is_the_floor_for_an_unpriceable_transfer():
    """A transfer of no bytes has no duration to charge, so the entry's own
    documented 1-cycle issue cost stands — never a fabricated number, and
    never zero either."""
    with _mover(count=0) as backend:
        mover = backend.backend_units["XMOV"]
        assert mover.instruction_occupancy("XMOV", 0) == 1
        assert mover.issueInstruction(XMOV, 0)
        mover.clock_tick(1)
        # occupy_for no-ops at <= 1 cycle, so a 1-cycle charge holds nothing.
        assert mover.busy_until is None


def test_a_transfer_in_flight_refuses_the_next_xmov():
    """ "Automatically stalled until the mover is able to start work" — which is
    exactly the existing issue-refusal path, so no new mechanism was needed."""
    with _mover(count=4096) as backend:
        mover = backend.backend_units["XMOV"]
        assert mover.issueInstruction(XMOV, 0)
        mover.clock_tick(1)
        for thread in range(3):
            assert not mover.issueInstruction(XMOV, thread)
        assert mover.next_wake_cycle(2) == 352
        mover.clock_tick(352)
        assert mover.busy_until is None
        assert mover.issueInstruction(XMOV, 1)


def test_a_transfer_in_flight_reads_as_outstanding_work():
    """What ``STALLWAIT``'s mover condition (Wormhole C12 / Blackhole C9) and
    the TDMA "mover wait" command poll. A background transfer that the thread
    can wait for must be visible as outstanding for as long as it runs."""
    with _mover(count=4096) as backend:
        mover = backend.backend_units["XMOV"]
        assert not mover.checkForOutstandingInstructions()
        assert mover.issueInstruction(XMOV, 0)
        mover.clock_tick(1)
        assert mover.checkForOutstandingInstructions()
        assert not mover.is_clock_idle() or mover.busy_until is not None
        mover.clock_tick(352)
        assert not mover.checkForOutstandingInstructions()


def test_a_tdma_queued_transfer_is_charged_and_waits_its_turn():
    """The TDMA path bypasses the instruction machinery entirely (the mover's
    ``clock_tick`` pops those commands before reaching the base drain), so it
    is charged where it runs — and a second queued command waits for the first
    rather than both landing in consecutive ticks."""
    with _mover(count=None) as backend:
        mover = backend.backend_units["XMOV"]
        command = (DST, SRC, 1024, MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1)
        mover.append_command_from_tdma(command)
        mover.append_command_from_tdma(command)
        mover.clock_tick(0)
        assert mover.busy_until == 88  # 1 KiB memcpy
        assert len(mover.tdma_commands) == 1
        for cycle in range(1, 88):
            mover.clock_tick(cycle)
            assert len(mover.tdma_commands) == 1, cycle
        mover.clock_tick(88)
        assert not mover.tdma_commands
        assert mover.busy_until == 88 + 88


def test_the_tdma_path_is_untouched_with_the_model_off():
    with _mover(cost_model=False, count=None) as backend:
        mover = backend.backend_units["XMOV"]
        command = (DST, SRC, 1024, MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1)
        mover.append_command_from_tdma(command)
        mover.append_command_from_tdma(command)
        mover.clock_tick(0)
        mover.clock_tick(1)
        assert not mover.tdma_commands
        assert mover.busy_until is None


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "mover_cost_model_test OK: memcpy 1 KiB at 88 cycles, memsets at 16 "
        "B/cycle, contended column unconsumed, TDMA queue charged, off by default"
    )


if __name__ == "__main__":
    main()
