"""Tests for the Blackhole-only non-SFPU Tensix ops (CFGSHIFTMASK / MOVDBGB2D /
RESOURCEDECL / STREAMWAIT / STREAMWRCFG).

Runs standalone (``python3 -m tt_sim.pe.tensix.blackhole_ops_test``) or under
pytest. Checks the decoder recognises the new opcodes and pins each op's
argument bit layout (the vendor tt-llk Blackhole ``assembly.yaml`` /  ttsim
``data/bh/tensix_isa.json`` ranges), then drives a real backend so CFGSHIFTMASK
is exercised exactly as a decoded instruction reaches it. The other four ops are
decoded but unmodelled — the vendor reference simulator does not implement them
either — so what is pinned there is that they fail loudly rather than silently
no-op.
"""

from contextlib import contextmanager

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import (
    TensixConfigurationConstants,
    TensixInstructionDecoder,
)

CFGSHIFTMASK = 0xB8
MOVDBGB2D = 0x0C
RESOURCEDECL = 0x05
STREAMWAIT = 0xA7
STREAMWRCFG = 0xB7


@contextmanager
def _blackhole_backend():
    """A Blackhole-configured Tensix backend.

    The config-register layout is a process-global selection, so restore the
    Wormhole layout on the way out — other tests in the same session (and the
    Wormhole replay guards) expect it.
    """
    try:
        yield TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size,
            BLACKHOLE_PROFILE.tensix_thd_state_size,
            blackhole=True,
        ).getBackend()
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _cfgshiftmask(
    cfg_reg,
    scratch_sel=0,
    right_cshift_amt=0,
    mask_width=31,
    operation=3,
    disable_mask_on_old_val=1,
):
    return (
        (CFGSHIFTMASK << 24)
        | (disable_mask_on_old_val << 23)
        | (operation << 20)
        | (mask_width << 15)
        | (right_cshift_amt << 10)
        | (scratch_sel << 8)
        | cfg_reg
    )


#: How many cycles :func:`_issue` will wait for the config unit to free itself.
#: Three is enough for the one multi-cycle opcode in either arch's table; eight
#: leaves room and still fails fast rather than spinning.
_ISSUE_RETRY_LIMIT = 8


def _issue(backend, instruction, thread=0):
    """Issue through the backend and tick the config unit until it retires.

    The cycle number **advances**; it used to be pinned at 0. That was fine
    while no Tensix unit could hold itself, and stopped being fine when the
    config unit was wired to the cycle-cost tables: Blackhole's
    ``CFGSHIFTMASK`` is a 2-cycle occupancy (``throughput_ipc: 0.5``, "requires
    two cycles in stage 0"), so after one executes the unit refuses further
    issues until ``busy_until`` — and a helper that always ran cycle 0 could
    never let that deadline pass, which made the second ``CFGSHIFTMASK`` in a
    test refused for ever under ``TT_SIM_COST_MODEL=1``.

    Retrying on the next cycle is what the frontend's wait gate does on a real
    run, and it is what the replay guards exercise. Nothing these tests assert
    changes: the back-pressure is throughput, so the instruction still executes
    and still has exactly the same effect, one or three cycles later.
    """
    cycle = getattr(backend, "_ops_test_cycle", 0)
    for _ in range(_ISSUE_RETRY_LIMIT):
        accepted = backend.issueInstruction(instruction, thread)
        backend.config_unit.clock_tick(cycle)
        cycle += 1
        if accepted:
            backend._ops_test_cycle = cycle
            return
    raise AssertionError(
        f"config unit refused the instruction for {_ISSUE_RETRY_LIMIT} cycles"
    )


def _set_config(backend, state_id, key, value):
    backend.config_unit.setConfig(
        state_id, TensixConfigurationConstants.get_addr32(key), value
    )


def test_decoder_recognises_blackhole_ops():
    for opcode, name in [
        (RESOURCEDECL, "RESOURCEDECL"),
        (MOVDBGB2D, "MOVDBGB2D"),
        (STREAMWAIT, "STREAMWAIT"),
        (STREAMWRCFG, "STREAMWRCFG"),
        (CFGSHIFTMASK, "CFGSHIFTMASK"),
    ]:
        instruction = opcode << 24
        assert TensixInstructionDecoder.isInstructionRecognised(instruction)
        assert TensixInstructionDecoder.getInstructionInfo(instruction)["name"] == name


def test_argument_bit_layouts():
    # An all-ones payload pins every field's width; the widths are the vendor
    # bit ranges (CfgReg 7:0, scratch_sel 9:8, ... for CFGSHIFTMASK, etc.).
    expected = {
        CFGSHIFTMASK: {
            "CfgReg": 0xFF,
            "scratch_sel": 0x3,
            "right_cshift_amt": 0x1F,
            "mask_width": 0x1F,
            "operation": 0x7,
            "disable_mask_on_old_val": 0x1,
        },
        MOVDBGB2D: {
            "dst": 0x7FF,
            "movb2d_instr_mod": 0x7,
            "addr_mode": 0x7,
            "src": 0x3F,
            "dest_32b_lo": 0x1,
        },
        RESOURCEDECL: {
            "op_class": 0xF,
            "resources": 0x1FF,
            "linger_time": 0xF,
            "reserved": 0x7F,
        },
        STREAMWAIT: {
            "wait_stream_sel": 0x7,
            "target_sel": 0x1,
            "target_value": 0x3FF,
            "reserved": 0x1,
            "stall_res": 0x1FF,
        },
        STREAMWRCFG: {
            "CfgReg": 0x7FF,
            "StreamRegAddr": 0x3FF,
            "stream_id_sel": 0x7,
        },
    }
    for opcode, fields in expected.items():
        info = TensixInstructionDecoder.getInstructionInfo((opcode << 24) | 0xFFFFFF)
        assert info["instr_args"] == fields, info["name"]


def test_cfgshiftmask_adds_scratch_into_config_register():
    with _blackhole_backend() as backend:
        _set_config(backend, 0, "SCRATCH_SEC1_val", 0x1000)
        backend.config_unit.setConfig(0, 41, 0x25)
        _issue(backend, _cfgshiftmask(41, scratch_sel=1))
        assert backend.config_unit.get_config_entry(0, 41) == 0x1025


def test_cfgshiftmask_scratch_sel_3_selects_by_thread():
    with _blackhole_backend() as backend:
        for thread in range(3):
            _set_config(backend, 0, f"SCRATCH_SEC{thread}_val", 1 << thread)
            backend.config_unit.setConfig(0, 60 + thread, 0)
        for thread in range(3):
            _issue(backend, _cfgshiftmask(60 + thread, scratch_sel=3), thread=thread)
        for thread in range(3):
            assert backend.config_unit.get_config_entry(0, 60 + thread) == 1 << thread


def test_cfgshiftmask_addition_wraps_to_32_bits():
    with _blackhole_backend() as backend:
        _set_config(backend, 0, "SCRATCH_SEC0_val", 0x8000_0000)
        backend.config_unit.setConfig(0, 41, 0x8000_0007)
        _issue(backend, _cfgshiftmask(41))
        assert backend.config_unit.get_config_entry(0, 41) == 7


def test_cfgshiftmask_unmodelled_modes_raise():
    # Only the mode ttsim models is implemented; every other shift / mask /
    # operation combination must fail loudly rather than mis-execute.
    for kwargs in [
        {"right_cshift_amt": 6},
        {"mask_width": 17},
        {"operation": 0},
        {"disable_mask_on_old_val": 0},
    ]:
        with _blackhole_backend() as backend:
            with pytest.raises(NotImplementedError):
                _issue(backend, _cfgshiftmask(41, **kwargs))


def test_cfgshiftmask_rejected_on_wormhole():
    backend = TensixCoProcessor(None).getBackend()
    with pytest.raises(NotImplementedError):
        _issue(backend, _cfgshiftmask(41))


def test_unmodelled_blackhole_ops_fail_loudly():
    # Including RESOURCEDECL, whose ex_resource is NONE and which would
    # otherwise be silently ignored like a NOP.
    for opcode in [MOVDBGB2D, RESOURCEDECL, STREAMWAIT, STREAMWRCFG]:
        with _blackhole_backend() as backend:
            with pytest.raises(NotImplementedError):
                backend.issueInstruction(opcode << 24, 0)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "blackhole_ops_test OK: CFGSHIFTMASK modelled; "
        "MOVDBGB2D/RESOURCEDECL/STREAMWAIT/STREAMWRCFG decoded and guarded"
    )


if __name__ == "__main__":
    main()
