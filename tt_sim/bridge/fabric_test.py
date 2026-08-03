"""The fabric's "that worker isn't materialised" guards.

These used to be inline in ``driver/wormhole/server/__main__.py``, so a
Blackhole run silently NullCore-swallowed exactly the traffic a Wormhole run
shouted about — the same one-architecture-only pattern that left Blackhole
without tracing and without the deadlock watchdog. They live here now and both
server entry points install them.
"""

import pytest

from tt_sim.bridge.fabric import Fabric, install_worker_guards

# Physical coord -> tile coord, as a server's TENSIX_COORD_MAP would be.
COORD_MAP = {(1, 1): (18, 18), (2, 1): (19, 18), (3, 1): (20, 18)}
POOL = [(1, 1)]


def _guarded():
    fabric = Fabric()
    install_worker_guards(fabric, POOL, COORD_MAP)
    return fabric


def test_warns_about_an_unmaterialised_worker(capsys):
    _guarded().unmapped_callback((2, 1))

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "2-1" in err
    assert "TT_SIM_TENSIX_COORDS" in err


def test_quiet_about_materialised_workers_and_non_workers(capsys):
    fabric = _guarded()
    fabric.unmapped_callback((1, 1))  # in the pool
    fabric.unmapped_callback((0, 11))  # not a worker at all (DRAM/eth/arc)

    assert capsys.readouterr().err == ""


def test_a_kernel_launch_on_an_unmaterialised_worker_is_fatal(capsys):
    fabric = _guarded()

    with pytest.raises(SystemExit):
        # ``os._exit`` is what the real guard calls (UMD has no "simulator
        # died" path); pytest's capture would lose the message, so assert on
        # the message and the exit separately.
        import os

        real_exit = os._exit
        os._exit = lambda code: (_ for _ in ()).throw(SystemExit(code))
        try:
            fabric.kernel_launch_callback((3, 1))
        finally:
            os._exit = real_exit

    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "go=GO" in err
    assert "3-1" in err


def test_a_kernel_launch_on_a_materialised_worker_is_fine(capsys):
    _guarded().kernel_launch_callback((1, 1))

    assert capsys.readouterr().err == ""
