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


def _guarded(host_stopper=None):
    """A guarded fabric whose host-stopper never finds a host, unless told to.

    The default stands in for "no tt-metal host on the other end" — which is
    the truth under pytest — so the launch guard takes its keep-serving branch
    and the test process survives to make assertions.
    """
    fabric = Fabric()
    install_worker_guards(
        fabric,
        POOL,
        COORD_MAP,
        host_stopper=host_stopper or (lambda *a, **k: False),
    )
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


def test_a_kernel_launch_on_an_unmaterialised_worker_is_an_error(capsys):
    _guarded().kernel_launch_callback((3, 1))

    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "go=GO" in err
    assert "3-1" in err
    # The message has to connect the swallowed traffic to the host's wait, or
    # the reader is still left wondering why nothing is happening.
    assert "waiting" in err
    assert "TT_SIM_TENSIX_COORDS (currently: 1-1)" in err


def test_a_kernel_launch_on_an_unmaterialised_worker_asks_for_the_host_to_be_stopped(
    capsys,
):
    calls = []
    fabric = _guarded(
        host_stopper=lambda reason, **kw: calls.append((reason, kw)) or False
    )

    fabric.kernel_launch_callback((3, 1))

    assert len(calls) == 1
    assert "3-1" in calls[0][0]


def test_the_launch_guard_only_exits_once_the_host_is_stopped(capsys):
    """Exiting without taking the host with us is what made this a hang."""
    fabric = _guarded(host_stopper=lambda *a, **k: True)

    import os

    real_exit = os._exit
    os._exit = lambda code: (_ for _ in ()).throw(SystemExit(code))
    try:
        with pytest.raises(SystemExit):
            fabric.kernel_launch_callback((3, 1))
    finally:
        os._exit = real_exit

    assert "ERROR" in capsys.readouterr().err


def test_a_kernel_launch_on_a_materialised_worker_is_fine(capsys):
    calls = []
    _guarded(
        host_stopper=lambda *a, **k: calls.append(a) or False
    ).kernel_launch_callback((1, 1))

    assert capsys.readouterr().err == ""
    assert calls == []
