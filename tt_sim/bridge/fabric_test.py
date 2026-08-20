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


def _guarded(host_stopper=None, lazy=False):
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
        lazy=lazy,
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


def test_lazy_materialisation_retires_the_unmaterialised_worker_error(capsys):
    """With workers built on demand there is no such thing as a worker the
    program has outgrown, so the guard has nothing to say about one."""
    calls = []
    fabric = _guarded(host_stopper=lambda *a, **k: calls.append(a) or False, lazy=True)

    fabric.kernel_launch_callback((3, 1))

    assert capsys.readouterr().err == ""
    assert calls == []


def test_a_kernel_launch_on_a_non_worker_is_reported_but_not_fatal(capsys):
    """The safety net that survives in both modes.

    Nothing can materialise a coord that is not a functional worker, so the
    launch is announced — but it is *not* worth stopping the run over: a
    NullCore reads back ``RUN_MSG_DONE`` immediately so the host cannot hang on
    it, and the detection is a 4-byte-write heuristic that a false positive
    would turn into a false kill.
    """
    calls = []
    fabric = _guarded(host_stopper=lambda *a, **k: calls.append(a) or False)

    fabric.kernel_launch_callback((0, 11))
    fabric.kernel_launch_callback((0, 11))

    err = capsys.readouterr().err
    assert err.count("ERROR") == 1  # once per coord, not once per write
    assert "0-11" in err
    assert "not a functional worker" in err
    assert calls == []


def test_the_dprint_init_handshake_completes_through_the_fabric():
    """``WriteInitMagic``, end to end on the wire path, for an unbuilt worker.

    The unit-level version of this lives in ``cores_test``; this one pins the
    route, because the failure it guards against ("DPRINT will not start
    against tt-sim") was only ever visible as a host that spun 100000 times on
    a fabric read and then threw. ``(2, 1)`` is deliberately *not* in ``POOL``,
    so the fabric lazily allocates a stand-in for it exactly as a live server
    does for a worker nothing has claimed yet.
    """
    fabric = _guarded()
    starting_magic = bytes([0x98, 0x98, 0x98, 0x98])
    # ``mailboxes_t.dprint_buf``: the host writes the whole struct, reads word 0.
    fabric.write((2, 1), 0x1B00, starting_magic + bytes(1020 - 4))

    assert bytes(fabric.read((2, 1), 0x1B00, 4)) == starting_magic


def test_an_unmaterialised_workers_go_message_still_reads_done():
    """The exception the DPRINT fix must not break: no firmware, no run-state.

    ``wait_until_cores_done`` polls this after writing ``go=INIT`` to every
    declared worker. Echo it back and device init never finishes.
    """
    fabric = _guarded()
    fabric.write((2, 1), 0x4A0, b"\x00\x00\x00\x40")

    assert bytes(fabric.read((2, 1), 0x4A0, 4))[3] == 0x00
