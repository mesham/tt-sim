"""The hang, reproduced: a host waiting on a tile that was never instantiated.

This is a three-process test with the real topology UMD sets up. A **fake
host** binds a TCP listener, exports its address as ``NNG_SOCKET_ADDR``, spawns
a **fake simulator** child, and then blocks waiting for that child exactly as
UMD blocks in ``recv_from_device`` — with no timeout, because UMD sets none.
The child runs the *real* :func:`install_worker_guards` over a real
:class:`Fabric` and pushes real wire traffic through it. No tt-sim device is
built, so the whole thing is sub-second; nothing about the trigger or the
consequence is stubbed.

``ghost``
    The host launches a kernel on a worker that is not in the pool. On the
    unfixed tree the guard printed its error and called ``os._exit(1)``, and
    the fake host then sat in ``accept()`` until its own 25-second patience ran
    out — which is the reported bug: *the diagnostic was already there and the
    run hung anyway*. The test asserts the host dies by ``SIGTERM``, so it
    fails on the unfixed tree (the host exits 3 with ``HOST-STILL-WAITING``).

``correct``
    A legitimate, deliberately slow run: a launch on a materialised worker, a
    grid-wide ``go=INIT`` and user data to an unmaterialised one, and two
    seconds of polling before it completes. Nothing may be stopped, and the
    pre-existing swallowed-write WARNING must still fire — the guard is not
    allowed to get quieter in exchange for getting more decisive.

``crash``
    The other route to the same infinite wait: the simulator dies of an
    exception mid-conversation. ``host_not_stranded`` must take the host with
    it rather than leave it blocked behind a traceback.
"""

import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Physical coord -> tile coord, as a server's TENSIX_COORD_MAP would be.
COORD_MAP = {(1, 1): (18, 18), (2, 1): (19, 18)}
POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
GO_GO = bytes([0, 0, 0, 0x80])  # RUN_MSG_GO   — a kernel launch
GO_INIT = bytes([0, 0, 0, 0x40])  # RUN_MSG_INIT — the grid-wide handshake

# How long the fake host waits before declaring itself hung. Long enough that a
# working stop (milliseconds) is unambiguous, short enough to fail a run.
HOST_WAIT_S = 25.0
# How long the "correct" program deliberately takes.
SLOW_S = 2.0


# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------


def _run_host(mode):
    env = {k: v for k, v in os.environ.items() if k != "TT_SIM_NO_HOST_STOP"}
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "host", mode],
        capture_output=True,
        text=True,
        timeout=HOST_WAIT_S + 40,
        env=env,
    )
    return proc, time.monotonic() - started


def test_a_launch_on_an_uninstantiated_tile_stops_the_host_instead_of_hanging():
    proc, elapsed = _run_host("ghost")

    assert "HOST-STILL-WAITING" not in proc.stdout, (
        "the host waited out its full patience — this is the hang itself"
    )
    assert proc.returncode == -signal.SIGTERM, (
        f"host exited {proc.returncode}, stdout={proc.stdout!r} stderr={proc.stderr[-2000:]!r}"
    )
    # The diagnostic names the tile, the env var, and what is being stopped.
    assert "ERROR" in proc.stderr
    assert "go=GO" in proc.stderr
    assert "2-1" in proc.stderr
    assert "TT_SIM_TENSIX_COORDS (currently: 1-1)" in proc.stderr
    assert "stopping the tt-metal host" in proc.stderr
    assert elapsed < HOST_WAIT_S, "stopped only because the host gave up on its own"


def test_a_slow_but_correct_program_is_left_alone():
    proc, elapsed = _run_host("correct")

    assert proc.returncode == 0, f"stderr={proc.stderr[-2000:]!r}"
    assert "HOST-COMPLETED" in proc.stdout
    assert "stopping the tt-metal host" not in proc.stderr
    assert "ERROR" not in proc.stderr
    # ...and the older, quieter guard still speaks.
    assert "WARNING" in proc.stderr
    assert "2-1" in proc.stderr
    assert elapsed >= SLOW_S, "the 'slow' program did not actually take any time"


def test_a_simulator_crash_stops_the_host_instead_of_hanging():
    proc, elapsed = _run_host("crash")

    assert "HOST-STILL-WAITING" not in proc.stdout
    assert proc.returncode == -signal.SIGTERM, (
        f"host exited {proc.returncode}, stderr={proc.stderr[-2000:]!r}"
    )
    assert "stopping the tt-metal host" in proc.stderr
    assert "BoomError" in proc.stderr
    assert elapsed < HOST_WAIT_S


# --------------------------------------------------------------------------
# The two child roles, re-entered via ``python host_stop_test.py <role> <mode>``
# --------------------------------------------------------------------------


class BoomError(RuntimeError):
    """Stands in for any simulator-side exception (e.g. NoCAlignmentError)."""


def _host_main(mode):
    """Bind the wire address, spawn the simulator, then wait like UMD does."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    env = dict(os.environ)
    env["NNG_SOCKET_ADDR"] = f"tcp://127.0.0.1:{port}"
    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "sim", mode, str(port)], env=env
    )

    listener.settimeout(HOST_WAIT_S)
    try:
        listener.accept()
    except TimeoutError:
        print("HOST-STILL-WAITING", flush=True)
        child.kill()
        child.wait()
        return 3
    print("HOST-COMPLETED", flush=True)
    child.wait()
    return 0


def _sim_main(mode, port):
    from tt_sim.bridge.fabric import Fabric, install_worker_guards
    from tt_sim.bridge.hostlink import host_not_stranded

    fabric = Fabric()
    install_worker_guards(fabric, POOL, COORD_MAP)

    if mode == "ghost":
        # The mistyped coordinate: the host launches on a worker tt-sim has no
        # tile for. If the guard fails to stop the host it must NOT exit here,
        # because exiting is what stranded the host in the first place.
        fabric.write((2, 1), GO_MSG_ADDR, GO_GO)
        time.sleep(HOST_WAIT_S + 10)
        return 1

    if mode == "crash":
        with host_not_stranded(os.environ["NNG_SOCKET_ADDR"]):
            raise BoomError("the simulator fell over mid-conversation")

    # mode == "correct": a real launch on a real worker, plus the traffic that
    # legitimately reaches unmaterialised ones, plus a slow kernel.
    fabric.write((2, 1), 0x20000, b"\x11" * 16)  # user data -> WARNING, not fatal
    fabric.write((2, 1), GO_MSG_ADDR, GO_INIT)  # grid-wide init, not a launch
    fabric.write((1, 1), GO_MSG_ADDR, GO_GO)  # launch on a materialised worker
    deadline = time.monotonic() + SLOW_S
    while time.monotonic() < deadline:
        fabric.read((1, 1), GO_MSG_ADDR, 4)
        time.sleep(0.005)
    with socket.create_connection(("127.0.0.1", port), timeout=5):
        pass  # normal completion — release the host's wait
    return 0


if __name__ == "__main__":  # pragma: no cover - re-entry for the child roles
    role = sys.argv[1]
    if role == "host":
        sys.exit(_host_main(sys.argv[2]))
    sys.exit(_sim_main(sys.argv[2], int(sys.argv[3])))
