"""Identifying — and refusing to guess at — the host on the other end of the wire.

The dangerous half of :mod:`tt_sim.bridge.hostlink` is not the signalling, it is
the identification: this code sends ``SIGTERM``, so every path that cannot prove
which process is the wire peer must return ``False`` rather than pick one. Most
of what follows tests the refusals.
"""

import io
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

from tt_sim.bridge import hostlink

_LISTENER_SNIPPET = (
    "import socket, sys, time\n"
    "s = socket.socket()\n"
    "s.bind(('127.0.0.1', 0))\n"
    "s.listen(1)\n"
    "print(s.getsockname()[1], flush=True)\n"
    "time.sleep(120)\n"
)


@pytest.fixture
def listener_process():
    """A separate process holding a TCP listener, and the port it holds."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _LISTENER_SNIPPET],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        port = int(proc.stdout.readline().strip())
        yield proc, port
    finally:
        proc.kill()
        proc.wait()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_wire_port_reads_the_umd_address_shape():
    assert hostlink.wire_port("tcp://devvm:54812") == 54812
    assert hostlink.wire_port("tcp://127.0.0.1:50000") == 50000


@pytest.mark.parametrize(
    "addr", [None, "", "ipc:///tmp/tt-sim.sock", "tcp://devvm", "tcp://devvm:99999"]
)
def test_wire_port_declines_anything_it_cannot_read(addr):
    assert hostlink.wire_port(addr) is None


def test_finds_the_process_holding_the_wire_address(listener_process):
    proc, port = listener_process

    assert hostlink.find_wire_peer(f"tcp://127.0.0.1:{port}") == proc.pid


def test_no_peer_when_nobody_is_listening():
    assert hostlink.find_wire_peer(f"tcp://127.0.0.1:{_free_port()}") is None


def test_no_peer_for_an_address_with_no_port():
    assert hostlink.find_wire_peer("ipc:///tmp/tt-sim.sock") is None


def test_we_are_never_our_own_peer():
    """A listener held by this process must not be reported as the host."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]

        assert hostlink.find_wire_peer(f"tcp://127.0.0.1:{port}") is None


def test_stop_host_refuses_when_it_cannot_identify_a_peer():
    stream = io.StringIO()

    stopped = hostlink.stop_host(
        "a go-message that will never arrive",
        addr=f"tcp://127.0.0.1:{_free_port()}",
        env={},
        stream=stream,
    )

    assert stopped is False
    out = stream.getvalue()
    assert "could not identify the tt-metal host" in out
    assert "a go-message that will never arrive" in out
    assert "Ctrl-C" in out


def test_stop_host_honours_the_escape_hatch(listener_process):
    proc, port = listener_process
    stream = io.StringIO()

    stopped = hostlink.stop_host(
        "anything at all",
        addr=f"tcp://127.0.0.1:{port}",
        env={hostlink.DISABLE_ENV: "1"},
        stream=stream,
    )

    assert stopped is False
    assert hostlink.DISABLE_ENV in stream.getvalue()
    assert proc.poll() is None, "the host was killed despite the escape hatch"


def test_stop_host_ends_the_process_holding_the_wire(listener_process):
    proc, port = listener_process
    stream = io.StringIO()

    stopped = hostlink.stop_host(
        "a kernel launch on worker 2-1 to complete",
        addr=f"tcp://127.0.0.1:{port}",
        env={},
        stream=stream,
    )

    assert stopped is True
    assert proc.wait(timeout=10) is not None
    out = stream.getvalue()
    assert f"stopping the tt-metal host (pid {proc.pid})" in out
    assert "a kernel launch on worker 2-1 to complete" in out


def test_stop_host_falls_back_to_the_environment_for_the_address(listener_process):
    proc, port = listener_process
    stream = io.StringIO()

    stopped = hostlink.stop_host(
        "a reply",
        env={"NNG_SOCKET_ADDR": f"tcp://127.0.0.1:{port}"},
        stream=stream,
    )

    assert stopped is True
    assert proc.wait(timeout=10) is not None


def test_host_not_stranded_is_transparent_when_nothing_goes_wrong():
    calls = []

    with hostlink.host_not_stranded(
        "tcp://127.0.0.1:1", stopper=lambda *a, **k: calls.append(a)
    ):
        pass

    assert calls == []


def test_host_not_stranded_stops_the_host_and_re_raises():
    calls = []

    def stopper(reason, **kwargs):
        calls.append((reason, kwargs["addr"]))
        return True

    with (
        pytest.raises(ValueError, match="kaboom"),
        hostlink.host_not_stranded("tcp://a:1", stopper=stopper),
    ):
        raise ValueError("kaboom")

    assert len(calls) == 1
    assert "ValueError: kaboom" in calls[0][0]
    assert calls[0][1] == "tcp://a:1"


def test_host_not_stranded_leaves_a_keyboard_interrupt_alone():
    """Ctrl-C is somebody already in control of the run; don't shoot the host."""
    calls = []

    with (
        pytest.raises(KeyboardInterrupt),
        hostlink.host_not_stranded(
            "tcp://a:1", stopper=lambda *a, **k: calls.append(a)
        ),
    ):
        raise KeyboardInterrupt

    assert calls == []


def test_a_dead_peer_is_reported_stopped_without_escalation(listener_process):
    """SIGTERM is enough for a process that takes it; SIGKILL is only a backstop."""
    proc, port = listener_process
    started = time.monotonic()

    assert hostlink.stop_host(
        "a reply", addr=f"tcp://127.0.0.1:{port}", env={}, stream=io.StringIO()
    )

    assert time.monotonic() - started < hostlink._TERM_GRACE_S
    assert proc.wait(timeout=5) == -int(signal.SIGTERM)


def test_pid_lookup_survives_a_process_vanishing_mid_scan():
    """``/proc`` entries disappear under the scan constantly; that is not an error."""
    assert hostlink._pids_owning(set()) == set()
    assert hostlink._pids_owning({"999999999999"}) == set()
    assert (
        hostlink._listening_inodes(54812, proc_net=("/proc/does-not-exist",)) == set()
    )
    assert os.getpid() not in hostlink._pids_owning({"0"})


def test_stop_host_defaults_its_output_stream_to_stderr(capsys):
    hostlink.stop_host("a reply", addr="ipc://nope", env={})

    assert "could not identify the tt-metal host" in capsys.readouterr().err
