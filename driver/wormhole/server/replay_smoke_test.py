"""End-to-end test for the record/replay pipeline.

Phase A: spin up a server with TraceWriter active, drive a small mix of
WRITE/READ/RESET/EXIT messages from a dialer in the main thread, let the
server shut down. Phase B: spin up a fresh server, run ``replay.py`` against
the recorded trace, expect every READ reply to match.

Run from anywhere:
    python3 -m driver.wormhole.server.replay_smoke_test
"""

import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pynng

from tt_sim.bridge import Fabric, TraceWriter, Transport
from tt_sim.bridge import protocol as proto


def _run_server(addr, fabric, tracer, ready):
    transport = Transport(addr, log_protocol=False, trace_writer=tracer)
    ready.append(transport)
    transport.serve(fabric)


def _listen(addr):
    """Bind the host side: UMD (or a stand-in like this test) is the LISTENER;
    the simulator server is the DIALER and retries until a listener appears."""
    return pynng.Pair1(listen=addr)


# A canned conversation that exercises every command and a few coords.
def _drive(sock):
    sock.recv_timeout = 2000
    sock.send_timeout = 2000

    ack = proto.parse(sock.recv())
    assert ack.cmd == proto.CMD_EXIT, f"expected EXIT ack, got {ack.cmd}"

    payload = bytes(range(1, 17))
    sock.send(
        proto.build_msg(
            proto.CMD_WRITE,
            data=payload,
            core=(1, 1),
            address=0x1000,
            size=len(payload),
        )
    )
    sock.send(proto.build_msg(proto.CMD_READ, core=(1, 1), address=0x1000, size=16))
    rr = proto.parse(sock.recv())
    assert rr.data == b"\x00" * 16

    sock.send(proto.build_msg(proto.CMD_RESET_ASSERT, core=(1, 1)))
    sock.send(proto.build_msg(proto.CMD_RESET_DEASSERT, core=(1, 1)))

    sock.send(proto.build_msg(proto.CMD_READ, core=(0, 11), address=0x360, size=8))
    rr2 = proto.parse(sock.recv())
    assert rr2.data == b"\x00" * 8

    sock.send(proto.build_msg(proto.CMD_EXIT))


def _record_phase(addr, trace_path):
    fabric = Fabric()
    ready: list[Transport] = []
    with TraceWriter(trace_path) as tracer:
        srv = threading.Thread(
            target=_run_server, args=(addr, fabric, tracer, ready), daemon=True
        )
        srv.start()
        with _listen(addr) as sock:
            _drive(sock)
        srv.join(timeout=2.0)
        assert not srv.is_alive(), "record-phase server did not shut down"
    assert ready[0].msg_count >= 6
    return ready[0].msg_count


def _replay_phase(addr, trace_path):
    fabric = Fabric()
    ready: list[Transport] = []
    srv = threading.Thread(
        target=_run_server, args=(addr, fabric, None, ready), daemon=True
    )
    srv.start()

    # Run replay.py as the user would — as a subprocess against the new server.
    repo_root = Path(__file__).resolve().parents[3]
    replay_py = repo_root / "driver" / "wormhole" / "replay.py"
    env = {"NNG_SOCKET_ADDR": addr, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [sys.executable, str(replay_py), str(trace_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    srv.join(timeout=2.0)
    assert not srv.is_alive(), "replay-phase server did not shut down"

    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise AssertionError(f"replay.py exited {result.returncode}")
    return result.stdout


def main():
    with tempfile.TemporaryDirectory() as tmp:
        addr_a = f"ipc://{tmp}/record.sock"
        addr_b = f"ipc://{tmp}/replay.sock"
        trace = Path(tmp) / "phase1.trace"

        n_recorded = _record_phase(addr_a, trace)
        n_lines = sum(1 for line in trace.open() if line.strip())
        assert n_lines == n_recorded, (
            f"trace has {n_lines} lines but server saw {n_recorded} messages"
        )

        stdout = _replay_phase(addr_b, trace)
        if "0 READ mismatches" not in stdout:
            raise AssertionError(f"unexpected replay output:\n{stdout}")

    print(f"replay smoke test OK (trace: {n_recorded} messages, all replies matched)")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
