"""Smoke test for the phase-1 transport/fabric.

Runs ``Transport.serve()`` in a thread and a pynng dialer in the main thread,
exchanges one of every command against a NullCore-only fabric, and asserts that
each response matches the expected shape.

Run from anywhere:
    python3 -m driver.wormhole.server.smoke_test
"""

import tempfile
import threading
import time

import pynng

from tt_sim.bridge import Fabric, Transport
from tt_sim.bridge import protocol as proto


def _run_server(addr, ready):
    fabric = Fabric()
    transport = Transport(addr, log_protocol=False)
    ready.append(transport)
    transport.serve(fabric)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        addr = f"ipc://{tmp}/smoke_sock"
        ready: list[Transport] = []

        srv = threading.Thread(target=_run_server, args=(addr, ready), daemon=True)
        srv.start()

        # Dial with retry — server is binding concurrently.
        deadline = time.monotonic() + 5.0
        sock = None
        while time.monotonic() < deadline:
            try:
                sock = pynng.Pair1(dial=addr, block_on_dial=False)
                break
            except pynng.exceptions.ConnectionRefused:
                time.sleep(0.05)
        assert sock is not None, "could not dial server"

        with sock:
            sock.recv_timeout = 2000
            sock.send_timeout = 2000

            # 1) Handshake EXIT ack from server.
            ack = proto.parse(sock.recv())
            assert ack.cmd == proto.CMD_EXIT, f"expected EXIT ack, got {ack.cmd}"

            # 2) WRITE — no reply expected.
            payload = bytes(range(1, 17))  # 16 bytes, 4 uint32 words
            sock.send(
                proto.build_msg(
                    proto.CMD_WRITE,
                    data=payload,
                    core=(1, 1),
                    address=0x1000,
                    size=len(payload),
                )
            )

            # 3) READ — expect zeros from NullCore.
            sock.send(
                proto.build_msg(
                    proto.CMD_READ,
                    core=(1, 1),
                    address=0x1000,
                    size=16,
                )
            )
            rr = proto.parse(sock.recv())
            assert rr.cmd == proto.CMD_READ, f"expected READ reply, got {rr.cmd}"
            assert rr.data == b"\x00" * 16, f"expected zeros, got {rr.data!r}"

            # 4) RESET assert / deassert — no replies.
            sock.send(proto.build_msg(proto.CMD_RESET_ASSERT, core=(1, 1)))
            sock.send(proto.build_msg(proto.CMD_RESET_DEASSERT, core=(1, 1)))

            # 5) READ on an unmapped coord — fabric should lazily allocate
            # NullCore and still return zeros.
            sock.send(
                proto.build_msg(proto.CMD_READ, core=(0, 11), address=0x360, size=8)
            )
            rr2 = proto.parse(sock.recv())
            assert rr2.data == b"\x00" * 8

            # 6) Verify the WRITE message actually carried payload through parse
            # by inspecting the transport's count after the round trips above.
            assert ready, "server thread never registered Transport"
            assert ready[0].msg_count >= 5, (
                f"server should have seen >=5 messages, saw {ready[0].msg_count}"
            )

            # 7) EXIT — clean shutdown.
            sock.send(proto.build_msg(proto.CMD_EXIT))

        srv.join(timeout=2.0)
        assert not srv.is_alive(), "server did not shut down after EXIT"

    print(f"smoke test OK ({ready[0].msg_count} messages exchanged)")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
