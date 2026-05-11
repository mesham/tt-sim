"""Smoke test for TensixCore + DramCore + cycle pumping.

Spins up a server with a real ``Wormhole`` behind translated coords (1,1) and
(0,11), and drives a small WRITE/READ conversation over the wire. Verifies:

* writes to L1 land at the expected unified coord (read-back matches);
* writes to DRAM land at the expected unified coord;
* reset assert/deassert do not crash and do trigger ``wormhole.run``;
* lazy NullCore allocation still works for unmapped coords.

Run from anywhere:
    python3 -m driver.wormhole.server.tensix_smoke_test

This test does NOT load tt-metal firmware — it only exercises the adapter
plumbing. Full BRISC execution against firmware happens in the end-to-end
``TT_METAL_SIMULATOR=...`` flow.
"""

import tempfile
import threading
import time

import pynng

from . import protocol as proto
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP
from .cores import DramCore, TensixCore
from .device import Device
from .fabric import Fabric
from .transport import Transport

# tt-sim Wormhole's tensix is at unified (18,18); DRAM at (16,16). The
# translated wire coords come from coords.py.
WIRE_TENSIX = next(iter(TENSIX_COORD_MAP))
WIRE_DRAM = next(iter(DRAM_COORD_MAP))


def _dial(addr, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return pynng.Pair1(dial=addr, block_on_dial=False)
        except pynng.exceptions.ConnectionRefused:
            time.sleep(0.05)
    raise RuntimeError(f"could not dial {addr}")


class _CountingWormhole:
    """Test wrapper: counts pump invocations on the real Wormhole."""

    def __init__(self, inner):
        self._inner = inner
        self.run_calls = 0
        self.reset_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def run(self, n):
        self.run_calls += 1
        return self._inner.run(n)

    def reset(self, *args, **kwargs):
        self.reset_calls += 1
        return self._inner.reset(*args, **kwargs)


def _build_server(addr, cycles_per_poll):
    device = Device(cycles_per_poll=cycles_per_poll)
    # Wrap wormhole so we can count pumps.
    device.wormhole = _CountingWormhole(device.wormhole)
    fabric = Fabric()
    for t, u in TENSIX_COORD_MAP.items():
        fabric.register(t, TensixCore(device, u))
    for t, u in DRAM_COORD_MAP.items():
        fabric.register(t, DramCore(device, u))
    return fabric, device


def main():
    with tempfile.TemporaryDirectory() as tmp:
        addr = f"ipc://{tmp}/tensix_smoke.sock"
        fabric, device = _build_server(addr, cycles_per_poll=10)
        transport = Transport(addr, log_protocol=False)
        srv = threading.Thread(target=transport.serve, args=(fabric,), daemon=True)
        srv.start()

        with _dial(addr) as sock:
            sock.recv_timeout = 5000
            sock.send_timeout = 5000

            ack = proto.parse(sock.recv())
            assert ack.cmd == proto.CMD_EXIT, "expected EXIT ack"

            # 1) WRITE to Tensix L1, READ back, verify.
            payload = bytes.fromhex("deadbeef" "cafebabe" "01020304" "05060708")
            sock.send(
                proto.build_msg(
                    proto.CMD_WRITE,
                    data=payload,
                    core=WIRE_TENSIX,
                    address=0x2010,
                    size=len(payload),
                )
            )
            sock.send(
                proto.build_msg(
                    proto.CMD_READ,
                    core=WIRE_TENSIX,
                    address=0x2010,
                    size=len(payload),
                )
            )
            rr = proto.parse(sock.recv())
            assert rr.data[: len(payload)] == payload, (
                f"L1 round-trip mismatch: wrote {payload.hex()}, "
                f"read {rr.data[: len(payload)].hex()}"
            )

            # No pumps yet — nothing has been deasserted.
            assert (
                device.wormhole.run_calls == 0
            ), f"expected 0 pumps before deassert, got {device.wormhole.run_calls}"

            # 2) WRITE to DRAM, READ back, verify.
            dram_payload = bytes.fromhex("11223344" * 4)
            sock.send(
                proto.build_msg(
                    proto.CMD_WRITE,
                    data=dram_payload,
                    core=WIRE_DRAM,
                    address=0x100,
                    size=len(dram_payload),
                )
            )
            sock.send(
                proto.build_msg(
                    proto.CMD_READ,
                    core=WIRE_DRAM,
                    address=0x100,
                    size=len(dram_payload),
                )
            )
            rr2 = proto.parse(sock.recv())
            assert rr2.data[: len(dram_payload)] == dram_payload, (
                f"DRAM round-trip mismatch: wrote {dram_payload.hex()}, "
                f"read {rr2.data[: len(dram_payload)].hex()}"
            )

            # 3) RESET_ASSERT then DEASSERT — pump should fire after deassert
            # and on every subsequent message. Seed L1[0x0] with `jal x0, 0`
            # (infinite self-jump, 0x0000006F) so BRISC spins harmlessly on
            # the first fetch instead of trying to execute zeros.
            sock.send(
                proto.build_msg(
                    proto.CMD_WRITE,
                    data=(0x0000006F).to_bytes(4, "little"),
                    core=WIRE_TENSIX,
                    address=0x0,
                    size=4,
                )
            )
            pumps_before = device.wormhole.run_calls
            sock.send(proto.build_msg(proto.CMD_RESET_ASSERT, core=WIRE_TENSIX))
            sock.send(proto.build_msg(proto.CMD_RESET_DEASSERT, core=WIRE_TENSIX))
            sock.send(
                proto.build_msg(
                    proto.CMD_READ, core=WIRE_TENSIX, address=0x2010, size=4
                )
            )
            proto.parse(sock.recv())
            pumps_after = device.wormhole.run_calls
            # 1 pump on deassert + 1 pump on the subsequent READ = 2.
            assert (
                pumps_after - pumps_before >= 2
            ), f"expected >=2 pumps after deassert+read, got {pumps_after - pumps_before}"

            # 4) Unmapped coord still works via lazy NullCore allocation.
            sock.send(proto.build_msg(proto.CMD_READ, core=(2, 2), address=0x0, size=4))
            null_rr = proto.parse(sock.recv())
            assert (
                null_rr.data[:4] == b"\x00\x00\x00\x00"
            ), f"unmapped coord should return zeros, got {null_rr.data.hex()}"

            sock.send(proto.build_msg(proto.CMD_EXIT))

        srv.join(timeout=5.0)
        assert not srv.is_alive(), "server did not shut down"

    print(
        f"tensix smoke test OK "
        f"(messages={transport.msg_count}, pumps={device.wormhole.run_calls}, "
        f"resets={device.wormhole.reset_calls})"
    )


if __name__ == "__main__":
    raise SystemExit(main() or 0)
