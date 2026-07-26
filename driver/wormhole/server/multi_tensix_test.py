"""Multi-Tensix plumbing smoke test (two materialized tiles).

Ports the coverage of the old standalone ``seven`` example onto the server
Device/Fabric plumbing, without the deleted standalone launch helpers
(``wormhole_driver`` / ``tt_metal``). It materializes two Tensix tiles —
physical ``(1, 1)`` and ``(2, 1)``, unified ``(18, 18)`` and ``(19, 18)`` —
and checks that:

* each tile's L1 is independent (a write to one tile is not visible in the
  other, and read-back matches per tile);
* a reset assert/deassert scoped to one tile pumps the clock without
  disturbing the sibling tile's L1 (``Device`` uses ``reset_tile`` rather than
  the global ``reset``);
* the shared DRAM tile round-trips.

The full "real firmware + kernel on two tiles" path that the old ``seven``
exercised is now covered end-to-end by the ``nine`` example in the
``TT_METAL_SIMULATOR`` flow (a CB bridged across two tiles over the NoC).

Run from anywhere:
    python3 -m driver.wormhole.server.multi_tensix_test
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

# Two adjacent worker tiles. Physical (1,1)->unified (18,18), (2,1)->(19,18).
WIRE_TILE_A = (1, 1)
WIRE_TILE_B = (2, 1)
WIRE_DRAM = next(iter(DRAM_COORD_MAP))


def _dial(addr, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return pynng.Pair1(dial=addr, block_on_dial=False)
        except pynng.exceptions.ConnectionRefused:
            time.sleep(0.05)
    raise RuntimeError(f"could not dial {addr}")


def _build_server(cycles_per_poll):
    device = Device(cycles_per_poll=cycles_per_poll)
    fabric = Fabric()
    for translated, unified in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, unified))
    for translated in (WIRE_TILE_A, WIRE_TILE_B):
        unified = TENSIX_COORD_MAP[translated]
        device.ensure_tensix_tile(translated)
        fabric.register(translated, TensixCore(device, unified))
    return fabric, device


def _write(sock, core, address, data):
    sock.send(
        proto.build_msg(
            proto.CMD_WRITE, data=data, core=core, address=address, size=len(data)
        )
    )


def _read(sock, core, address, size):
    sock.send(proto.build_msg(proto.CMD_READ, core=core, address=address, size=size))
    return proto.parse(sock.recv()).data[:size]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        addr = f"ipc://{tmp}/multi_tensix.sock"
        fabric, _ = _build_server(cycles_per_poll=10)
        transport = Transport(addr, log_protocol=False)
        srv = threading.Thread(target=transport.serve, args=(fabric,), daemon=True)
        srv.start()

        with _dial(addr) as sock:
            sock.recv_timeout = 5000
            sock.send_timeout = 5000

            ack = proto.parse(sock.recv())
            assert ack.cmd == proto.CMD_EXIT, "expected EXIT ack"

            # 1) Per-tile L1 independence: distinct payloads to the same L1
            # address on each tile must not bleed across tiles.
            payload_a = bytes.fromhex("aaaaaaaa11111111")
            payload_b = bytes.fromhex("bbbbbbbb22222222")
            _write(sock, WIRE_TILE_A, 0x2010, payload_a)
            _write(sock, WIRE_TILE_B, 0x2010, payload_b)
            back_a = _read(sock, WIRE_TILE_A, 0x2010, len(payload_a))
            back_b = _read(sock, WIRE_TILE_B, 0x2010, len(payload_b))
            assert back_a == payload_a, (
                f"tile A L1 mismatch: {back_a.hex()} != {payload_a.hex()}"
            )
            assert back_b == payload_b, (
                f"tile B L1 mismatch: {back_b.hex()} != {payload_b.hex()}"
            )
            assert back_a != back_b, "tile A and B L1 are not independent"

            # 2) Shared DRAM round-trip.
            dram_payload = bytes.fromhex("deadbeef" * 4)
            _write(sock, WIRE_DRAM, 0x200, dram_payload)
            back_dram = _read(sock, WIRE_DRAM, 0x200, len(dram_payload))
            assert back_dram == dram_payload, (
                f"DRAM mismatch: {back_dram.hex()} != {dram_payload.hex()}"
            )

            # 3) Reset one tile without disturbing the sibling. Seed both L1[0]
            # with `jal x0, 0` (0x0000006F) so a deasserted BRISC spins
            # harmlessly instead of executing zeros. Resetting tile A must
            # leave tile B's L1 payload intact.
            jal = (0x0000006F).to_bytes(4, "little")
            _write(sock, WIRE_TILE_A, 0x0, jal)
            _write(sock, WIRE_TILE_B, 0x0, jal)
            sock.send(proto.build_msg(proto.CMD_RESET_ASSERT, core=WIRE_TILE_A))
            sock.send(proto.build_msg(proto.CMD_RESET_DEASSERT, core=WIRE_TILE_A))
            # Give the deasserted core a few pumps' worth of messages.
            still_b = _read(sock, WIRE_TILE_B, 0x2010, len(payload_b))
            assert still_b == payload_b, (
                f"tile B L1 was disturbed by tile A reset: {still_b.hex()} != {payload_b.hex()}"
            )

            sock.send(proto.build_msg(proto.CMD_EXIT))

        srv.join(timeout=5.0)
        assert not srv.is_alive(), "server did not shut down"

    print(f"multi-tensix smoke test OK (messages={transport.msg_count})")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
