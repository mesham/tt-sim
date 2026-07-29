"""Blackhole bring-up regression tests (no tt-metal / no socket required).

Two checks:
1. ``bringup.main()`` — the standalone DRAM->L1 NoC read with Blackhole's real
   HI-register address encoding.
2. The wire path: build the fabric + ``Device`` exactly as ``server/__main__``
   does, then route WRITE/READ messages through ``Transport._handle`` (the same
   dispatch UMD drives over the socket) into the Blackhole device — confirming
   the shared bridge drives Blackhole end-to-end.

Run:  python3 -m driver.blackhole.tests.bringup_test  (or via pytest)
"""

from types import SimpleNamespace

from driver.blackhole.bringup import main as bringup_main
from driver.blackhole.server.bh_device import make_device
from driver.blackhole.server.coords import DRAM_COORD_MAP, TENSIX_COORD_MAP
from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto


def _msg(cmd, core, addr, size=0, data=b""):
    return SimpleNamespace(cmd=cmd, core=core, address=addr, size=size, data=data)


def test_bringup_noc_read():
    # Raises (asserts) internally on mismatch; a clean return is the pass.
    bringup_main()


def test_bridge_routes_write_read_into_blackhole():
    device = make_device()
    fabric = Fabric()
    for translated, tile in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, tile))
    device.ensure_tensix_tile((1, 2))
    fabric.register((1, 2), TensixCore(device, TENSIX_COORD_MAP[(1, 2)]))

    transport = Transport(addr=None)  # never connects; only _handle is used
    payload = bytes(range(16))

    # DRAM round-trip
    transport._handle(fabric, _msg(proto.CMD_WRITE, (0, 11), 0x100, data=payload))
    dram = transport._handle(fabric, _msg(proto.CMD_READ, (0, 11), 0x100, 16))
    assert bytes(dram) == payload

    # Tensix L1 round-trip
    transport._handle(fabric, _msg(proto.CMD_WRITE, (1, 2), 0x200, data=payload[:8]))
    l1 = transport._handle(fabric, _msg(proto.CMD_READ, (1, 2), 0x200, 8))
    assert bytes(l1) == payload[:8]

    # A second worker materialises on demand.
    unified = device.ensure_tensix_tile((2, 2))
    assert unified == (2, 2)
    assert (2, 2) in device.tt_device.tile_directory

    device.tt_device.shutdown()


def main():
    test_bringup_noc_read()
    test_bridge_routes_write_read_into_blackhole()
    print("blackhole bringup_test OK: NoC read + bridge WRITE/READ into Blackhole")


if __name__ == "__main__":
    main()
