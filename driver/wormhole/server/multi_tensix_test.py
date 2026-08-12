"""Multi-Tensix plumbing, socket-free: two live tiles, and the whole grid.

Two halves, both driving the ``Fabric`` + ``Device`` wrapper directly (the
``offline_replay_test`` shape) rather than over nng — no socket, no server
thread, so it is fast and deterministic enough to sit in the ordinary suite.
The previous version of this file dialled a socket that nothing was listening
on (``Transport.serve`` dials too, so both ends were dialers) and had therefore
been failing whenever anyone ran it by hand; it also defined no ``test_``
function, so pytest collected the module and ran nothing.

1. **Two materialised tiles behave as two tiles** — independent L1, a shared
   DRAM, and a reset scoped to one of them. ``Device.deassert_reset`` uses
   ``reset_tile`` rather than the global ``reset`` precisely because the latter
   clobbers every other tile's PCs once more than one worker exists.

2. **The whole 80-worker grid stays addressable.** Every worker must resolve to
   itself on NoC 0 by its canonical coord and on NoC 1 by its mirror — the two
   forms real tt-metal kernels emit (``NOC_X``/``NOC_Y`` mirror on NoC 1). This
   is the invariant bug 4 broke: NoC 1's directory is ambiguous by construction,
   mirrors win over canonicals, and the collisions only appear once enough of
   the grid is built. Checking it at the full grid costs about a second and is
   the cheapest coverage there is of the regime a real workload runs in.

The "real firmware + kernel on two tiles" path is covered end to end by the
``nine`` example in the ``TT_METAL_SIMULATOR`` flow, and by the
``noc_tile_transfer`` replay guards on both architectures.

Run from anywhere:
    python3 -m pytest driver/wormhole/server/multi_tensix_test.py
"""

from tt_sim.bridge import DramCore, Fabric, TensixCore

from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

# Two adjacent worker tiles. Physical (1,1)->unified (18,18), (2,1)->(19,18).
WIRE_TILE_A = (1, 1)
WIRE_TILE_B = (2, 1)
WIRE_DRAM = next(iter(DRAM_COORD_MAP))

# `jal x0, 0` — a one-instruction spin, so a deasserted BRISC parks harmlessly
# instead of executing whatever zeros are at L1[0].
JAL_SELF = (0x0000006F).to_bytes(4, "little")


def _build_fabric(tiles=(WIRE_TILE_A, WIRE_TILE_B), cycles_per_poll=10):
    device = make_device(cycles_per_poll=cycles_per_poll)
    fabric = Fabric()
    for translated, unified in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, unified))
    for translated in tiles:
        device.ensure_tensix_tile(translated)
        fabric.register(translated, TensixCore(device, TENSIX_COORD_MAP[translated]))
    return device, fabric


def test_two_tiles_have_independent_l1_and_a_shared_dram():
    _, fabric = _build_fabric()

    payload_a = bytes.fromhex("aaaaaaaa11111111")
    payload_b = bytes.fromhex("bbbbbbbb22222222")
    fabric.write(WIRE_TILE_A, 0x2010, payload_a)
    fabric.write(WIRE_TILE_B, 0x2010, payload_b)
    assert fabric.read(WIRE_TILE_A, 0x2010, len(payload_a)) == payload_a
    assert fabric.read(WIRE_TILE_B, 0x2010, len(payload_b)) == payload_b

    dram_payload = bytes.fromhex("deadbeef" * 4)
    fabric.write(WIRE_DRAM, 0x200, dram_payload)
    assert fabric.read(WIRE_DRAM, 0x200, len(dram_payload)) == dram_payload


def test_resetting_one_tile_leaves_its_sibling_alone():
    _, fabric = _build_fabric()

    payload_b = bytes.fromhex("bbbbbbbb22222222")
    fabric.write(WIRE_TILE_B, 0x2010, payload_b)
    fabric.write(WIRE_TILE_A, 0x0, JAL_SELF)
    fabric.write(WIRE_TILE_B, 0x0, JAL_SELF)

    fabric.assert_reset(WIRE_TILE_A)
    fabric.deassert_reset(WIRE_TILE_A)
    # Pump a little by driving more traffic, as the wire flow would.
    for _ in range(4):
        fabric.read(WIRE_TILE_A, 0x2010, 8)

    assert fabric.read(WIRE_TILE_B, 0x2010, len(payload_b)) == payload_b


def test_every_worker_of_the_full_grid_resolves_to_itself_on_both_nocs():
    device, _ = _build_fabric(tiles=sorted(TENSIX_COORD_MAP))
    tt_device = device.tt_device
    grid_x, grid_y = tt_device.profile.noc_grid_x, tt_device.profile.noc_grid_y

    noc0, noc1 = tt_device.noc_0_directory, tt_device.noc_1_directory
    wrong_noc0 = [
        c for c in TENSIX_COORD_MAP if noc0.get(c) is None or noc0[c].id_pair != c
    ]
    assert wrong_noc0 == [], (
        f"NoC 0 misroutes {len(wrong_noc0)} workers: {wrong_noc0[:5]}"
    )

    wrong_noc1 = []
    for coord in TENSIX_COORD_MAP:
        mirror = (grid_x - 1 - coord[0], grid_y - 1 - coord[1])
        endpoint = noc1.get(mirror)
        if endpoint is None or endpoint.id_pair != coord:
            wrong_noc1.append((coord, mirror))
    assert wrong_noc1 == [], (
        f"NoC 1 misroutes {len(wrong_noc1)} workers by their mirror coord: "
        f"{wrong_noc1[:5]}"
    )
