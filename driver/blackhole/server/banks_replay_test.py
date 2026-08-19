"""Socket-free replay of a captured tt-metal "banks" wire trace on Blackhole.

``banks`` is the only example in the tree whose DRAM buffers span **more than
one bank**. Every other example allocates a single-page DRAM buffer and reaches
it with ``get_noc_addr_from_bank_id<true>(0, ...)`` — bank 0, hardcoded — so a
simulator that modelled DRAM as one flat bank would replay all of them green.

What this guard is actually checking is that the host's *scatter* and the
kernel's *gather* meet. tt-metal splits an interleaved buffer round-robin over
the device's DRAM banks (8 on Blackhole: ``blackhole_140_arch.yaml`` declares
8 channels x 1 ``dram_view`` each; 12 on Wormhole, 6 channels x 2 views), and
the two sides compute the landing site independently:

* **Host**, per page, in ``WriteToDeviceInterleavedContiguous``:
  ``bank = page % num_banks``, ``addr = base + (page / num_banks) *
  aligned_page_size``, sent to that bank's own preferred worker core. That is
  visible in the trace as the interleaving below — 24 pages walking the 8
  coords in ``BANK_COORDS`` before the address advances by ``PAGE_SIZE``.
* **Device**, per page, in ``InterleavedAddrGen<true>::get_noc_addr``, from
  ``NUM_DRAM_BANKS`` (a JIT define) and the ``dram_bank_to_noc_xy`` /
  ``bank_to_dram_offset`` tables the host wrote into L1 at init.

Nothing cross-checks the two. If tt-sim aliased the banks onto one flat store,
or routed every bank to one endpoint, the kernel would still complete, every
address would still be a legal address, and the result would simply be wrong —
no fault, no ``TT_FATAL``, clean shutdown. So this test reads the destination
back **bank by bank**, at each bank's own coordinate and its own within-bank
offset, rather than as one contiguous range: reading it any other way would
not be able to tell the two layouts apart.

The failure is real and reachable: tt-metal's own emulation runner carries a
comment at ``tt_metal/impl/emulation/emulated_program_runner.cpp`` recording
that without the pow2/non-pow2 bank defines, "non-pow2 bank counts (12 on
WH-N150) silently fall through to a 0-bit shift and every page lands in bank
0".

Pumps until the go-message flips to DONE, like the other replays.

Run:  python3 -m driver.blackhole.server.banks_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "banks.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 2_000_000

# Blackhole's 8 DRAM banks in tt-metal bank order, read straight off the
# capture: the host writes page p of a buffer to ``BANK_COORDS[p % 8]``. Every
# Blackhole ``dram_view`` has ``address_offset: 0``, so a bank's base offset
# contributes nothing here and the whole of the layout is the coordinate plus
# the within-bank page stride.
BANK_COORDS = [(0, 11), (0, 2), (0, 9), (0, 5), (9, 11), (9, 3), (9, 8), (9, 6)]

# Third of the three 24-page allocations (src0 at 0x593e80, src1 at 0x594a80,
# dst here); each occupies 24/8 = 3 pages per bank, i.e. 3 * 0x400 per bank.
DST_ADDR = 0x595680
PAGE_SIZE = 1024
N_PAGES = 24
PAGE_ELEMS = PAGE_SIZE // 4


def _expected(index):
    """``src0[i] + src1[i]`` for the host's generators, mod 2**32."""
    return ((0x1000 + index) + (0x7000000 - 3 * index)) & 0xFFFFFFFF


def _build_fabric():
    device = make_device()
    fabric = Fabric()
    for translated, tile in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, tile))
    for physical in TENSIX_POOL:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
    return device, fabric


def _go_signal(device, core):
    return device.tt_device.read(core, GO_MSG_ADDR, 4)[3]


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric()
    transport = Transport(addr=None)

    n_msgs = 0
    with TRACE.open() as f:
        for line in f:
            parsed = parse_trace_line(line)
            if parsed is None:
                continue
            req = SimpleNamespace(
                cmd=parsed["cmd"],
                core=parsed["core"],
                address=parsed["address"],
                size=parsed["size"],
                data=parsed["data"],
            )
            transport._handle(fabric, req)
            n_msgs += 1
            if (
                parsed["cmd"] == proto.CMD_READ
                and parsed["core"] in TENSIX_POOL
                and parsed["address"] == GO_MSG_ADDR
                and _go_signal(device, parsed["core"]) == RUN_MSG_GO
            ):
                pumped = 0
                while (
                    _go_signal(device, parsed["core"]) != RUN_MSG_DONE
                    and pumped < PUMP_CAP
                ):
                    device.tt_device.run(PUMP_CHUNK)
                    pumped += PUMP_CHUNK

    # Read the destination back the way the host wrote it: page by page, each
    # from its own bank's coordinate at its own within-bank offset. Reading it
    # as one contiguous range at one coordinate could not distinguish a banked
    # layout from a flat one, which is the whole point of the example.
    n_banks = len(BANK_COORDS)
    wrong = []
    banks_touched = set()
    for page in range(N_PAGES):
        coord = BANK_COORDS[page % n_banks]
        addr = DST_ADDR + (page // n_banks) * PAGE_SIZE
        raw = device.read(coord, addr, PAGE_SIZE)
        banks_touched.add(coord)
        for j in range(PAGE_ELEMS):
            got = int.from_bytes(raw[j * 4 : j * 4 + 4], "little")
            index = page * PAGE_ELEMS + j
            if got != _expected(index):
                wrong.append((page, coord, index, got, _expected(index)))
    device.tt_device.shutdown()

    if wrong:
        page, coord, index, got, want = wrong[0]
        raise AssertionError(
            f"{len(wrong)}/{N_PAGES * PAGE_ELEMS} result elements wrong replaying "
            f"{TRACE.name} (interleaved DRAM buffer over {n_banks} banks); first: "
            f"page {page} in bank {page % n_banks} at {coord} element {index} "
            f"= 0x{got:08x}, expected 0x{want:08x}"
        )
    if len(banks_touched) != n_banks:
        raise AssertionError(
            f"only {len(banks_touched)} of {n_banks} DRAM banks were read back "
            f"({sorted(banks_touched)}) — the buffer did not span the banks, so "
            f"this replay is no longer testing what it claims to"
        )
    print(
        f"blackhole banks_replay test OK ({n_msgs} messages; all {N_PAGES} pages "
        f"correct across all {n_banks} DRAM banks)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
