"""Socket-free replay of a captured tt-metal "six" wire trace on Blackhole.

``six`` is a 128x128x128 bf16 matmul (``matmul_tiles`` -> the Tensix matrix
engine's MVMUL with K-accumulation). It is the first example to exercise the
full matmul path, and shook out three Blackhole bugs the earlier examples never
reached (see ``docs/plans/blackhole-support.md`` and the ``six`` handoff):

1. **Interleaved DRAM.** The A/B/C buffers are ``InterleavedBufferConfig`` — the
   host round-robins each buffer's tiles across all 8 DRAM channels. The bring-up
   profile modelled only channel 0, so 7 of every 8 input tiles read back zero.
   Fixed by modelling all 8 channels (``BLACKHOLE_PROFILE.dram_channel_*``).
2. **MVMUL addr_mode decode.** Blackhole widens the math ``addr_mode`` field to
   3 bits (raw 16:14) vs Wormhole's 2 bits (16:15); reading it at the Wormhole
   offset dropped ADDR_MOD_5, so the second K-face of each tile never
   accumulated (each output was ~half its value). Fixed in the matrix backend.
3. **Packer Y/Z advance.** The pack ADC Y/Z counters must *accumulate* across the
   PACRs that pack one multi-face tile (they *set* before), so only the first
   16x16 face of the 32x32 output landed in the right rows. four/five pack a
   single face and never hit it. Fixed in the packer backend.

Because the inputs are random (seeded host-side) the golden isn't a fixed
constant: we read A, B and the device's C back out of the interleaved DRAM,
recompute ``A @ B`` on the host, and compare with Pearson correlation >= 0.97 —
exactly the tolerance ``six``'s host program uses (bf16 + HiFi won't be
bit-exact).

Run:  python3 -m driver.blackhole.server.six_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tt_sim.arch import BLACKHOLE_PROFILE
from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "six.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Matrix dims (six.cpp): M=N=K=128, 32x32 tiles -> 4x4 tiles per matrix.
MT = KT = NT = 4
PAGE = 2048  # bf16 32x32 tile, bytes

# Interleaved-buffer DRAM layout. The three buffers are allocated back-to-back;
# each is round-robined tile-by-tile across the 8 DRAM channels (page p -> bank
# p % 8, at base + (p // 8) * PAGE). Bank order is the profile's channel order.
BANKS = list(BLACKHOLE_PROFILE.dram_channel_unified_coords)
A_BASE, B_BASE, C_BASE = 0x593E80, 0x594E80, 0x595E80

PCC_THRESHOLD = 0.97


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


def _bf16_to_f32(u16):
    return np.frombuffer((u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32)


def _untilize(u16):
    """A tt 32x32 bf16 tile is 4 row-major 16x16 faces (TL, TR, BL, BR)."""
    out = np.zeros((32, 32), dtype=np.float32)
    flat = _bf16_to_f32(u16)
    for i, (fr, fc) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        out[fr * 16 : fr * 16 + 16, fc * 16 : fc * 16 + 16] = flat[
            i * 256 : i * 256 + 256
        ].reshape(16, 16)
    return out


def _read_matrix(device, base, tiles_r, tiles_c):
    mat = np.zeros((tiles_r * 32, tiles_c * 32), dtype=np.float32)
    for tr in range(tiles_r):
        for tc in range(tiles_c):
            page = tr * tiles_c + tc
            coord = BANKS[page % len(BANKS)]
            addr = base + (page // len(BANKS)) * PAGE
            raw = np.frombuffer(device.read(coord, addr, PAGE), dtype=np.uint16)
            mat[tr * 32 : tr * 32 + 32, tc * 32 : tc * 32 + 32] = _untilize(raw)
    return mat


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

    a = _read_matrix(device, A_BASE, MT, KT)
    b = _read_matrix(device, B_BASE, KT, NT)
    c_device = _read_matrix(device, C_BASE, MT, NT)
    device.tt_device.shutdown()

    golden = a @ b
    pcc = float(np.corrcoef(golden.flatten(), c_device.flatten())[0, 1])

    if pcc < PCC_THRESHOLD:
        raise AssertionError(
            f"replaying {TRACE.name}: matmul PCC {pcc:.4f} < {PCC_THRESHOLD} "
            f"(golden mean {golden.mean():.2f}, device mean {c_device.mean():.2f})"
        )
    print(
        f"blackhole six_replay test OK ({n_msgs} messages; 128^3 bf16 matmul, "
        f"PCC(golden, device) = {pcc:.4f} >= {PCC_THRESHOLD})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
