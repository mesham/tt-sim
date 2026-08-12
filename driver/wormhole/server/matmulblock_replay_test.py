"""Socket-free replay of the captured tt-metal "matmulblock" wire trace on Wormhole.

The Wormhole twin of ``driver/blackhole/server/matmulblock_replay_test.py``, and
like ``matmulidx`` a *value* check rather than a replay of recorded READ replies
(see ``optests/diff.sh``, which takes ``TT_SIM_ARCH=wormhole``).

``matmulblock`` (see ``optests/matmulblock``) is the block twin of
``matmulidx``: where ``matmul_tiles`` unpacks one operand tile per Src bank,
``matmul_block`` walks several of them per call at addresses the unpacker's MOP
computes for itself, so it exercises far more of the operand addressing --
``_llk_unpack_AB_matmul_`` reprograms ``THCON_SEC{0,1}_REG3`` base addresses per
step while the replay buffer bumps them by a tile between unpacks, alternating
the two config contexts.

Both operand CBs hold a whole 2x2 tile block, resident at once (A: 1.0, 2.0,
4.0, 8.0; B: 3.0, 5.0, 16.0, 32.0), so for 32x32 constant tiles every element of
C = A @ B is ``32 * sum_k A[r][k] * B[k][c]`` -- 1120 / 2208 / 4480 / 8832, all
exactly representable in bfloat16. The golden is therefore **computed here**
rather than frozen from a dump, and the comparison is bit-exact rather than by
PCC.

One ``matmul_block`` call is a *single* K step: ``kt_dim`` only supplies operand
A's row stride, so the kernel loops K itself. It emits the same C block three
times under the three interesting shapes, covering both sides of the LLK's
``reuse_a = ct_dim >= rt_dim`` split:

    ct=2 rt=1  reuse A, one output tile-row per call
    ct=2 rt=2  reuse A, the whole 2x2 block per call
    ct=1 rt=2  reuse B (the ``!reuse_a`` branch), one output tile-column per call

All 12288 bf16 elements must match. (``diff.sh`` reports 6144: it assumes 8 hex
chars per element, which halves the count for a bfloat16 program.)

This program is also the tree's sensor for **config-write ordering**, which is
why ``test_matmulblock_replay_with_a_multi_cycle_config_unit`` runs it a second
time with the Tensix config unit charged more than one cycle. See
``_ForcedConfigOccupancy`` for what that is guarding.

Run:  python3 -m driver.wormhole.server.matmulblock_replay_test
      (or under pytest, as ``test_matmulblock_replay``)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "matmulblock.trace"
# Single worker at Wormhole physical coord (1, 1) — logical (0, 0).
TENSIX_POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Output DRAM buffer: twelve 32x32 bf16 tiles, contiguous in one interleaved
# page at channel (0, 11). The two operand blocks (8 KiB each) are allocated
# first.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D8C40
TILE_ELEMS = 1024
TILE_BYTES = 2 * TILE_ELEMS

RT = 2
CT = 2
KT = 2
# Constant tile values, row-major: A is RT x KT, B is KT x CT.
A_VALUES = (1.0, 2.0, 4.0, 8.0)
B_VALUES = (3.0, 5.0, 16.0, 32.0)
# Per shape, the (r, c) of each packed output tile, in pack order.
SHAPES = (
    ("ct=2 rt=1 (reuse A, one output row per call)", ((0, 0), (0, 1), (1, 0), (1, 1))),
    ("ct=2 rt=2 (reuse A, whole block per call)", ((0, 0), (0, 1), (1, 0), (1, 1))),
    (
        "ct=1 rt=2 (reuse B, one output column per call)",
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ),
)
NUM_TILES = sum(len(tiles) for _, tiles in SHAPES)

GOLDEN = {
    (r, c): 32.0 * sum(A_VALUES[r * KT + k] * B_VALUES[k * CT + c] for k in range(KT))
    for r in range(RT)
    for c in range(CT)
}


class _ForcedConfigOccupancy:
    """A stand-in cost model that charges the Tensix config unit ``cycles``.

    Stands in for a unit cost model (``tt_sim/perf/model.py``) so that the
    *ordering* guarantee around a config write can be exercised without editing
    the cost tables — deliberately a stand-in and not the real thing, so this
    guard stays outside the allow-list of modules that read those tables. Every
    entry the config unit has on Wormhole is a 1-cycle occupancy, and
    ``RDCFG``'s 1 in particular is a Blackhole-silicon measurement corroborated
    by two hardware runs, so it must not be nudged upwards to reach a code path.

    ``RDCFG`` is the opcode charged because it is the one that found the bug: a
    2-cycle occupancy on it used to defer the math thread's ``SETC16`` of
    ``DEST_TARGET_REG_CFG_MATH_Offset`` — accepted in the same cycle, so the
    thread had already moved on — past two of that thread's own ``MVMUL``s,
    which then accumulated into the wrong half of Dst and made this program
    print 608.0 where it should print 1120.0. See
    ``TensixBackendUnit.clock_tick``.
    """

    def __init__(self, cycles):
        self.cycles = cycles

    def occupancy(self, instruction_name):
        return self.cycles if instruction_name == "RDCFG" else 1

    def latency(self, instruction_name):
        """The real Wormhole table's Latency column, which this leaves alone.

        Residency and occupancy are different columns (the config unit reads
        both), so forcing one must not silently move the other: this guard is
        about a *throughput* hold reordering an accepted write, and reporting a
        longer residency alongside it would confound the two. The numbers are
        ``ConfigurationUnit.md``'s: 2 cycles for ``WRCFG`` and ``RDCFG``, 1 for
        everything else this unit executes.
        """
        return 2 if instruction_name in ("WRCFG", "RDCFG") else 1

    def is_exact(self, instruction_name):
        return True

    #: No IPC groups, which is the *Wormhole* config unit's real answer: that
    #: arch's page states its throughput limits as prose and publishes no "IPC
    #: group" column, so occupancy there is a whole-unit hold. (Blackhole's page
    #: does publish one, and its entries carry ``ipc_group``.) Stated rather
    #: than omitted because it is what keeps this guard testing the thing it was
    #: written for: with no groups, the ``RDCFG`` hold below refuses every
    #: opcode of every thread, exactly as it did before groups existed.
    has_ipc_groups = False

    def ipc_group(self, instruction_name):
        return None


def _build_fabric(config_unit_occupancy=None):
    device = make_device()
    fabric = Fabric()
    for translated, unified in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, unified))
    for translated, unified in ETH_COORD_MAP.items():
        fabric.register(translated, EthCore(device, unified))
    for physical in TENSIX_POOL:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
    if config_unit_occupancy is not None:
        for tile in device.tt_device.tensix_tiles:
            backend = tile.tensix_coprocessor.getBackend()
            backend.config_unit.cost_model = _ForcedConfigOccupancy(
                config_unit_occupancy
            )
    return device, fabric


def _go_signal(device, core):
    return device.tt_device.read(TENSIX_COORD_MAP[core], GO_MSG_ADDR, 4)[3]


def _bf16(raw, offset):
    """Widen the little-endian bfloat16 at ``offset`` to a Python float."""
    bits = int.from_bytes(raw[offset : offset + 2], "little") << 16
    sign = -1.0 if bits >> 31 else 1.0
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0:
        return sign * mantissa * 2.0**-149
    return sign * (mantissa + (1 << 23)) * 2.0 ** (exponent - 150)


def main(config_unit_occupancy=None):
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric(config_unit_occupancy)
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
            # Pump the worker until its go-message reports DONE (bounded).
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

    raw = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, NUM_TILES * TILE_BYTES)
    device.tt_device.shutdown()

    tile = 0
    for shape, positions in SHAPES:
        for r, c in positions:
            expected = GOLDEN[(r, c)]
            for datum in range(TILE_ELEMS):
                got = _bf16(raw, tile * TILE_BYTES + datum * 2)
                if got != expected:
                    raise AssertionError(
                        f"replaying {TRACE.name}: matmul_block {shape} output tile "
                        f"{tile} (C[{r}][{c}]) datum {datum} is {got}, "
                        f"expected {expected}"
                    )
            tile += 1
    charged = (
        ""
        if config_unit_occupancy is None
        else f"; config unit charged {config_unit_occupancy} cycles"
    )
    print(
        f"wormhole matmulblock_replay test OK ({n_msgs} messages; all "
        f"{NUM_TILES * TILE_ELEMS} bf16 elements match across the three "
        f"matmul_block shapes {[shape for shape, _ in SHAPES]}{charged})"
    )
    return 0


def test_matmulblock_replay():
    assert main() == 0


def test_matmulblock_replay_with_a_multi_cycle_config_unit():
    """The same program, with the config unit held for more than one cycle.

    The regression test for tt-sim's config-write ordering, run end to end
    rather than at the unit (``tt_sim/pe/tensix/backend_cost_model_test.py``
    pins the same invariant one cycle at a time). Charging ``RDCFG`` two cycles
    used to make this program print 608.0 for C[0][0] instead of 1120.0, because
    a ``SETC16`` the config unit had already accepted was pushed behind two of
    the issuing thread's own later instructions.

    The occupancy must reach the unit *without* changing the cost tables, so it
    arrives as ``_ForcedConfigOccupancy``. Nothing about this run is a claim
    that ``RDCFG`` costs two cycles — silicon says it costs one — only that the
    simulator stays correct if some future entry in this unit costs more.
    """
    assert main(config_unit_occupancy=2) == 0


if __name__ == "__main__":
    raise SystemExit(main())
