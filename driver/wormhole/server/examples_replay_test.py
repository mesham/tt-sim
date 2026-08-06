"""Offline value replay of every captured Wormhole example trace.

Mirrors the Blackhole guards' *poll-until-DONE* shape: replay the captured
host->device traffic (writes, resets) exactly as recorded, but instead of
requiring the device to finish the kernel within the host's recorded poll
count, pump the device until each launched worker's go-message reaches
``RUN_MSG_DONE`` (bounded — a hung kernel fails loudly rather than spinning),
*then* verify values. That makes the guard independent of the recorded poll
budget, so a legitimate timing change (``TT_SIM_COST_MODEL``) cannot fail it
spuriously; a wrong value still can.

Value coverage is not weakened relative to the old byte-identical style: every
non-spin-polled READ reply is still asserted bit-for-bit against the recording
— crucially including the host's final result-buffer read-back, which in every
captured trace happens after the last go-poll and therefore reads settled
post-DONE state. The only replies exempted are the go-message mailbox (whose
intermediate values are literally "how far had the kernel got when the host
looked" — unreproducible under any timing change, in either direction) and
Ethernet-core reads (the traces predate ``EthTile``). Because the live run
validated its own result and the host read that result back over the wire, the
recorded replies double as the frozen expected data — no separate ``.expected``
blob is needed.

Traces are captured by ``driver/wormhole/tests/capture_traces.sh`` (build + run
each shared example against ``driver/wormhole`` with ``TT_SIM_RECORD``). Missing
traces are skipped, so this is safe to run with only a subset captured.

Run:  python3 -m driver.wormhole.server.examples_replay_test
"""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

TRACES = Path(__file__).resolve().parent / "traces"
EXAMPLES = [
    "one",
    "two",
    "three",
    "four",
    "four-fp",
    "five",
    "five-fp",
    "six",
    "eight",
    "nine",
    "loopback",
]

# The go-message mailbox the host spin-polls waiting for RUN_MSG_DONE. Its
# intermediate poll values depend on cycle-exact timing no timing model need
# reproduce, so replies at these addresses are never compared (on any core: the
# non-worker cores get a single setup read here too, and 0x2a0 is the older
# tt-metal go-message offset — tolerate both so the test survives a tt-metal
# bump). Completion is instead asserted directly: each launched worker is
# pumped until its go-message reads RUN_MSG_DONE.
GO_MSG_ADDR = 0x4A0
_TRANSIENT_ADDRS = {0x2A0, 0x4A0}

RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000


def _discover_pool(trace):
    """Tensix tiles the host spin-polls (the compute workers), by go-read count.

    Every core gets a single setup read at the go-message address; only the
    launched worker(s) are polled repeatedly, so ``count > 1`` selects them
    (one tile for single-core examples, two for ``nine``).
    """
    counts = Counter()
    for line in trace.open():
        p = parse_trace_line(line)
        if p and p["cmd"] == proto.CMD_READ and p["address"] == GO_MSG_ADDR:
            counts[p["core"]] += 1
    return sorted(core for core, n in counts.items() if n > 1)


def _go_signal(device, core):
    return device.tt_device.read(TENSIX_COORD_MAP[core], GO_MSG_ADDR, 4)[3]


def _pump_until_done(device, core, name):
    """Pump the device until ``core``'s go-message reads DONE (bounded)."""
    pumped = 0
    while _go_signal(device, core) != RUN_MSG_DONE and pumped < PUMP_CAP:
        device.tt_device.run(PUMP_CHUNK)
        pumped += PUMP_CHUNK
    if _go_signal(device, core) != RUN_MSG_DONE:
        raise AssertionError(
            f"{name}: worker {core} go-message never reached RUN_MSG_DONE "
            f"within {PUMP_CAP} pumped cycles (still "
            f"{_go_signal(device, core):#04x})"
        )


def replay(name):
    """Poll-until-DONE value replay of one example trace; a dict or None."""
    trace = TRACES / f"{name}.trace"
    if not trace.exists():
        return None

    device = make_device()
    fabric = Fabric()
    for translated, unified in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, unified))
    for translated, unified in ETH_COORD_MAP.items():
        fabric.register(translated, EthCore(device, unified))
    pool = _discover_pool(trace)
    for physical in pool:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))

    transport = Transport(addr=None)
    eth = set(ETH_COORD_MAP)
    reads = verified = tolerated = mismatches = 0
    first = []
    with trace.open() as f:
        for lineno, line in enumerate(f, 1):
            p = parse_trace_line(line)
            if p is None:
                continue
            reply = transport._handle(
                fabric,
                SimpleNamespace(
                    cmd=p["cmd"],
                    core=p["core"],
                    address=p["address"],
                    size=p["size"],
                    data=p["data"],
                ),
            )
            if p["cmd"] != proto.CMD_READ or p["reply"] is None:
                continue
            # Pump the worker until its go-message reports DONE (bounded):
            # this is where the recorded poll budget stops mattering.
            if (
                p["address"] == GO_MSG_ADDR
                and p["core"] in pool
                and _go_signal(device, p["core"]) == RUN_MSG_GO
            ):
                _pump_until_done(device, p["core"], name)
            reads += 1
            # Compare first: an exempt read that happens to match still counts
            # as verified (model off, everything reproduces). The exemption
            # only decides what a mismatch means.
            if bytes(reply) == bytes(p["reply"]):
                verified += 1
            elif p["core"] in eth or p["address"] in _TRANSIENT_ADDRS:
                tolerated += 1
            else:
                mismatches += 1
                if len(first) < 5:
                    first.append(
                        f"line {lineno} READ {p['core']}@0x{p['address']:x}: "
                        f"expected {p['reply'].hex()} got {bytes(reply).hex()}"
                    )
    device.tt_device.shutdown()
    return {
        "reads": reads,
        "verified": verified,
        "tolerated": tolerated,
        "mismatches": mismatches,
        "first": first,
    }


def _check(name):
    r = replay(name)
    if r is None:
        return "skip", f"{name}: trace not present"
    if r["mismatches"]:
        return "fail", (
            f"{name}: {r['mismatches']}/{r['reads']} data READ replies mismatched "
            f"after pump-to-DONE\n    " + "\n    ".join(r["first"])
        )
    return "ok", (
        f"{name}: pumped to DONE; {r['verified']}/{r['reads']} data READs "
        f"bit-for-bit ({r['tolerated']} go-message/eth reads exempt)"
    )


# --- pytest entry point (only defined when pytest is installed) ---------------
try:
    import pytest

    @pytest.mark.parametrize("name", EXAMPLES)
    def test_example_replay(name):
        if not (TRACES / f"{name}.trace").exists():
            pytest.skip(f"{name}.trace not present")
        status, msg = _check(name)
        assert status == "ok", msg

except ImportError:
    pass


def main():
    any_fail = False
    for name in EXAMPLES:
        status, msg = _check(name)
        print(f"{'OK  ' if status == 'ok' else status.upper() + ' '} {msg}")
        any_fail = any_fail or status == "fail"
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
