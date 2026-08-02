"""Byte-identical offline replay of every captured Wormhole example trace.

The Blackhole side guards each example with a per-example *value* test
(``driver/blackhole/server/<name>_replay_test.py``, which pumps the kernel and
checks the result buffer). Wormhole instead uses the **byte-identical** style of
``offline_replay_test.py``, generalised here to every example: replay the
captured wire trace and assert that every host READ reply reproduces bit-for-bit
against the live run. Because the live run validated its own result and the host
reads the result buffer back over the wire, a byte-identical replay is a full
correctness guard — and it needs no per-example coord translation or result
address (the trace carries the expected replies).

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

# The go-message mailbox the host spin-polls waiting for RUN_MSG_DONE; its
# intermediate poll values depend on cycle-exact firmware timing the functional
# sim need not reproduce, so mismatches at these addresses are tolerated (as are
# Ethernet reads). 0x2a0 is the older tt-metal go-message offset, 0x4a0 the
# current one — tolerate both so the test survives a tt-metal bump.
GO_MSG_ADDR = 0x4A0
_TRANSIENT_ADDRS = {0x2A0, 0x4A0}


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


def replay(name):
    """Byte-identical replay of one example trace; returns a result dict or None."""
    trace = TRACES / f"{name}.trace"
    if not trace.exists():
        return None

    device = make_device()
    fabric = Fabric()
    for translated, unified in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, unified))
    for translated, unified in ETH_COORD_MAP.items():
        fabric.register(translated, EthCore(device, unified))
    for physical in _discover_pool(trace):
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
            reads += 1
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
            f"{name}: {r['mismatches']}/{r['reads']} READ replies mismatched "
            f"(beyond tolerated eth/mailbox)\n    " + "\n    ".join(r["first"])
        )
    return "ok", (
        f"{name}: {r['verified']}/{r['reads']} READs bit-for-bit "
        f"({r['tolerated']} eth/mailbox tolerated)"
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
