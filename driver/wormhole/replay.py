#!/usr/bin/env python3
"""Replay a captured wire trace against a running server.

The server must already be listening on the same nng address. By default the
address comes from ``$NNG_SOCKET_ADDR`` (matching what UMD exports when it
spawns the simulator). For READ messages, the server's reply is compared
against the recorded reply; mismatches print to stderr and exit non-zero
unless ``--no-verify`` is given.

Spin-polled READs — a (core, address) the host reads many times over, i.e. the
go-message mailbox it polls waiting for ``RUN_MSG_DONE`` — are replayed the way
a live host behaves rather than verbatim: the READ is re-sent until the reply
reaches the *final* value the recording captured for that location (bounded).
Each re-sent READ pumps the server its usual cycles-per-message, so a device
whose timing legitimately differs from the recording (``TT_SIM_COST_MODEL``)
simply gets polled longer, exactly as tt-metal itself would poll it. All other
READs — the result buffer included, which the host reads only after its poll
loop saw DONE — must still reproduce bit-for-bit.

Typical usage:

    # Terminal 1
    export NNG_SOCKET_ADDR=ipc:///tmp/replay.sock
    python3 -m driver.wormhole.server

    # Terminal 2
    export NNG_SOCKET_ADDR=ipc:///tmp/replay.sock
    python3 driver/wormhole/replay.py traces/some.trace
"""

import argparse
import os
import sys

import pynng

# Allow running this script directly without setting PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tt_sim.bridge import protocol as proto  # noqa: E402
from tt_sim.bridge.trace import parse_trace_line  # noqa: E402

#: READs from the same core at the same address, this many times or more, are a
#: spin-poll (matching driver/tests/cost_model_gate.py: in the captured traces
#: the polled go-message is read 30+ times and every other address once).
SPIN_POLL_READS = 8
#: Extra polls allowed per spin-polled READ before declaring the device hung.
#: Each poll advances the server its usual cycles-per-message (default 100),
#: so this bounds the wait at ~2M extra cycles — generous, and it fails
#: loudly rather than spinning for ever.
POLL_RETRY_CAP = 20_000


def _spin_poll_finals(trace_path):
    """Map each spin-polled (core, address) to its final recorded reply.

    The final recorded reply is the settled state the recorded host's poll
    loop was waiting for (RUN_MSG_DONE in the go-message); replaying "poll
    until the reply reaches it" is timing-independent where replaying the
    recorded poll count verbatim is not.
    """
    finals = {}
    counts = {}
    with open(trace_path) as f:
        for raw in f:
            entry = parse_trace_line(raw)
            if entry is None or entry["cmd"] != proto.CMD_READ:
                continue
            if entry["reply"] is None:
                continue
            key = (entry["core"], entry["address"])
            counts[key] = counts.get(key, 0) + 1
            finals[key] = entry["reply"]
    return {k: v for k, v in finals.items() if counts[k] >= SPIN_POLL_READS}


def _listen(addr):
    """Bind the host side of the pair socket.

    The wire protocol's host (UMD, or this replayer standing in for it) is the
    LISTENER; the simulator server is the DIALER (see
    ``tt_sim/bridge/transport.py``). The server's dial retries until a
    listener appears, so binding after the server started is fine.
    """
    try:
        return pynng.Pair1(listen=addr)
    except pynng.exceptions.NNGException as exc:
        print(f"error: could not listen on {addr}: {exc}", file=sys.stderr)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="replay")
    ap.add_argument("trace", help="path to trace file")
    ap.add_argument(
        "--addr",
        default=None,
        help="nng IPC address (default: $NNG_SOCKET_ADDR)",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="don't compare READ replies against recorded values",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N trace lines (debugging)",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-message progress",
    )
    args = ap.parse_args(argv)

    addr = args.addr or os.environ.get("NNG_SOCKET_ADDR")
    if not addr:
        print(
            "error: NNG_SOCKET_ADDR not set and --addr not provided",
            file=sys.stderr,
        )
        return 2

    sock = _listen(addr)
    if sock is None:
        return 1

    with sock:
        sock.recv_timeout = 10000
        sock.send_timeout = 10000

        ack = proto.parse(sock.recv())
        if ack.cmd != proto.CMD_EXIT:
            print(f"error: expected EXIT handshake, got cmd={ack.cmd}", file=sys.stderr)
            return 1
        if not args.quiet:
            print("[replay] received EXIT ack")

        finals = _spin_poll_finals(args.trace)
        sent = 0
        mismatches = 0
        with open(args.trace) as f:
            for lineno, raw in enumerate(f, start=1):
                if args.limit is not None and sent >= args.limit:
                    break
                entry = parse_trace_line(raw)
                if entry is None:
                    continue

                msg = proto.build_msg(
                    entry["cmd"],
                    data=entry["data"] or None,
                    core=entry["core"],
                    address=entry["address"],
                    size=entry["size"],
                )
                sock.send(msg)
                sent += 1

                if entry["cmd"] == proto.CMD_READ:
                    reply = proto.parse(sock.recv())
                    key = (entry["core"], entry["address"])
                    final = finals.get(key)
                    if final is not None:
                        # A spin-polled location: poll like a live host would,
                        # until the reply reaches the recording's final value
                        # (bounded). Intermediate values are timing, not data.
                        polls = 0
                        while (
                            reply.data[: len(final)] != final and polls < POLL_RETRY_CAP
                        ):
                            sock.send(msg)
                            reply = proto.parse(sock.recv())
                            polls += 1
                        if reply.data[: len(final)] != final:
                            mismatches += 1
                            if not args.quiet:
                                print(
                                    f"[replay] line {lineno}: spin-polled READ "
                                    f"core={entry['core']} "
                                    f"addr=0x{entry['address']:x} never reached "
                                    f"its final recorded value {final.hex()} "
                                    f"after {polls} extra polls "
                                    f"(last {reply.data[: len(final)].hex()})",
                                    file=sys.stderr,
                                )
                        continue
                    expected = entry["reply"]
                    if expected is not None and not args.no_verify:
                        actual = reply.data[: len(expected)]
                        if actual != expected:
                            mismatches += 1
                            if not args.quiet:
                                print(
                                    f"[replay] line {lineno}: READ "
                                    f"core={entry['core']} addr=0x{entry['address']:x} "
                                    f"size={entry['size']}: reply mismatch",
                                    file=sys.stderr,
                                )
                                print(
                                    f"           expected {expected.hex()}",
                                    file=sys.stderr,
                                )
                                print(
                                    f"           got      {actual.hex()}",
                                    file=sys.stderr,
                                )

                if entry["cmd"] == proto.CMD_EXIT:
                    break

        if not args.quiet:
            print(f"[replay] sent {sent} messages, {mismatches} READ mismatches")
        return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
