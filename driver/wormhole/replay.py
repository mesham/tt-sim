#!/usr/bin/env python3
"""Replay a captured wire trace against a running server.

The server must already be listening on the same nng address. By default the
address comes from ``$NNG_SOCKET_ADDR`` (matching what UMD exports when it
spawns the simulator). For READ messages, the server's reply is compared
against the recorded reply; mismatches print to stderr and exit non-zero
unless ``--no-verify`` is given.

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
import time

import pynng

# Allow running this script directly without setting PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from driver.wormhole.server import protocol as proto  # noqa: E402
from driver.wormhole.server.trace import parse_trace_line  # noqa: E402


def _dial_with_retry(addr, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return pynng.Pair1(dial=addr, block_on_dial=False)
        except pynng.exceptions.ConnectionRefused:
            time.sleep(0.1)
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

    sock = _dial_with_retry(addr)
    if sock is None:
        print(f"error: could not dial server at {addr}", file=sys.stderr)
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
