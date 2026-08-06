"""Regression test: replay the captured tt-metal "one" wire trace.

Spawns a fresh server (real ``Wormhole`` behind translated coords (1,1) and
(0,11)), then runs ``replay.py`` against ``server/traces/one.trace``. This is
the guard for the *socket* path — the same trace's values are pinned socket-
free by ``offline_replay_test``.

``replay.py`` replays spin-polled READs (the go-message the host polls waiting
for ``RUN_MSG_DONE``) the way a live host behaves — re-polling until the reply
reaches its final recorded value, bounded — and exits non-zero if any *other*
READ reply mismatches or a poll never settles. A clean run therefore means the
kernel ran to DONE and every data reply — the result buffer included —
reproduced bit-for-bit, independent of the recorded poll budget (so a
legitimate timing change such as ``TT_SIM_COST_MODEL`` cannot fail it
spuriously).

Skips with a clear message if the trace hasn't been captured yet. To capture::

    TT_SIM_RECORD=$(pwd)/driver/wormhole/server/traces/one.trace \\
    TT_METAL_SIMULATOR=$(pwd)/driver/wormhole <your tt-metal "one" program>

Run from anywhere::

    python3 -m driver.wormhole.server.one_replay_test
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRACE = REPO / "driver" / "wormhole" / "server" / "traces" / "one.trace"
REPLAY = REPO / "driver" / "wormhole" / "replay.py"


def main():
    if not TRACE.exists():
        print(
            f"skipped: {TRACE.relative_to(REPO)} not present\n"
            "         capture it first — see server/README.md "
            "'Capturing a real tt-metal trace'",
            file=sys.stderr,
        )
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        addr = f"ipc://{tmp}/one_replay.sock"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO)
        env["NNG_SOCKET_ADDR"] = addr

        server = subprocess.Popen(
            [sys.executable, "-m", "driver.wormhole.server", "--addr", addr],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for "listening on" then "Wormhole ready" — the latter signals
        # the server has finished building the (slow) tt-sim Wormhole.
        deadline = time.monotonic() + 30.0
        ready = False
        while time.monotonic() < deadline:
            line = server.stderr.readline()
            if not line:
                break
            sys.stderr.write(line.decode(errors="replace"))
            if b"Wormhole ready" in line:
                ready = True
                break
        if not ready:
            server.kill()
            raise AssertionError("server failed to come up within 30s")

        t0 = time.monotonic()
        replay = subprocess.run(
            [sys.executable, str(REPLAY), str(TRACE), "--quiet"],
            env=env,
            capture_output=True,
            text=True,
            timeout=300.0,  # a slower device (cost model on) polls longer
        )
        elapsed = time.monotonic() - t0

        server.wait(timeout=5.0)
        # Drain remaining server stderr after replay sent EXIT.
        sys.stderr.write(server.stderr.read().decode(errors="replace"))

        if replay.returncode != 0:
            sys.stderr.write(replay.stdout)
            sys.stderr.write(replay.stderr)
            raise AssertionError(f"replay.py exited {replay.returncode}")

    n_lines = sum(1 for _ in TRACE.open())
    print(f"one_replay test OK ({n_lines} trace messages, {elapsed:.1f}s)")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
