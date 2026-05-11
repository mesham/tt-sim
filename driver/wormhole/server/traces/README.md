# Captured wire traces

This directory holds captured `host → sim` wire traces (text format, one
message per line — see `server/README.md` § "Trace format"). They serve as
regression fixtures for `replay.py`: any server change can be re-validated
against a recorded conversation without needing tt-metal in the loop.

## Recommended naming

`<example>.trace` for canonical end-to-end captures (e.g. `one.trace`,
`loopback.trace`).
`<example>_<variant>.trace` where the same example is captured under
different server modes — e.g. `one_mocktensix.trace` is the same tt-metal
run against `--mock-tensix` (all replies are zeros, useful as a transport
regression).

## Capturing

See the parent README's "Capturing a real tt-metal trace" section. Short
form:

```bash
NNG_SOCKET_ADDR=ipc:///tmp/cap.sock python3 -m driver.wormhole.server \
    --addr ipc:///tmp/cap.sock --mock-tensix \
    --record driver/wormhole/server/traces/one_mocktensix.trace &
TT_METAL_SIMULATOR=$(pwd)/driver/wormhole <your tt-metal program>
```

## Replaying

```bash
NNG_SOCKET_ADDR=ipc:///tmp/replay.sock python3 -m driver.wormhole.server \
    --addr ipc:///tmp/replay.sock --mock-tensix &
python3 driver/wormhole/replay.py driver/wormhole/server/traces/one_mocktensix.trace
```

Replay exits 0 iff every READ reply matches the recorded value. Use
`--no-verify` to drive traffic through the server without comparing.
