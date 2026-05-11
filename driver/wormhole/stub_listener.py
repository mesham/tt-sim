#!/usr/bin/env python3
import os, sys, flatbuffers, pynng

sys.path.insert(0, "/tmp/fb_py")
import DeviceRequestResponse as DRR

WRITE, READ, RESET_DEASSERT, RESET_ASSERT, START, EXIT = 0, 1, 2, 3, 4, 5
CMD_NAMES = {0: "WRITE", 1: "READ", 2: "RESET_DEASSERT",
             3: "RESET_ASSERT", 4: "START", 5: "EXIT"}

def build_msg(command: int, data=None) -> bytes:
    b = flatbuffers.Builder(256)
    data_vec = None
    if data:
        DRR.DeviceRequestResponseStartDataVector(b, len(data))
        for v in reversed(data):
            b.PrependUint32(v)
        data_vec = b.EndVector()
    DRR.DeviceRequestResponseStart(b)
    DRR.DeviceRequestResponseAddCommand(b, command)
    if data_vec is not None:
        DRR.DeviceRequestResponseAddData(b, data_vec)
    root = DRR.DeviceRequestResponseEnd(b)
    b.Finish(root)
    return bytes(b.Output())

def parse(msg):
    req = DRR.DeviceRequestResponse.GetRootAsDeviceRequestResponse(msg, 0)
    core = req.Core()
    data_len = req.DataLength()
    data = [req.Data(i) for i in range(data_len)]
    return {
        "cmd": req.Command(),
        "core": (core.X(), core.Y()) if core else None,
        "address": req.Address(),
        "size": req.Size(),
        "data": data,
    }

addr = os.environ["NNG_SOCKET_ADDR"]
print(f"[stub] listening on {addr}", file=sys.stderr, flush=True)

with pynng.Pair1(listen=addr) as sock:
    print("[stub] socket bound", file=sys.stderr, flush=True)

    sock.send(build_msg(EXIT))
    print("[stub] sent EXIT ack", file=sys.stderr, flush=True)

    n = 0
    while True:
        msg = sock.recv()
        n += 1
        r = parse(msg)
        name = CMD_NAMES.get(r["cmd"], f"?({r['cmd']})")
        print(f"[stub] #{n} {name} core={r['core']} addr=0x{r['address']:x} "
              f"size={r['size']} data_len={len(r['data'])}",
              file=sys.stderr, flush=True)

        if r["cmd"] == READ:
            reply_words = max(1, r["size"] // 4)
            sock.send(build_msg(READ, data=[0] * reply_words))
            print(f"[stub] -> READ reply: {reply_words} zero words",
                  file=sys.stderr, flush=True)
