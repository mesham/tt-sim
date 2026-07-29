"""Thin helpers over the generated flatbuffer bindings.

Converts ``data`` to/from ``bytes`` at this boundary so the rest of the server
never deals with uint32 vectors directly.
"""

from dataclasses import dataclass

import flatbuffers

from ._flatbuf import DeviceRequestResponse as DRR
from ._flatbuf import tt_vcs_core as TVC

CMD_WRITE = 0
CMD_READ = 1
CMD_RESET_DEASSERT = 2
CMD_RESET_ASSERT = 3
CMD_START = 4
CMD_EXIT = 5

CMD_NAMES = {
    CMD_WRITE: "WRITE",
    CMD_READ: "READ",
    CMD_RESET_DEASSERT: "RESET_DEASSERT",
    CMD_RESET_ASSERT: "RESET_ASSERT",
    CMD_START: "START",
    CMD_EXIT: "EXIT",
}


@dataclass
class Request:
    cmd: int
    core: tuple[int, int]
    address: int
    size: int
    data: bytes


def build_msg(
    cmd: int,
    *,
    data: bytes | None = None,
    core: tuple[int, int] | None = None,
    address: int = 0,
    size: int = 0,
) -> bytes:
    b = flatbuffers.Builder(256)
    data_vec = None
    if data:
        padded = data + b"\x00" * ((-len(data)) % 4)
        n_words = len(padded) // 4
        DRR.DeviceRequestResponseStartDataVector(b, n_words)
        for i in range(n_words - 1, -1, -1):
            word = int.from_bytes(padded[i * 4 : (i + 1) * 4], "little")
            b.PrependUint32(word)
        data_vec = b.EndVector()

    DRR.DeviceRequestResponseStart(b)
    DRR.DeviceRequestResponseAddCommand(b, cmd)
    if data_vec is not None:
        DRR.DeviceRequestResponseAddData(b, data_vec)
    if core is not None:
        # Struct must be inlined inside the table build sequence.
        core_off = TVC.Creatett_vcs_core(b, core[0], core[1])
        DRR.DeviceRequestResponseAddCore(b, core_off)
    if address:
        DRR.DeviceRequestResponseAddAddress(b, address)
    if size:
        DRR.DeviceRequestResponseAddSize(b, size)
    root = DRR.DeviceRequestResponseEnd(b)
    b.Finish(root)
    return bytes(b.Output())


def parse(buf: bytes) -> Request:
    req = DRR.DeviceRequestResponse.GetRootAsDeviceRequestResponse(buf, 0)
    core_obj = req.Core()
    coord = (int(core_obj.X()), int(core_obj.Y())) if core_obj is not None else (0, 0)

    n = req.DataLength()
    if n:
        np_arr = req.DataAsNumpy()
        data_bytes = bytes(np_arr.tobytes())
    else:
        data_bytes = b""

    return Request(
        cmd=int(req.Command()),
        core=coord,
        address=int(req.Address()),
        size=int(req.Size()),
        data=data_bytes,
    )
