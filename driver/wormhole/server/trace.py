"""Line-oriented trace format and recorder.

Each line records one host->sim message; whitespace-separated, ASCII so it's
diff-able. Fields:

    <CMD> core=<x>,<y> addr=0x<hex> size=<n> data=<hex|-> [reply=<hex|->]

CMD is the textual command name from ``protocol.CMD_NAMES``. Hex blobs are
lowercase with no leading ``0x``; ``-`` means "no data". ``reply=`` is only
present on READ lines and records the bytes the server returned.

Comment lines (``#``) and blank lines are skipped on parse.
"""

from . import protocol as proto

_NAME_TO_CMD = {v: k for k, v in proto.CMD_NAMES.items()}


def _hex(data: bytes) -> str:
    return data.hex() if data else "-"


def _parse_hex(s: str) -> bytes:
    return b"" if s == "-" else bytes.fromhex(s)


class TraceWriter:
    """Append-style trace recorder. Use as a context manager."""

    def __init__(self, path):
        self.path = path
        self._f = None

    def __enter__(self):
        self._f = open(self.path, "w")  # noqa: SIM115 — released in __exit__
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._f is not None:
            self._f.close()
            self._f = None

    def record(self, req, reply_data=None):
        if self._f is None:
            raise RuntimeError("TraceWriter used outside its context")
        name = proto.CMD_NAMES.get(req.cmd, f"CMD{req.cmd}")
        parts = [
            name,
            f"core={req.core[0]},{req.core[1]}",
            f"addr=0x{req.address:x}",
            f"size={req.size}",
            f"data={_hex(req.data)}",
        ]
        if req.cmd == proto.CMD_READ and reply_data is not None:
            parts.append(f"reply={_hex(reply_data)}")
        self._f.write(" ".join(parts) + "\n")
        self._f.flush()


def parse_trace_line(line: str) -> dict | None:
    """Parse one line into a dict: cmd, core, address, size, data, reply.

    Returns ``None`` for blank or comment lines. Raises ``ValueError`` for
    malformed lines.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    tokens = stripped.split()
    cmd_name = tokens[0]
    if cmd_name not in _NAME_TO_CMD:
        raise ValueError(f"unknown command in trace: {cmd_name!r}")

    fields = {}
    for tok in tokens[1:]:
        if "=" not in tok:
            raise ValueError(f"malformed token {tok!r} in trace line: {stripped!r}")
        k, v = tok.split("=", 1)
        fields[k] = v

    try:
        x_str, y_str = fields["core"].split(",")
        core = (int(x_str), int(y_str))
    except (KeyError, ValueError) as e:
        raise ValueError(f"bad core field in trace line: {stripped!r}") from e

    return {
        "cmd": _NAME_TO_CMD[cmd_name],
        "core": core,
        "address": int(fields.get("addr", "0"), 0),
        "size": int(fields.get("size", "0")),
        "data": _parse_hex(fields.get("data", "-")),
        "reply": _parse_hex(fields["reply"]) if "reply" in fields else None,
    }
