"""nng pair1 transport + dispatch loop.

UMD is the LISTENER (binds the port in ``simulation_host.cpp::init`` and
exports it via ``NNG_SOCKET_ADDR``); the simulator is the DIALER. On
connect, we immediately send an EXIT message as the "I'm alive"
handshake UMD expects in ``start_device()``.
"""

import sys

import pynng

from . import protocol as proto


class Transport:
    def __init__(self, addr, log_protocol=False, trace_writer=None):
        self.addr = addr
        self.log_protocol = log_protocol
        self.trace_writer = trace_writer
        self.msg_count = 0

    def serve(self, fabric):
        with pynng.Pair1(dial=self.addr) as sock:
            # Handshake: tell UMD we're alive.
            sock.send(proto.build_msg(proto.CMD_EXIT))
            self._log("sent EXIT ack")

            while True:
                try:
                    buf = sock.recv()
                except pynng.exceptions.Closed:
                    return

                req = proto.parse(buf)
                self.msg_count += 1
                self._log_request(req)

                reply_data = self._handle(fabric, req)

                if self.trace_writer is not None:
                    self.trace_writer.record(req, reply_data)

                if reply_data is not None:
                    sock.send(proto.build_msg(proto.CMD_READ, data=reply_data))

                if req.cmd == proto.CMD_EXIT:
                    # Host is shutting us down.
                    return

    def _handle(self, fabric, req):
        """Run side effects for ``req``; return reply payload bytes for READ."""
        if req.cmd == proto.CMD_WRITE:
            fabric.write(req.core, req.address, req.data)
            return None
        if req.cmd == proto.CMD_READ:
            return fabric.read(req.core, req.address, req.size)
        if req.cmd == proto.CMD_RESET_ASSERT:
            fabric.assert_reset(req.core)
            return None
        if req.cmd == proto.CMD_RESET_DEASSERT:
            fabric.deassert_reset(req.core)
            return None
        if req.cmd == proto.CMD_EXIT:
            return None
        if req.cmd == proto.CMD_START:
            # TODO(future): START (cmd=4) is reserved in the schema but never
            # observed in the tt-metal "one" trace and not sent by UMD's
            # ``tt_SimulationDevice``. If a future tt-metal release starts
            # sending it, the body of the message (data/core/address/size)
            # will inform what side effect is required.
            self._log(f"ignoring unhandled command START core={req.core}")
            return None
        raise ValueError(f"unknown command: {req.cmd}")

    def _log(self, msg):
        if self.log_protocol:
            print(f"[server] {msg}", file=sys.stderr, flush=True)

    def _log_request(self, req):
        if not self.log_protocol:
            return
        name = proto.CMD_NAMES.get(req.cmd, f"?({req.cmd})")
        print(
            f"[server] #{self.msg_count} {name} core={req.core} "
            f"addr=0x{req.address:x} size={req.size} data_len={len(req.data)}",
            file=sys.stderr,
            flush=True,
        )
