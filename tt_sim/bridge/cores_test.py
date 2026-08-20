"""What a stand-in core answers when the host reads back its own writes.

The bug this pins: ``DPrintServer`` writes a magic word into every core's
``dprint_buf`` and then spins up to 100000 times waiting to read it back
(``dprint_server.cpp:WriteInitMagic``). tt-sim's stand-ins zero-filled every
read, so the spin could never succeed and **any** tt-metal run with DPRINT
enabled died with ``TT_THROW: Timed out writing init magic`` after ~2 minutes —
taking the LLK sanitizer, and kernel printf debugging generally, with it.

Both directions are pinned, because the fix has an exception that is easy to
lose: the go message must go on reading ``RUN_MSG_DONE`` no matter what
run-state the host wrote, or the grid-wide init handshake hangs forever
instead. Delete the echo and the magic tests fail; delete the substitution and
the go-message tests hang the host (here, fail).
"""

import pytest

from tt_sim.bridge.cores import DeferredTensixCore, NullCore

COORD = (2, 1)

#: ``mailboxes_t.dprint_buf`` lives in the low L1 mailbox region
#: (``MEM_MAILBOX_BASE`` 16, ``MEM_MAILBOX_SIZE`` 13168); the exact offset is a
#: tt-metal release detail, so any address in the region will do.
DPRINT_ADDR = 0x1B00
#: ``DEBUG_PRINT_SERVER_STARTING_MAGIC`` / ``_DISABLED_MAGIC``, little-endian,
#: from ``hostdevcommon/dprint_common.h``.
STARTING_MAGIC = bytes([0x98, 0x98, 0x98, 0x98])
DISABLED_MAGIC = bytes([0xF8, 0xF8, 0xF8, 0xF8])
#: The host writes the whole ``DevicePrintMemoryLayout`` (204 B per processor,
#: 5 processors on a Wormhole Tensix) and then reads back only its first word.
DPRINT_STRUCT_SIZE = 204 * 5

GO_ADDR = 0x4A0
GO_INIT = b"\x00\x00\x00\x40"
GO_GO = b"\x00\x00\x00\x80"
FIRMWARE_ADDR = 0x1000
FIRMWARE = bytes(range(32, 64))


def _deferred(on_materialise=None):
    return DeferredTensixCore(COORD, on_materialise or (lambda coord: None))


def _null():
    return NullCore(COORD)


@pytest.fixture(params=["deferred", "null"])
def stub(request):
    """Both stand-ins, because a pinned worker set uses ``NullCore`` instead."""
    return _deferred() if request.param == "deferred" else _null()


# ---------------------------------------------------------------------------
# The bug: the host must be able to read back what it wrote.
# ---------------------------------------------------------------------------


def test_the_dprint_init_magic_reads_back(stub):
    """``WriteInitMagic``'s spin, exactly: write the struct, read word 0."""
    initbuf = STARTING_MAGIC + bytes(DPRINT_STRUCT_SIZE - len(STARTING_MAGIC))
    stub.write(DPRINT_ADDR, initbuf)
    assert stub.read(DPRINT_ADDR, 4) == STARTING_MAGIC


def test_the_dprint_disabled_magic_reads_back(stub):
    """``init_device`` disables prints on *every* core before attaching any."""
    stub.write(DPRINT_ADDR, DISABLED_MAGIC + bytes(DPRINT_STRUCT_SIZE - 4))
    assert stub.read(DPRINT_ADDR, 4) == DISABLED_MAGIC


def test_the_last_write_to_an_address_wins(stub):
    stub.write(DPRINT_ADDR, DISABLED_MAGIC)
    stub.write(DPRINT_ADDR, STARTING_MAGIC)
    assert stub.read(DPRINT_ADDR, 4) == STARTING_MAGIC


def test_a_read_spanning_written_and_untouched_bytes(stub):
    """Zero-fill survives where nothing was written — that is real L1."""
    stub.write(FIRMWARE_ADDR, FIRMWARE)
    got = stub.read(FIRMWARE_ADDR - 4, len(FIRMWARE) + 8)
    assert got == bytes(4) + FIRMWARE + bytes(4)


def test_a_write_crossing_the_shadow_page_boundary(stub):
    """The shadow is paged; a write across the seam must still read back."""
    payload = bytes(range(256)) * 32  # 8 KB, spans three 4 KB pages
    addr = 0x3F00
    stub.write(addr, payload)
    assert stub.read(addr, len(payload)) == payload
    assert stub.read(addr + 4096, 16) == payload[4096:4112]


def test_an_address_never_written_reads_zero(stub):
    assert stub.read(0x40000, 32) == bytes(32)


# ---------------------------------------------------------------------------
# The exception: the go message must never echo.
# ---------------------------------------------------------------------------


def test_the_go_message_reads_done_after_an_init_write(stub):
    """``wait_until_cores_done`` polls this; a stand-in has no firmware.

    Echoing ``go=INIT`` back would hang the host forever, which is a worse
    failure than the one being fixed.
    """
    stub.write(GO_ADDR, GO_INIT)
    assert stub.read(GO_ADDR, 4)[3] == 0x00


def test_the_go_messages_other_bytes_are_still_echoed(stub):
    """Only the signal byte is substituted — the rest is the host's own."""
    stub.write(GO_ADDR, b"\x11\x22\x33\x40")
    assert stub.read(GO_ADDR, 4) == b"\x11\x22\x33\x00"


def test_a_null_cores_go_message_reads_done_after_a_launch():
    """A pinned-out worker never runs, so even ``go=GO`` must read ``DONE``."""
    core = _null()
    core.write(GO_ADDR, GO_GO)
    assert core.read(GO_ADDR, 4)[3] == 0x00


# ---------------------------------------------------------------------------
# The shadow must not leak into replay.
# ---------------------------------------------------------------------------


class _RecordingDevice:
    """Just enough ``Device`` for ``DeferredTensixCore.replay``."""

    def __init__(self):
        self.writes = []

    def write_without_pump(self, unified, addr, data):
        self.writes.append((addr, bytes(data)))

    def deassert_reset_without_pump(self, unified):
        pass

    def assert_reset(self, unified):
        pass


def test_replay_hands_the_tile_the_run_state_the_host_actually_sent():
    """The journal is verbatim; only the *answer* to the host is doctored.

    ``LazyTensixPool.materialise`` needs ``replay`` to report the pending go
    address so the firmware can be run forward through the init handshake it
    slept through. Shadowing the journal too would write ``RUN_MSG_DONE`` into
    L1 and report nothing pending, and the worker would never run its init.
    """
    core = _deferred()
    device = _RecordingDevice()
    core.write(FIRMWARE_ADDR, FIRMWARE)
    core.write(GO_ADDR, GO_INIT)

    pending = core.replay(device, (18, 16))

    assert pending == GO_ADDR
    assert device.writes == [(FIRMWARE_ADDR, FIRMWARE), (GO_ADDR, GO_INIT)]


def test_a_write_out_of_reset_materialises_rather_than_shadowing():
    """The trigger is unchanged: the shadow must not swallow the signal."""
    built = []

    class _Real:
        def write(self, addr, data):
            built.append((addr, bytes(data)))

    core = _deferred(lambda coord: built.append(coord) or _Real())
    core.deassert_reset()
    core.write(FIRMWARE_ADDR, FIRMWARE)

    assert built == [COORD, (FIRMWARE_ADDR, FIRMWARE)]
    assert core.journal == [("d", None, None)]
