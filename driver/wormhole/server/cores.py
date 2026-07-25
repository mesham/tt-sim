"""Core implementations used by the fabric.

``NullCore`` swallows writes and returns zeros (proven enough for tt-metal
device init). ``TensixCore`` and ``DramCore`` are thin shims over ``Device``
that route the wire's translated NoC coords to the corresponding tt-sim
unified coord.

The ``--mock-tensix`` CLI flag in ``__main__.py`` substitutes ``NullCore`` for
``TensixCore``/``DramCore`` registration — useful for diffing against the
phase-1 zero-stub behaviour without rebuilding the simulator.
"""


class NullCore:
    """Accepts all writes silently, returns zeros for all reads.

    Used both for genuinely unmodelled tiles (eth / pcie / arc /
    router-only) and as the fallback for worker coords the user didn't
    list in ``TT_SIM_TENSIX_COORDS``. Two optional hooks let
    ``__main__.py`` surface config bugs:

    - ``on_user_data_write`` fires the first time a write targets L1
      above 0x10000 — past the kernel firmware / init scratch region, so
      traffic landing there means tt-metal is treating this core as a
      real worker (a warning-level hint).
    - ``on_kernel_launch`` fires when the host writes go=GO (a kernel
      launch) to this coord. Unlike the grid-wide go=INIT handshake that
      touches every worker during device init, a go=GO only ever targets
      cores a program actually runs on — so a go=GO to an un-materialised
      worker is a hard error: the program needs more cores than tt-sim
      was started with.
    """

    USER_DATA_ADDR_THRESHOLD = 0x10000
    # go_msg_t is a 4-byte union; ``signal`` is the top byte (offset 3).
    # RUN_MSG_GO means "run the kernel here" (tt_metal/hw/inc/hostdev/dev_msgs.h);
    # RUN_MSG_INIT (0x40) is the grid-wide init handshake and is NOT a launch.
    _GO_MSG_SIZE = 4
    _RUN_MSG_GO = 0x80

    def __init__(self, coord, on_user_data_write=None, on_kernel_launch=None):
        self.coord = coord
        self.in_reset = True
        self._on_user_data_write = on_user_data_write
        self._on_kernel_launch = on_kernel_launch
        self._user_write_seen = False

    def write(self, addr, data):
        if (
            self._on_kernel_launch is not None
            and len(data) == NullCore._GO_MSG_SIZE
            and data[NullCore._GO_MSG_SIZE - 1] == NullCore._RUN_MSG_GO
        ):
            # Raises (see __main__) — a launch on an un-materialised worker.
            self._on_kernel_launch(self.coord)
        if (
            not self._user_write_seen
            and self._on_user_data_write is not None
            and addr >= NullCore.USER_DATA_ADDR_THRESHOLD
        ):
            self._user_write_seen = True
            self._on_user_data_write(self.coord)

    def read(self, addr, size):
        return b"\x00" * size

    def assert_reset(self):
        self.in_reset = True

    def deassert_reset(self):
        self.in_reset = False


class TensixCore:
    """Routes wire messages for a Tensix coord into the tt-sim Wormhole."""

    def __init__(self, device, unified_coord):
        self.device = device
        self.unified = unified_coord
        self.in_reset = True
        device.register_tensix(unified_coord)

    def write(self, addr, data):
        self.device.write(self.unified, addr, data)

    def read(self, addr, size):
        return self.device.read(self.unified, addr, size)

    def assert_reset(self):
        self.in_reset = True
        self.device.assert_reset(self.unified)

    def deassert_reset(self):
        # TODO(future): honour the launch_msg ``enables`` field to fan out to
        # NCRISC/TRISC0/1/2. tt-metal "one" uses enables=0x1 (BRISC only), so
        # BRISC-only is correct for the immediate target. The launch message
        # arrives via a WRITE to the launch_msg L1 offset (0x20 for "one"); a
        # future hook in ``write()`` can snapshot the enables field and apply
        # it here when the host then sends DEASSERT.
        self.in_reset = False
        self.device.deassert_reset_brisc(self.unified)


class EthCore:
    """Routes wire messages for an eth coord into the tt-sim Wormhole.

    Eth tiles in tt-sim today are L1 SRAM only — no ERisc CPU — so reset
    assert/deassert are no-ops (there is no core to gate). Reads and writes
    land in the eth tile's 256 KB L1; this replaces the previous NullCore
    zero-fill behaviour so kernels that hardcode an eth coord
    (e.g. ``hello_world_datatypes_kernel`` reading ``(1, 0)``) get
    deterministic memory-backed state.

    Go-message completion. tt-metal's ``LaunchProgram``/device init writes a
    ``go_msg_t`` to every core (workers *and* eth) with the ``signal`` byte
    set to a launch run-state (INIT/GO/...), then spins in
    ``wait_until_cores_done`` until it reads back ``RUN_MSG_DONE`` (signal
    byte 0). Real eth cores run base firmware that flips the byte; tt-sim has
    no ERisc, so with plain memory-backed L1 the signal would stay at its
    launched value and the host would hang forever. (Unmodelled *worker*
    coords dodge this only because ``NullCore`` zero-fills, and 0 happens to
    equal ``RUN_MSG_DONE``.) We model the missing firmware minimally: a
    go-message write is stored with its signal byte forced to
    ``RUN_MSG_DONE`` — i.e. the eth "kernel" is treated as having completed
    instantly. Everything else lands in L1 untouched.
    """

    # go_msg_t is a 4-byte union; ``signal`` is the top byte (offset 3).
    # Launch run-states from tt_metal/hw/inc/hostdev/dev_msgs.h — anything in
    # this set is a signal the host will then poll to RUN_MSG_DONE (0x0).
    _GO_MSG_SIZE = 4
    _RUN_MSG_DONE = 0x00
    _LAUNCH_SIGNALS = frozenset({0x40, 0x80, 0xC0, 0xE0, 0xF0})

    def __init__(self, device, unified_coord):
        self.device = device
        self.unified = unified_coord

    def write(self, addr, data):
        if (
            len(data) == self._GO_MSG_SIZE
            and data[self._GO_MSG_SIZE - 1] in self._LAUNCH_SIGNALS
        ):
            data = data[: self._GO_MSG_SIZE - 1] + bytes([self._RUN_MSG_DONE])
        self.device.write(self.unified, addr, data)

    def read(self, addr, size):
        return self.device.read(self.unified, addr, size)

    def assert_reset(self):
        pass

    def deassert_reset(self):
        pass


class DramCore:
    """Routes wire messages for a DRAM coord into the tt-sim Wormhole.

    DRAM has no reset — assert/deassert are no-ops. Reads/writes still pump
    the device so BRISC progresses on background traffic to DRAM.

    All 12 DRAM sub-endpoints from the Wormhole B0 SoC descriptor are
    registered (see ``coords.py``), backed by 6 ``DRAMTile`` instances in
    ``Wormhole.__init__`` — one per physical controller.
    """

    def __init__(self, device, unified_coord):
        self.device = device
        self.unified = unified_coord

    def write(self, addr, data):
        self.device.write(self.unified, addr, data)

    def read(self, addr, size):
        return self.device.read(self.unified, addr, size)

    def assert_reset(self):
        pass

    def deassert_reset(self):
        pass
