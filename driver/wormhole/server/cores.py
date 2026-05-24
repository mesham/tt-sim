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
    list in ``TT_SIM_TENSIX_COORDS``. The optional
    ``on_user_data_write`` hook fires the first time a write targets
    L1 above 0x10000 — that's past the kernel firmware / init scratch
    region, so traffic landing there means tt-metal is treating this
    core as a real worker. ``__main__.py`` uses this to warn about the
    most common silent-config bug.
    """

    USER_DATA_ADDR_THRESHOLD = 0x10000

    def __init__(self, coord, on_user_data_write=None):
        self.coord = coord
        self.in_reset = True
        self._on_user_data_write = on_user_data_write
        self._user_write_seen = False

    def write(self, addr, data):
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
