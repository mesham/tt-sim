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
    """Accepts all writes silently, returns zeros for all reads."""

    def __init__(self, coord):
        self.coord = coord
        self.in_reset = True

    def write(self, addr, data):
        pass

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
