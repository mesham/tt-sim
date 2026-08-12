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


class DeferredTensixCore:
    """A functional worker the simulator has not built a tile for *yet*.

    On the wire it is indistinguishable from :class:`NullCore` — writes are
    swallowed, reads return zeros — which is exactly what tt-metal's grid-wide
    init handshake needs: it writes ``go=INIT`` to every declared worker and
    then polls until the mailbox reads back ``RUN_MSG_DONE`` (0), so a
    zero-filled read is what lets init complete without a tile existing. Serving
    reads out of the journal instead would echo ``INIT`` straight back and hang
    the host.

    What it adds is the *journal*: every write and every reset transition it
    sees **while the core is still held in reset**, in arrival order, so when
    the tile is stood up (:meth:`replay`) it reaches the state an eagerly
    materialised worker would have had at the moment it was released. Ordering
    is the whole point — the launch message's ``enables`` bitmask has to be in
    L1 before the DEASSERT that reads it, or the kernel's NCRISC/TRISCs never
    leave reset.

    **The journal stops at the DEASSERT, and that is load-bearing.** Once the
    host has released a core it expects that core to be *executing*: tt-metal
    writes ``go=INIT`` before the DEASSERT and then polls every worker to
    ``RUN_MSG_DONE`` before it writes a single kernel binary. Journalling past
    that point and replaying it in one go would hand BRISC an L1 in which the
    second launch message has already overwritten the first, and it would run
    the INIT run-state against it — measured, that is a wild store a third of a
    megabyte past the top of L1, on every multi-core program. So a write that
    arrives while the core is out of reset **materialises immediately**, which
    is also a precise "this worker is used" signal in its own right: the host
    only writes kernel binaries to cores the program runs on.

    Three triggers, then, all of which end with a real tile that has caught up:

    1. a write while out of reset — the binaries a program's core receives;
    2. ``go=GO`` — a launch, for a core that somehow received no binaries;
    3. a NoC directory miss — a peer addressing it first (see
       ``TT_Device.set_directory_miss_hook``).

    Whichever fires first wins; the others find the tile already there.
    """

    _GO_MSG_SIZE = 4
    _RUN_MSG_GO = 0x80
    #: Run-states the host writes and then polls to ``RUN_MSG_DONE``. ``GO`` is
    #: excluded because it can never be journalled — it is a trigger.
    _HANDSHAKE_SIGNALS = frozenset({0x40, 0xC0, 0xE0, 0xF0})

    def __init__(self, coord, materialise):
        self.coord = coord
        self.in_reset = True
        self._materialise = materialise
        #: ``(kind, addr, data)`` in arrival order; ``kind`` is ``"w"`` for a
        #: write, ``"a"``/``"d"`` for a reset assert/deassert.
        self.journal = []

    def write(self, addr, data):
        if self.in_reset and not (
            len(data) == DeferredTensixCore._GO_MSG_SIZE
            and data[DeferredTensixCore._GO_MSG_SIZE - 1]
            == DeferredTensixCore._RUN_MSG_GO
        ):
            self.journal.append(("w", addr, bytes(data)))
            return
        # Either a kernel launch or the host talking to a core it has already
        # released — both say "this worker is used". Build it, give it the
        # journal, let its firmware finish the init handshake it slept through,
        # and only then apply this write, which is the host's own order.
        self._materialise(self.coord).write(addr, data)

    def read(self, addr, size):
        return b"\x00" * size

    def assert_reset(self):
        self.in_reset = True
        self.journal.append(("a", None, None))

    def deassert_reset(self):
        self.in_reset = False
        self.journal.append(("d", None, None))

    def replay(self, device, unified):
        """Apply the journal to a now-existing tile, oldest first.

        Returns the address of the go-message the firmware still owes a
        ``RUN_MSG_DONE`` for, or ``None``. The *host* was answered ``DONE`` out
        of the zero-fill long ago and has moved on; what has not happened is
        the firmware executing that run-state, and everything after it depends
        on that having run. The caller settles it — see
        :meth:`LazyTensixPool.materialise`.

        Uses the *non-pumping* device entry points throughout. Replay can be
        driven from inside ``tt_device.run`` (a peer's packet arriving at this
        coord), and pumping there would re-enter the clock mid-cycle.
        """
        pending_go = None
        for kind, addr, data in self.journal:
            if kind == "w":
                device.write_without_pump(unified, addr, data)
                if (
                    len(data) == DeferredTensixCore._GO_MSG_SIZE
                    and data[DeferredTensixCore._GO_MSG_SIZE - 1]
                    in DeferredTensixCore._HANDSHAKE_SIGNALS
                ):
                    pending_go = addr
            elif kind == "d":
                device.deassert_reset_without_pump(unified)
            else:
                device.assert_reset(unified)
        self.journal.clear()
        return pending_go


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
        # A wire DEASSERT releases the master BRISC plus whichever subordinate
        # RISCs (NCRISC / TRISC0-2) the launch message enabled — see
        # ``Device.deassert_reset``. This is what lets a program with a writer on
        # NCRISC (RISCV_1) or compute on the TRISCs actually run those cores.
        self.in_reset = False
        self.device.deassert_reset(self.unified)


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
