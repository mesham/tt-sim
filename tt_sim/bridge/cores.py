"""Core implementations used by the fabric.

``NullCore`` and ``DeferredTensixCore`` are stand-ins for a tile the simulator
has not built: they swallow writes and answer reads out of a ``_WriteShadow``
— the host's own bytes back, zeros for anything untouched, ``RUN_MSG_DONE``
for the go message. ``TensixCore`` and ``DramCore`` are thin shims over
``Device`` that route the wire's translated NoC coords to the corresponding
tt-sim unified coord.

The ``--mock-tensix`` CLI flag in ``__main__.py`` substitutes ``NullCore`` for
``TensixCore``/``DramCore`` registration — a device that remembers but does not
execute, for wire-level debugging without building the simulator. (It was the
phase-1 zero stub before the shadow; a mock read now returns the host's own
bytes rather than zeros.)
"""


class _WriteShadow:
    """The bytes a stand-in core has been told, so it can be asked back.

    A core the simulator has not built a tile for still has to answer host
    reads, and the answer used to be an unconditional zero-fill. That is wrong
    for any address the host **wrote itself and then reads back**, and tt-metal
    has exactly such a handshake: ``DPrintServer`` writes a magic word into
    every core's ``dprint_buf`` and spins 100000 times waiting to read it back
    (``dprint_server.cpp:WriteInitMagic``). Against a zero-filling stand-in
    that spin can never succeed, so *any* run with DPRINT enabled aborted with
    ``TT_THROW: Timed out writing init magic`` — DPRINT, and everything built on
    it (the LLK sanitizer, watcher-style debugging), was unusable on tt-sim.

    So writes are shadowed, sparsely, in 4 KB pages — only the pages actually
    written cost anything — and reads are served out of the shadow, zero-filled
    where nothing has been written. That is what real L1 would do.

    **The one address that must not echo is the go message.** tt-metal writes a
    launch run-state (``go=INIT`` and friends) and then polls until the core's
    firmware writes ``RUN_MSG_DONE`` back. A stand-in has no firmware, so
    echoing the run-state hangs the host forever; the old zero-fill "worked"
    only because ``RUN_MSG_DONE`` is 0. :meth:`shadow_write` therefore stores a
    handshake go-message with its signal byte already set to ``RUN_MSG_DONE``,
    which says the same thing the zero-fill did but says it deliberately, and
    keeps saying it once the neighbouring bytes are real. The journal that
    drives replay is untouched — see :meth:`DeferredTensixCore.replay`, which
    still has to see the run-state the host actually sent.
    """

    _PAGE_BITS = 12
    _PAGE = 1 << _PAGE_BITS
    #: go_msg_t is a 4-byte union; ``signal`` is the top byte (offset 3).
    _GO_MSG_SIZE = 4
    _RUN_MSG_DONE = 0x00
    #: Run-states the host writes and then polls to ``RUN_MSG_DONE``. Includes
    #: ``GO`` (0x80) for the benefit of a stand-in that never becomes a real
    #: core: for those it is not a trigger, just another poll to answer.
    _POLLED_SIGNALS = frozenset({0x40, 0x80, 0xC0, 0xE0, 0xF0})

    def __init__(self):
        #: ``page index -> 4 KB bytearray``, allocated on first touch.
        self._pages: dict[int, bytearray] = {}

    def shadow_write(self, addr, data):
        if (
            len(data) == _WriteShadow._GO_MSG_SIZE
            and data[_WriteShadow._GO_MSG_SIZE - 1] in _WriteShadow._POLLED_SIGNALS
        ):
            data = bytes(data[: _WriteShadow._GO_MSG_SIZE - 1]) + bytes(
                [_WriteShadow._RUN_MSG_DONE]
            )
        end, pos = addr + len(data), 0
        while addr < end:
            page, off = divmod(addr, _WriteShadow._PAGE)
            n = min(_WriteShadow._PAGE - off, end - addr)
            buf = self._pages.get(page)
            if buf is None:
                buf = self._pages[page] = bytearray(_WriteShadow._PAGE)
            buf[off : off + n] = data[pos : pos + n]
            addr += n
            pos += n

    def shadow_read(self, addr, size):
        if not self._pages:
            return b"\x00" * size
        out, end, pos = bytearray(size), addr + size, 0
        while addr < end:
            page, off = divmod(addr, _WriteShadow._PAGE)
            n = min(_WriteShadow._PAGE - off, end - addr)
            buf = self._pages.get(page)
            if buf is not None:
                out[pos : pos + n] = buf[off : off + n]
            addr += n
            pos += n
        return bytes(out)


class NullCore(_WriteShadow):
    """Accepts all writes silently, answers reads out of its write shadow.

    Used both for genuinely unmodelled tiles (eth / pcie / arc /
    router-only) and as the fallback for worker coords the user didn't
    list in ``TT_SIM_TENSIX_COORDS``. Reads used to be an unconditional
    zero-fill; they are now :class:`_WriteShadow`'s echo of what the host
    wrote, which is what makes the DPRINT init handshake terminate on a
    pinned worker set. Untouched addresses still read zero, and the go
    message still reads ``RUN_MSG_DONE``, so the init handshake this stub
    was built for is unchanged.

    Two optional hooks let ``__main__.py`` surface config bugs:

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
        super().__init__()
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
        self.shadow_write(addr, data)

    def read(self, addr, size):
        return self.shadow_read(addr, size)

    def assert_reset(self):
        self.in_reset = True

    def deassert_reset(self):
        self.in_reset = False


class DeferredTensixCore(_WriteShadow):
    """A functional worker the simulator has not built a tile for *yet*.

    On the wire it is indistinguishable from :class:`NullCore`: writes are
    swallowed and reads answered out of the :class:`_WriteShadow` — the host's
    own bytes back, zeros where it has written nothing, and ``RUN_MSG_DONE``
    for the go message however the host last set it. That last exception is
    what tt-metal's grid-wide init handshake needs (it writes ``go=INIT`` to
    every declared worker and then polls the mailbox to ``RUN_MSG_DONE``), and
    it is the *only* address that must not echo — echoing the rest is what lets
    DPRINT's ``WriteInitMagic`` readback terminate.

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
        super().__init__()
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
            # The journal keeps the host's bytes verbatim — ``replay`` has to
            # see the run-state actually sent. The shadow is what the host is
            # answered from, and it substitutes ``RUN_MSG_DONE``.
            self.journal.append(("w", addr, bytes(data)))
            self.shadow_write(addr, data)
            return
        # Either a kernel launch or the host talking to a core it has already
        # released — both say "this worker is used". Build it, give it the
        # journal, let its firmware finish the init handshake it slept through,
        # and only then apply this write, which is the host's own order.
        self._materialise(self.coord).write(addr, data)

    def read(self, addr, size):
        return self.shadow_read(addr, size)

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
