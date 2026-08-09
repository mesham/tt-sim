"""NIU register reads that exist to be *measured*, not modelled.

``CMD_BUF_AVAIL`` is the one register in the NoC block whose whole value to
this project is that nobody knows what it says. Blackhole's
``noc_parameters.h`` puts it at ``NOC_REGS_START_ADDR + 0x64`` and comments it
``[28:24], [20:16], [12:8], [4:0]`` — four 5-bit per-command-buffer occupancy
fields — while the ISA docs call ``0x0064`` the "NIU request FIFO status" and
give **no depth**. tt-metal defines the address and never references it, and
Wormhole has no analogue at all. Reading it at rest on silicon is a roadmap
item (`perfbench/nocreadbench` records it) precisely because a clean small
integer there is a number the documentation withholds.

``CMD_BUF_OVFL`` at ``0x68`` is its neighbour and is here for the same reason.
It is the only reading that can turn a measured peak occupancy into a *depth*
rather than a lower bound on one, so ``nocreadbench`` reads it at rest and at
the end of its sampled burst. Both addresses are covered by every test below.

That makes the simulator's obligation an unusual one. This NUI models no
command buffer, so there is no honest value to return — and a plausible small
integer would be *worse* than useless, because it would be indistinguishable
from the measurement the exercise exists to obtain. The register therefore
reads as the all-ones sentinel ``nocreadbench`` already uses for "this part
does not expose the register", so a simulator run reads as **absent** rather
than as a reading.

Before this, ``0x64`` raised ``NotImplementedError``, which meant the Blackhole
half of ``nocreadbench`` — the half that carries the ``CMD_BUF_AVAIL`` bullet —
could not be smoke-tested against tt-sim at all. ``0x68`` raised the same thing
until the probe started reading it.

Runs standalone (``python3 -m tt_sim.network.noc_registers_test``) or under
pytest.
"""

from tt_sim.network.tt_noc import NUI
from tt_sim.util.conversion import conv_to_uint32

CMD_BUF_AVAIL_OFFSET = 0x64
CMD_BUF_OVFL_OFFSET = 0x68
OFFSETS = (CMD_BUF_AVAIL_OFFSET, CMD_BUF_OVFL_OFFSET)
ABSENT = 0xFFFFFFFF


def _nui():
    return NUI(0, 1, 1, None)


def test_cmd_buf_avail_reads_as_absent_rather_than_raising():
    """The point of the change: it answers instead of aborting the run."""
    for offset in OFFSETS:
        assert conv_to_uint32(_nui().read(offset, 4)) == ABSENT


def test_the_sentinel_is_the_one_nocreadbench_already_treats_as_absent():
    """``perfbench/nocreadbench``'s kernel returns ``0xFFFFFFFF`` from its own
    ``#else`` branch when ``CMD_BUF_AVAIL`` is undefined, and its card-side
    sanity check reads "on Blackhole, cmdbuf_avail_rest is NOT 0xFFFFFFFF". A
    simulator run must land on that same value, or the harness would report a
    simulated null as a Blackhole reading."""
    for offset in OFFSETS:
        value = conv_to_uint32(_nui().read(offset, 4))
        assert value == ABSENT
        # Not a legal four-5-bit-field value, so it can never be mistaken for one.
        assert value > 0x1F1F1F1F


def test_it_is_not_a_read_what_you_wrote_register():
    """The neighbouring ``NOC_CFG`` block is generic read-back storage. This one
    must not be, or a stray write would turn the sentinel into a number that
    looks like a measurement. Only the read side was added, so a write still
    refuses — which is right for a status register, and keeps the sentinel the
    only value the address can ever produce."""
    for offset in OFFSETS:
        nui = _nui()
        try:
            nui.write(offset, 4)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"writing {hex(offset)} should still refuse")
        assert conv_to_uint32(nui.read(offset, 4)) == ABSENT


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("noc_registers_test OK")


if __name__ == "__main__":
    main()
