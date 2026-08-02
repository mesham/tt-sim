"""Tests for NoC address-alignment checking.

Runs standalone (``python3 -m tt_sim.network.alignment_test``) or under pytest.

Covers the per-arch congruence values, that a correctly aligned transfer passes,
that a misaligned one raises with an actionable message, and that
``TT_SIM_DISABLE_ALIGNMENT_CHECKS`` turns the checking off.

The rules under test are *congruence* rules — ``(src % n) == (dst % n)`` — not
absolute-alignment rules. That distinction is load-bearing: the length-mode table
in ``WormholeB0/NoC/Alignment.md`` contains only ``C4``/``C16``/``C32`` codes for
the L1 and DRAM cells, and the vendor reference simulator (``ttsim``
``src/tile.cpp``) likewise only compares low bits. An absolute-alignment check
would be stricter than hardware and would fire on correct kernels, so the tests
below deliberately pin that e.g. ``0x1004 -> 0x2004`` is *accepted*.
"""

import os

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.arch.wormhole import WORMHOLE_PROFILE
from tt_sim.network.alignment import (
    DISABLE_ENV_VAR,
    L1_CONGRUENCE,
    MMIO_BASE,
    MMIO_CONGRUENCE,
    NoCAlignmentError,
    check_congruence,
    checking_enabled,
    congruence_for_read,
    refresh_from_env,
    set_checking_enabled,
)


def test_per_arch_congruence_values():
    """Wormhole reads DRAM with a 32 B byte-enable span, Blackhole with 64 B."""
    assert WORMHOLE_PROFILE.noc_dram_read_congruence == 32
    assert BLACKHOLE_PROFILE.noc_dram_read_congruence == 64
    # Both must flow to the NUI through the profile's NoC kwargs, not a default.
    assert WORMHOLE_PROFILE.noc_kwargs["noc_dram_read_congruence"] == 32
    assert BLACKHOLE_PROFILE.noc_kwargs["noc_dram_read_congruence"] == 64


def test_congruence_selection_by_source_kind():
    """A read picks its modulus from what it reads *from*."""
    for profile, dram_mod in ((WORMHOLE_PROFILE, 32), (BLACKHOLE_PROFILE, 64)):
        m = profile.noc_dram_read_congruence
        # DRAM source -> the arch's DRAM modulus.
        assert congruence_for_read(0x1000, True, m) == dram_mod
        # L1 source -> C16, on both arches.
        assert congruence_for_read(0x1000, False, m) == L1_CONGRUENCE
        # MMIO source -> C4.
        assert congruence_for_read(MMIO_BASE + 0x20000, False, m) == MMIO_CONGRUENCE


def test_aligned_transfers_pass():
    # Identical low bits: the canonical aligned case.
    check_congruence(16, 0x1000, 0x2000, path="L1 -> L1 write")
    check_congruence(32, 0x4000, 0x8000, path="DRAM -> L1 read")
    check_congruence(64, 0x4000, 0x8000, path="DRAM -> L1 read")
    # Congruent but NOT absolutely aligned — hardware accepts this, so we must
    # too. An absolute-alignment check would wrongly reject these.
    check_congruence(16, 0x1004, 0x2004, path="L1 -> L1 write")
    check_congruence(32, 0x100C, 0x200C, path="DRAM -> L1 read")
    check_congruence(4, 0xFFB20000 + 2, 0x1002, path="MMIO -> L1 read")


def test_misaligned_transfer_raises_actionable_message():
    with pytest.raises(NoCAlignmentError) as excinfo:
        check_congruence(
            32,
            0x1004,
            0x2000,
            path="DRAM -> L1 read",
            noc_number=0,
            initiator=(1, 1),
        )
    msg = str(excinfo.value)
    # Names the access path, both addresses, and the required alignment.
    assert "DRAM -> L1 read" in msg
    assert "NoC0 (1, 1)" in msg
    assert "0x1004" in msg
    assert "0x2000" in msg
    assert "congruent modulo 32" in msg
    # Shows the actual remainders so the fix is obvious.
    assert "% 32 = 4" in msg
    assert "% 32 = 0" in msg
    # Tells the user how to turn the check off.
    assert DISABLE_ENV_VAR in msg


def test_wormhole_dram_read_accepted_but_blackhole_rejected():
    """32 B-congruent but not 64 B-congruent: legal on Wormhole, not Blackhole.

    This is the case the per-arch profile value exists to distinguish.
    """
    src, dst = 0x1020, 0x2000
    assert src % 32 == dst % 32
    assert src % 64 != dst % 64
    check_congruence(
        WORMHOLE_PROFILE.noc_dram_read_congruence, src, dst, path="DRAM -> L1 read"
    )
    with pytest.raises(NoCAlignmentError):
        check_congruence(
            BLACKHOLE_PROFILE.noc_dram_read_congruence, src, dst, path="DRAM -> L1 read"
        )


def test_env_var_disables_checking():
    original = os.environ.get(DISABLE_ENV_VAR)
    try:
        for truthy in ("1", "true", "YES", "On"):
            os.environ[DISABLE_ENV_VAR] = truthy
            assert refresh_from_env() is False, truthy
            assert not checking_enabled()
            # The offending transfer now sails through.
            check_congruence(32, 0x1004, 0x2000, path="DRAM -> L1 read")

        for falsy in ("0", "false", "no", "", "off"):
            os.environ[DISABLE_ENV_VAR] = falsy
            assert refresh_from_env() is True, falsy
            assert checking_enabled()
            with pytest.raises(NoCAlignmentError):
                check_congruence(32, 0x1004, 0x2000, path="DRAM -> L1 read")

        # Unset means checking is on: it must default ON, not off.
        del os.environ[DISABLE_ENV_VAR]
        assert refresh_from_env() is True
        assert checking_enabled()
    finally:
        if original is None:
            os.environ.pop(DISABLE_ENV_VAR, None)
        else:
            os.environ[DISABLE_ENV_VAR] = original
        refresh_from_env()


def test_programmatic_toggle_restores():
    assert checking_enabled()
    try:
        set_checking_enabled(False)
        check_congruence(16, 0x1004, 0x2000, path="L1 -> L1 write")
    finally:
        set_checking_enabled(True)
    assert checking_enabled()


def test_noc_read_and_write_paths_are_checked():
    """End-to-end through a real device: the NUI rejects a misaligned DRAM read.

    Exercises the wiring (profile -> NUI -> RequestInitiator), not just the
    predicate, and confirms the DRAM tile is tagged so the 32 B rule is picked.
    """
    from tt_sim.device.wormhole import Wormhole

    device = Wormhole()
    tensix = device.tensix_tiles[0]
    dram = device.dram_tiles[0]
    assert dram.noc0_router.tile_kind == "D"
    assert tensix.noc0_router.tile_kind == "T"
    assert tensix.noc0_router.noc_dram_read_congruence == 32

    initiator = tensix.noc0_router.request_initiators[0]
    dram_coord = dram.noc0_router.id_pair
    initiator.target_addr_mid = (dram_coord[0] << 4) | (dram_coord[1] << 10)

    # A DRAM read that IS congruent modulo 16 but NOT modulo 32. It would pass
    # the L1 rule, so rejecting it proves the DRAM tile was resolved and the
    # 32 B DRAM rule — not L1's 16 B — was applied end to end.
    initiator.target_addr_low = 0x1010
    initiator.ret_addr_low = 0x2000
    assert 0x1010 % 16 == 0x2000 % 16
    assert 0x1010 % 32 != 0x2000 % 32
    initiator.at_len_be = 64
    initiator.ctrl = 0  # mode 0 = read
    initiator.cmd_ctrl = 1
    with pytest.raises(NoCAlignmentError, match="DRAM -> L1 read.*modulo 32"):
        initiator.initiate()

    # The same read with matching low bits is accepted.
    initiator.ret_addr_low = 0x2010
    initiator.cmd_ctrl = 1
    initiator.initiate()

    # A write out of L1 uses C16 regardless of destination kind.
    initiator.target_addr_low = 0x1000
    initiator.ret_addr_low = 0x2008
    initiator.ctrl = 2  # mode 2 = write
    initiator.cmd_ctrl = 1
    with pytest.raises(NoCAlignmentError, match="write"):
        initiator.initiate()


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "alignment_test OK: per-arch congruence (WH 32 / BH 64), congruent "
        "transfers accepted, misaligned rejected with an actionable message, "
        f"{DISABLE_ENV_VAR} disables checking"
    )


if __name__ == "__main__":
    main()
