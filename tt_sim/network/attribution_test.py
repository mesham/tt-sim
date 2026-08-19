"""Tests for naming *which* NoC transfer failed an alignment check.

Two properties matter more than the feature itself, and both are pinned here:

1. **The set of programs that raise is unchanged.** Attribution runs only after
   something has already decided to raise, and it is wrapped so that a broken
   describer cannot turn a passing transfer into a failing one, a failing one
   into a passing one, or a :class:`NoCAlignmentError` into some other
   exception. ``test_a_broken_describer_...`` deliberately sabotages the
   describer and re-runs the whole accept/reject matrix.
2. **The existing message is not regressed.** The enriched message begins with
   exactly the bytes ``check_congruence`` produced — addresses, moduli,
   remainders, path, tile, NoC and the ``TT_SIM_DISABLE_ALIGNMENT_CHECKS``
   hint — with the new lines appended after it.

The end-to-end cases assemble a real RV32 program and let a real BRISC execute
it, because that is the only thing that proves the claim the feature rests on:
that the issuing core is still on the Python stack when the check fires. A test
that called ``initiate()`` directly would pass with the issuer lookup deleted.

Runs standalone (``python3 -m tt_sim.network.attribution_test``) or under
pytest.
"""

import shutil
import subprocess
import sys
import textwrap

import pytest

from tt_sim.network.alignment import NoCAlignmentError
from tt_sim.network.attribution import (
    Issuer,
    attach_provenance,
    describe_source,
    find_issuer,
    provenance,
    span,
)
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType

# --- a tiny RV32I assembler, enough to program the NoC command registers -----


def _lui(rd, imm20):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0x37


def _addi(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13


def _sw(rs2, rs1, off):
    imm = off & 0xFFF
    return (
        ((imm >> 5) & 0x7F) << 25
        | (rs2 << 20)
        | (rs1 << 15)
        | (0x2 << 12)
        | (imm & 0x1F) << 7
        | 0x23
    )


def _li(rd, value):
    """``lui``/``addi`` for an arbitrary 32-bit constant."""
    value &= 0xFFFFFFFF
    lo = value & 0xFFF
    signed_lo = lo - 0x1000 if lo & 0x800 else lo
    hi = (value - signed_lo) >> 12
    if not hi:
        return [_addi(rd, 0, signed_lo)]
    out = [_lui(rd, hi)]
    if lo:
        out.append(_addi(rd, rd, signed_lo))
    return out


#: NoC 0 register block as a baby core sees it, and the per-initiator offsets
#: used below (NOC_TARG_ADDR_LO, _MID, NOC_RET_ADDR_LO, NOC_CTRL,
#: NOC_AT_LEN_BE, NOC_CMD_CTRL).
_NOC0 = 0xFFB20000
_TARG_LO, _TARG_MID, _RET_LO, _CTRL, _AT_LEN, _CMD_CTRL = (
    0x0,
    0x4,
    0xC,
    0x1C,
    0x20,
    0x28,
)


def _noc_read_program(dram_mid, target_addr, ret_addr, size=64):
    """A program that issues one DRAM -> L1 NoC read, and the PC of the store
    to ``NOC_CMD_CTRL`` that starts it."""
    words = []
    words += _li(1, _NOC0)
    for value, offset in (
        (target_addr, _TARG_LO),
        (dram_mid, _TARG_MID),
        (ret_addr, _RET_LO),
        (0, _CTRL),  # mode 0 = read
        (size, _AT_LEN),
    ):
        words += _li(2, value)
        words.append(_sw(2, 1, offset))
    words += _li(2, 1)
    issue_pc = len(words) * 4
    words.append(_sw(2, 1, _CMD_CTRL))
    words.append(0x0000006F)  # j . -- park the core after issuing
    return b"".join(w.to_bytes(4, "little") for w in words), issue_pc


def _run_program(program, cycles=200):
    """Boot a one-Tensix Wormhole, run ``program`` on BRISC, return the device
    and whatever it raised."""
    from tt_sim.device.wormhole import Wormhole

    device = Wormhole()
    device.reset()
    coord = next(p for p, tile in device.tile_directory.items() if tile.is_tensix)
    device.write(coord, 0x0, program)
    device.deassert_soft_reset(coord, core_type=BabyRISCVCoreType.BRISC)
    try:
        device.run(cycles)
    except NoCAlignmentError as exc:
        return device, exc
    return device, None


def _dram_mid(device):
    coord = device.dram_tiles[0].noc0_router.id_pair
    return (coord[0] << 4) | (coord[1] << 10)


@pytest.fixture(scope="module")
def dram_mid():
    from tt_sim.device.wormhole import Wormhole

    return _dram_mid(Wormhole())


# --- 1. the issuing core and PC ---------------------------------------------


def test_the_issuing_core_and_pc_are_named(dram_mid):
    """A real core executing a real store is named, with the PC of that store.

    0x1010 and 0x2000 agree modulo 16 but not modulo 32, so this is rejected by
    the DRAM rule and would have passed the L1 one.
    """
    program, issue_pc = _noc_read_program(dram_mid, 0x1010, 0x2000)
    _device, exc = _run_program(program)
    assert exc is not None, "the misaligned DRAM read was not rejected"
    message = str(exc)
    assert f"Issued by: BRISC at PC={hex(issue_pc)}." in message, message
    # ...and the PC really is the store to NOC_CMD_CTRL, not an approximation.
    assert issue_pc == len(program) - 8


def test_the_transfer_itself_is_described(dram_mid):
    """Size, transaction id and both address spans, so the transfer is
    identifiable without re-deriving it from the two bare addresses."""
    program, _ = _noc_read_program(dram_mid, 0x1010, 0x2000, size=64)
    _device, exc = _run_program(program)
    message = str(exc)
    assert "Transfer: 64 bytes, transaction id 0" in message, message
    assert "source 0x1010..0x104f" in message, message
    assert "destination 0x2000..0x203f" in message, message


def test_no_issuer_is_invented_when_no_core_is_running():
    """A transfer driven straight from Python has no issuing core, and the
    message must say nothing rather than name whatever ran last."""
    from tt_sim.device.wormhole import Wormhole

    device = Wormhole()
    tensix = device.tensix_tiles[0]
    initiator = tensix.noc0_router.request_initiators[0]
    initiator.target_addr_mid = _dram_mid(device)
    initiator.target_addr_low = 0x1010
    initiator.ret_addr_low = 0x2000
    initiator.at_len_be = 64
    initiator.ctrl = 0
    initiator.cmd_ctrl = 1
    with pytest.raises(NoCAlignmentError) as caught:
        initiator.initiate()
    message = str(caught.value)
    assert "Issued by:" not in message, message
    # The transfer is still described -- that part needs no core.
    assert "Transfer: 64 bytes" in message, message


def test_find_issuer_returns_none_off_a_core_stack():
    assert find_issuer() is None


class _FakeRegister:
    def read_uint(self):
        return 0xABC


class _FakeCore:
    """The shape ``find_issuer`` recognises as an executing RV32 core."""

    core_label = "TRISC2"
    pc_register = _FakeRegister()
    visible_memory = None

    def probe(self, inner):
        return inner.probe()


class _FakeMemorySpace:
    """The shape that carries ``caller_context``, with a *stale* PC."""

    caller_context = (None, "BRISC", 0x1111)

    def probe(self):
        return find_issuer()


def test_the_caller_context_fallback_finds_the_issuer():
    """With no core object on the stack, the ``(unit, label, pc)`` tuple the
    memory space already carries answers instead."""
    issuer = _FakeMemorySpace().probe()
    assert issuer is not None
    assert (issuer.core, issuer.pc) == ("BRISC", 0x1111)


def test_a_live_core_outranks_a_stale_caller_context():
    """``caller_context`` is stamped once per tick and never cleared, so it can
    be stale. A core actually on the stack is proof, and wins -- even though
    the memory space's frame is the inner one."""
    issuer = _FakeCore().probe(_FakeMemorySpace())
    assert issuer is not None
    assert (issuer.core, issuer.pc) == ("TRISC2", 0xABC)


# --- 2. the existing message is not regressed --------------------------------


def _expected_prefix(src, dst, modulus, path, noc, tile):
    from tt_sim.network.alignment import DISABLE_ENV_VAR

    src_rem, dst_rem = src % modulus, dst % modulus
    return (
        f"NoC{noc} {tile} {path}: misaligned transfer. Source address {hex(src)} "
        f"and destination address {hex(dst)} must be congruent modulo "
        f"{modulus} (required alignment: matching low {modulus.bit_length() - 1} "
        f"address bits), but {hex(src)} % {modulus} = {src_rem} and "
        f"{hex(dst)} % {modulus} = {dst_rem}. Hardware would silently "
        f"corrupt or drop this transfer (see WormholeB0/NoC/Alignment.md). "
        f"Set {DISABLE_ENV_VAR}=1 to disable alignment checking."
    )


def test_the_original_message_survives_verbatim(dram_mid):
    """Everything the message said before is still the *prefix* of what it
    says now: addresses, moduli, remainders, path, tile, NoC, and the
    disable-checking hint."""
    from tt_sim.device.wormhole import Wormhole

    device = Wormhole()
    tile = device.tensix_tiles[0].noc0_router.id_pair
    program, _ = _noc_read_program(dram_mid, 0x1010, 0x2000)
    _device, exc = _run_program(program)
    prefix = _expected_prefix(0x1010, 0x2000, 32, "DRAM -> L1 read", 0, tile)
    assert str(exc).startswith(prefix), (
        f"\n got: {str(exc)[: len(prefix)]!r}\nwant: {prefix!r}"
    )
    # The additions are appended on their own indented lines, so a consumer
    # matching the first line of the message is unaffected.
    assert str(exc)[len(prefix) :].startswith("\n  ")


# --- 3. attribution cannot change *when* the error fires ---------------------

#: (src, dst, modulus, raises) over the DRAM read rule. Includes the pair that
#: satisfies C16 but not C32, which is the one a weaker rule would let through.
_MATRIX = [
    (0x1000, 0x2000, True),
    (0x1010, 0x2010, True),
    (0x1004, 0x2004, True),
    (0x1010, 0x2000, False),
    (0x1004, 0x2000, False),
    (0x1000, 0x2008, False),
]


def _accepts(device, src, dst):
    initiator = device.tensix_tiles[0].noc0_router.request_initiators[0]
    initiator.target_addr_mid = _dram_mid(device)
    initiator.target_addr_low = src
    initiator.ret_addr_low = dst
    initiator.at_len_be = 64
    initiator.ctrl = 0
    initiator.cmd_ctrl = 1
    try:
        initiator.initiate()
    except NoCAlignmentError:
        return False
    return True


def test_the_accept_reject_matrix_is_the_congruence_rule():
    from tt_sim.device.wormhole import Wormhole

    device = Wormhole()
    for src, dst, expected in _MATRIX:
        assert _accepts(device, src, dst) is expected, (hex(src), hex(dst))
        assert (src % 32 == dst % 32) is expected


def test_a_broken_describer_changes_nothing_about_when_it_raises(monkeypatch):
    """Sabotage the describer: the same transfers must still raise, with the
    same exception type and the same original message."""
    import tt_sim.network.attribution as attribution
    from tt_sim.device.wormhole import Wormhole

    def boom(*args, **kwargs):
        raise RuntimeError("the describer is broken")

    monkeypatch.setattr(attribution, "provenance", boom)
    device = Wormhole()
    tile = device.tensix_tiles[0].noc0_router.id_pair
    for src, dst, expected in _MATRIX:
        assert _accepts(device, src, dst) is expected, (hex(src), hex(dst))
    initiator = device.tensix_tiles[0].noc0_router.request_initiators[0]
    initiator.target_addr_low, initiator.ret_addr_low = 0x1010, 0x2000
    initiator.ctrl, initiator.cmd_ctrl = 0, 1
    with pytest.raises(NoCAlignmentError) as caught:
        initiator.initiate()
    # Not merely "still raises": the message is the untouched original, so a
    # describer that starts throwing degrades to the old behaviour exactly.
    assert str(caught.value) == _expected_prefix(
        0x1010, 0x2000, 32, "DRAM -> L1 read", 0, tile
    )


def test_attach_provenance_never_raises():
    """Every input a caller could plausibly have, including nonsense."""

    class Hostile:
        @property
        def at_len_be(self):
            raise ValueError("no")

    exc = NoCAlignmentError("original")
    for request in (None, object(), Hostile(), 17):
        attach_provenance(exc, request, 0x1000, 0x2000)
    assert str(exc).startswith("original")


def test_provenance_is_empty_when_there_is_nothing_to_say():
    assert provenance(None, 0x1000, 0x2000) == ""


def test_span_covers_the_whole_transfer():
    assert span(0x1000, 64) == "0x1000..0x103f"
    assert span(0x1000, 0) == "0x1000"


# --- 4. the describer is not loaded at all by a run that does not fault ------


def test_attribution_is_not_imported_by_a_passing_run():
    """The cost of attribution on the passing path is zero, and this is what
    makes that checkable rather than asserted: the module is not even loaded
    by a device that issues correctly aligned transfers."""
    code = textwrap.dedent(
        """
        import sys
        from tt_sim.device.wormhole import Wormhole

        device = Wormhole()
        initiator = device.tensix_tiles[0].noc0_router.request_initiators[0]
        coord = device.dram_tiles[0].noc0_router.id_pair
        initiator.target_addr_mid = (coord[0] << 4) | (coord[1] << 10)
        initiator.at_len_be = 64
        for offset in range(0, 0x400, 0x20):
            initiator.target_addr_low = 0x1000 + offset
            initiator.ret_addr_low = 0x20000 + offset
            initiator.ctrl = 0
            initiator.cmd_ctrl = 1
            initiator.initiate()
        assert "tt_sim.network.tt_noc" in sys.modules
        print("attribution" if "tt_sim.network.attribution" in sys.modules else "clean")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip() == "clean", proc.stdout


# --- 5. naming the function, when an ELF can be trusted ----------------------


@pytest.fixture(scope="module")
def dwarf_elf(tmp_path_factory):
    """A host ``gcc -g`` ELF. DWARF is machine-independent for the parts used
    here, and there is no RISC-V toolchain to rely on -- the same fixture
    strategy ``tt_sim/trace/attribution_test.py`` uses."""
    if shutil.which("gcc") is None:
        pytest.skip("no gcc to build a DWARF fixture")
    tmp = tmp_path_factory.mktemp("noc_dwarf")
    src = tmp / "kernel.c"
    src.write_text(
        textwrap.dedent(
            """
            static inline int write_reg(int x) { return x * 3 + 1; }
            static inline int async_write(int x) { return write_reg(x) + 2; }
            int kernel_main(int x) { return async_write(x) + async_write(x + 1); }
            int main(void) { return kernel_main(7) & 1; }
            """
        )
    )
    out = tmp / "kernel.elf"
    proc = subprocess.run(
        ["gcc", "-g3", "-O2", "-o", str(out), str(src)], capture_output=True
    )
    if proc.returncode != 0:
        pytest.skip(f"gcc could not build the fixture: {proc.stderr.decode()[:200]}")
    return out


def test_an_explicit_elf_names_the_function_and_line(dwarf_elf, monkeypatch):
    """``TT_SIM_PROFILE_ELFS`` is an ``explicit`` selection, which is trusted,
    so the PC resolves to a call chain and a source line."""
    from tt_sim.trace.dwarf import DwarfIndex

    index = DwarfIndex()
    assert index.load(dwarf_elf, unit="BRISC"), "fixture carries no DWARF"
    pc = sorted(index._by_unit["BRISC"])[len(index._by_unit["BRISC"]) // 2]

    monkeypatch.setenv("TT_SIM_PROFILE_ELFS", f"BRISC:{dwarf_elf}")
    source = describe_source(Issuer("BRISC", pc, None))
    assert source.note == "", source.note
    assert source.how == "explicit"
    assert source.elf == str(dwarf_elf)
    assert source.location, "no source line resolved"
    assert source.chain, "no enclosing function resolved"


def test_an_unproved_elf_is_never_named(tmp_path, monkeypatch):
    """A cache hit chosen by mtime alone is a guess. Naming a function from it
    would put some other kernel's symbols on this run's fatal error, so the
    message says why it has nothing instead."""
    directory = tmp_path / "hash" / "kernels" / "gemm" / "cfg" / "brisc"
    directory.mkdir(parents=True)
    (directory / "brisc.elf").write_bytes(b"\x7fELFnot-really")
    monkeypatch.delenv("TT_SIM_PROFILE_ELFS", raising=False)
    monkeypatch.setenv("TT_METAL_CACHE", str(tmp_path))
    monkeypatch.delenv("TT_METAL_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("TT_METAL_HOME", raising=False)

    source = describe_source(Issuer("BRISC", 0x1234, None))
    assert source.kernel == ""
    assert source.chain == []
    assert "no ELF proved resident for BRISC" in source.note
    assert "TT_SIM_PROFILE_ELFS" in source.note


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if fn.__code__.co_argcount:
                continue  # needs a pytest fixture
            fn()
    print("attribution tests OK")


if __name__ == "__main__":
    main()
