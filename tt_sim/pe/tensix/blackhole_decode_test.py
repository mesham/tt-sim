"""Tests for the Blackhole shifted-field instruction decodes.

Runs standalone (``python3 -m tt_sim.pe.tensix.blackhole_decode_test``) or under
pytest. The shared ``tensix_instructions.yaml`` encodes the Wormhole bit
positions, so any field Blackhole moved has to be re-read from the raw 32-bit
word by the handler. These tests pin the raw-bit readers for STALLWAIT's
``wait_res`` and GAPOOL/GMPOOL's ``pool_addr_mode``, check the generic math
``addr_mode`` reader still behaves, and assert Wormhole decodes are untouched.

Bit ranges are from ttsim's ``data/{bh,wh}/tensix_isa.json`` **except**
Blackhole's ``wait_res``, where that file and the published ISA page disagree
and the page wins -- see ``test_stallwait_c12_survives_the_blackhole_trim`` and
``TensixSyncUnit._read_wait_res``.
"""

from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixInstructionDecoder

STALLWAIT = 0xA2
GAPOOL = 0x34
MVMUL = 0x26
MOP = 0x1


def _backend(blackhole):
    return TensixCoProcessor(blackhole=blackhole).getBackend()


def _decode(instruction):
    info = TensixInstructionDecoder.getInstructionInfo(instruction)
    return info, info["instr_args"]


def _stallwait(wait_res, stall_res=1):
    return (STALLWAIT << 24) | (stall_res << 15) | wait_res


def _gapool(pool_addr_mode, max_pool_index_en=0, dst=0):
    # Blackhole layout: dst 9:0, max_pool_index_en 14:14, pool_addr_mode 17:15
    return (GAPOOL << 24) | (pool_addr_mode << 15) | (max_pool_index_en << 14) | dst


def _stallwait_cond_mask(backend, instruction):
    info, args = _decode(instruction)
    backend.sync_unit.handle_stallwait(info, 0, args)
    return backend.getFrontendThread(0).wait_gate.latchedWaitInstruction.condition_mask


def test_stallwait_wait_res_is_13_bits_on_blackhole():
    # Blackhole's wait_res is raw 12:0 (Wormhole 14:0), so anything the shared
    # (Wormhole-width) table pulls in from bits 14:13 must not reach the
    # condition mask. 0x400 is p_stall::TRISC_CFG as the Blackhole LLK emits it.
    instruction = _stallwait(0x400 | (1 << 13))
    assert _stallwait_cond_mask(_backend(True), instruction) == 0x400


def test_stallwait_c12_survives_the_blackhole_trim():
    """Bit 12 is *in* the field, against ttsim's data file and with the page.

    The disagreement, and why this side of it: ``STALLWAIT.md`` (BlackholeA0)
    writes the syntax as ``TT_STALLWAIT(/* u9 */ BlockMask, /* u13 */
    ConditionMask)``, its encoding diagram gives ConditionMask bits 0-12, and
    its condition table runs C0 through C12 -- C12 being "Any thread has an
    instruction in any stage of the Configuration Unit pipeline". tt-metal's own
    Blackhole LLK header defines ``p_stall::CFGEXU = 0x1000`` for it. Against
    that, ttsim's ``data/bh/tensix_isa.json`` says ``wait_res`` is ``11:0`` --
    while ttsim's executor still tests the residual for ``0x1000``. See
    ``TensixSyncUnit._read_wait_res``.

    Trimming it did not make the wait shorter, it made it *different*: with the
    mask emptied, ``handle_stallwait``'s empty-mask fallback selected the
    all-resources set (C0-C3 today, C0-C6 while that fallback was still
    Wormhole's ``0x7F``), which is another set of conditions entirely.
    """
    assert _stallwait_cond_mask(_backend(True), _stallwait(1 << 12)) == 0x1000
    assert _stallwait_cond_mask(_backend(True), _stallwait(0x400 | (1 << 12))) == 0x1400


def test_stallwait_wait_res_unchanged_on_wormhole():
    instruction = _stallwait(0x400 | (1 << 13))
    _, args = _decode(instruction)
    assert args["wait_res"] == 0x2400
    assert _stallwait_cond_mask(_backend(False), instruction) == 0x2400


def test_stallwait_empty_mask_default_is_the_arch_s_own_all_resources():
    """``0x0F`` on Blackhole, ``0x7F`` on Wormhole -- each page's own constant.

    ``STALLWAIT.md``'s functional model reads ``ConditionMask = ConditionMask ?
    ConditionMask : 0x7F`` on Wormhole and ``: 0x0F`` on Blackhole. They mean
    the same thing -- every resource this thread has outstanding -- in each
    arch's numbering: Wormhole's C0-C6 are ThCon, both unpackers and all four
    packers, and Blackhole collapses the four packer conditions into one, so its
    C0-C3 are the same set.

    tt-sim used ``0x7F`` on both until 2026-08-12. On Blackhole that is not a
    superset but a different set: bits 4-6 there are C4 (an instruction in any
    stage of the Matrix Unit pipeline) and C5/C6 (``SrcA``/``SrcB`` not yet
    handed back to the *unpackers*) -- an invented wait plus two inverted ones.

    On Blackhole the empty mask includes one left empty by the 13-bit trim, i.e.
    one whose only set bits were 14:13.
    """
    assert _stallwait_cond_mask(_backend(True), _stallwait(1 << 13)) == 0x0F
    assert _stallwait_cond_mask(_backend(True), _stallwait(0)) == 0x0F
    assert _stallwait_cond_mask(_backend(False), _stallwait(0)) == 0x7F


def test_gapool_pool_addr_mode_is_bits_17_15_on_blackhole():
    # Blackhole renames GAPOOL/GMPOOL's addr_mode to pool_addr_mode and widens
    # it to 17:15 (Wormhole 16:15) - unlike every other math instruction, which
    # moves to 16:14, so bit 14 here is max_pool_index_en and must not leak in.
    matrix = _backend(True).matrix_unit
    for addr_mod in range(8):
        info, args = _decode(_gapool(addr_mod, max_pool_index_en=1))
        assert matrix._read_pool_addr_mode(info, args) == addr_mod


def test_gapool_generic_addr_mode_reader_would_be_wrong_on_blackhole():
    # Guards the reason _read_pool_addr_mode exists: the generic reader takes
    # bits 16:14, which for GAPOOL is a shifted-in max_pool_index_en.
    matrix = _backend(True).matrix_unit
    info, args = _decode(_gapool(5, max_pool_index_en=1))
    assert matrix._read_addr_mode(info, args) == ((5 << 1) | 1) & 0x7


def test_gapool_pool_addr_mode_unchanged_on_wormhole():
    matrix = _backend(False).matrix_unit
    info, args = _decode(_gapool(3))
    assert matrix._read_pool_addr_mode(info, args) == args["addr_mode"]


def test_math_addr_mode_reader_unaffected():
    # MVMUL keeps the common math layout: raw 16:14 on Blackhole, the decoded
    # (16:15) argument on Wormhole.
    instruction = (MVMUL << 24) | (5 << 14)
    info, args = _decode(instruction)
    assert _backend(True).matrix_unit._read_addr_mode(info, args) == 5
    assert _backend(False).matrix_unit._read_addr_mode(info, args) == args["addr_mode"]


def test_mop_fields_are_architecture_independent():
    # MOP's zmask/loop_count fields sit at the same bits on both architectures
    # (Blackhole only renames zmask_lo16), so the MOP expander needs no raw-bit
    # read - this pins that.
    instruction = (MOP << 24) | (1 << 23) | (0x2A << 16) | 0xBEEF
    _, args = _decode(instruction)
    assert args["zmask_lo16"] == 0xBEEF
    assert args["loop_count"] == 0x2A
    assert args["mop_type"] == 1


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("blackhole_decode_test OK: STALLWAIT/GAPOOL/MVMUL/MOP decodes verified")


if __name__ == "__main__":
    main()
