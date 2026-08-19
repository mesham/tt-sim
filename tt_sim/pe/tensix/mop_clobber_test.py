"""The MOP expander is last-writer-wins, and that is why tt-sim sees init clobbers.

Compute-kernel ``*_init()`` functions are, at the hardware level, bursts of
writes to backend configuration, address modifiers and **MOP expander
configuration**. The last of those is the one that makes a *reordering* of two
inits change the answer rather than merely the timing, and it is the mechanism
behind this fault class's deadlocks. Per
https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MOPExpander.md
the expander has "separate write-only configuration mapped into the RISCV
address space" — one bank per thread, nine words, no shadow and no restore.

tt-metal's side of the same fact: ``ckernel_template::program()`` and
``ckernel_unpack_template::program()`` (``tt_metal/tt-llk/common/inc/
ckernel_template.h``) write all nine words unconditionally — no dirty check, no
compare-and-skip — the destructor is empty, and ``run()`` is ``static`` and
argument-free, so ``TTI_MOP`` executes whatever was programmed last. An init
that programs the expander therefore *destroys* the macro a previous init
programmed, and any op still expecting the earlier macro runs the later one.

The concrete case, which is the deadlock the compiler team reported hitting:
``_llk_unpack_AB_mop_config_`` programs the unpacker macro with ``m_unpackB =
true`` (issue both the SrcA and the SrcB unpack), while
``_llk_unpack_A_mop_config_`` — reached from ``init_sfpu`` /
``unary_op_init_common`` / ``copy_tile_init`` — programs it with ``m_unpackB =
false``. Emit the second after the first and the following ``add_tiles``
unpacks no SrcB at all: no SrcB dvalid, the matrix unit's ``ELWADD`` waits for
ever, the output CB never drains and the producer blocks on
``cb_reserve_back``.

**This test is the load-bearing half of a documented claim.**
``docs/cost-model-caveats-for-consumers.md`` tells consumers that tt-sim
*catches* this class — measured live, a ``copy_tile_init`` inserted between
``add_tiles_init`` and ``add_tiles`` in ``examples/four`` returns 254 of 256
elements wrong instead of passing. It catches it precisely because the model
below is faithful. If this file goes green-but-wrong, that claim rots silently,
so both directions are pinned: a clobber *does* change the expansion, and an
unrelated write *does not*.

Run standalone (``python3 -m tt_sim.pe.tensix.mop_clobber_test``) or under
pytest.
"""

from tt_sim.pe.tensix.frontend import TensixMOPExpander

# Stand-ins for the encoded Tensix instructions the LLK stores into the
# expander's config. Their values are arbitrary and deliberately distinct; only
# *which* of them come back out of an expansion is under test.
UNPACK_A0 = 0xA0A0A0A0
UNPACK_B = 0xB0B0B0B0
SKIP_A0 = 0x5A5A5A5A
SKIP_B = 0x5B5B5B5B

# ``mop_cfg[1]`` bit 0 is the expander's "there is a SrcB instruction" flag —
# ``m_unpackB`` on the tt-metal side, ``hasb`` in ``expand_template_zero``.
FLAG_HAS_B = 1


def _program_unpack_ab(mop):
    """What ``_llk_unpack_AB_mop_config_`` leaves in the expander: A then B."""
    mop.write(1 * 4, FLAG_HAS_B)
    mop.write(2 * 4, UNPACK_B)
    mop.write(3 * 4, UNPACK_A0)
    mop.write(7 * 4, SKIP_A0)
    mop.write(8 * 4, SKIP_B)


def _program_unpack_a(mop):
    """What ``_llk_unpack_A_mop_config_`` leaves in the expander: A only.

    This is the clobber: it is a *complete* reprogramming of the same nine
    words, so clearing the has-B flag is enough to strip the SrcB unpack out of
    every subsequent expansion.
    """
    mop.write(1 * 4, 0)
    mop.write(3 * 4, UNPACK_A0)
    mop.write(7 * 4, SKIP_A0)


def _expand_one_iteration(mop):
    """Expand a single-iteration, nothing-masked-off template-zero MOP."""
    return list(mop.expand_template_zero(mask=0, count1=0))


def test_unpack_ab_macro_issues_both_operands():
    """The baseline: the AB macro emits a SrcA unpack and a SrcB unpack."""
    mop = TensixMOPExpander(frontend=None)
    _program_unpack_ab(mop)

    assert _expand_one_iteration(mop) == [UNPACK_A0, UNPACK_B]


def test_a_only_init_clobbers_the_ab_macro_and_drops_srcb():
    """The fault: a second init reprograms the shared bank, and SrcB vanishes.

    ``add_tiles_init()`` then ``copy_tile_init()`` then ``add_tiles()`` — the
    ``add_tiles`` runs the macro ``copy_tile_init`` left behind, which never
    unpacks SrcB. Nothing warns, because from the expander's point of view the
    second programming is simply the current one.
    """
    mop = TensixMOPExpander(frontend=None)
    _program_unpack_ab(mop)
    _program_unpack_a(mop)

    expanded = _expand_one_iteration(mop)
    assert expanded == [UNPACK_A0], (
        "the A-only init must strip the SrcB unpack out of the macro; "
        f"got {[hex(i) for i in expanded]}"
    )
    assert UNPACK_B not in expanded


def test_reprogramming_is_unconditional_not_a_dirty_check():
    """Writing the *same* value is still a write, and order still decides.

    ``program()`` has no compare-and-skip, so "the config already holds this"
    is never a reason the expander leaves a word alone. Pinning this stops a
    well-meaning optimisation from introducing a dirty check that would make
    tt-sim miss the clobber above.
    """
    mop = TensixMOPExpander(frontend=None)
    _program_unpack_ab(mop)
    _program_unpack_a(mop)
    # The clobbered macro is not sticky either: the AB init reinstates it.
    _program_unpack_ab(mop)

    assert _expand_one_iteration(mop) == [UNPACK_A0, UNPACK_B]


def test_an_unrelated_config_word_does_not_change_the_macro():
    """The other direction: not every write to the bank is a clobber.

    A guard that treated *any* write to the expander as destroying the current
    macro would fire on correct kernels. The words an expansion reads are the
    only ones that can change it.
    """
    mop = TensixMOPExpander(frontend=None)
    _program_unpack_ab(mop)
    before = _expand_one_iteration(mop)

    # Word 0 is the outer-loop count, read only by template *one*; a
    # template-zero expansion never looks at it.
    mop.write(0 * 4, 0x7F)

    assert _expand_one_iteration(mop) == before


def test_template_one_runs_whatever_was_programmed_last():
    """The same last-writer-wins property on the MATH-side (template one) path.

    ``add_tiles_init`` programs this one via ``ckernel_template::program()``;
    ``run()`` is static, so the macro that executes is a property of the
    expander's state, never of the ``MOP`` instruction that triggers it.
    """
    mop = TensixMOPExpander(frontend=None)
    # One outer iteration, one inner iteration, a single distinguishable body.
    mop.write(0 * 4, 1)
    mop.write(1 * 4, 1)
    mop.write(5 * 4, UNPACK_A0)
    first = list(mop.expand_template_one())

    mop.write(5 * 4, UNPACK_B)
    second = list(mop.expand_template_one())

    assert UNPACK_A0 in first
    assert UNPACK_B not in first
    assert UNPACK_B in second
    assert UNPACK_A0 not in second


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("mop_clobber_test: all passed")
