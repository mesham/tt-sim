"""The energy coefficients are FITTED, and this is the tripwire that keeps them
out of the provenance ladder.

Why a test rather than a comment
--------------------------------

``tt_sim/perf/unit_costs.yaml`` and ``tt_sim/pe/tensix/tensix_instruction_costs.yaml``
exist to make un-sourced numbers impossible to charge: every entry carries a
provenance, they are ranked, and ``costs_test.py`` records that there are
currently **zero** ``estimated`` entries in either file. Tenstorrent publishes no
per-event energy figure at all, so an energy coefficient can never be better than
``estimated`` -- it is a regression coefficient from a ~1 Hz board-level power
meter against a nine-point design.

The obvious failure mode is not malice, it is tidying: a later reader finds
``fitted_energy_coefficients.yaml`` sitting in ``perfbench/`` and moves it "where
the other cost tables are". Three independent things have to stop that, and each
is asserted here in **both** directions:

1. the ``fitted`` provenance token is **not in** :data:`PROVENANCE_RANK`, so the
   cost loader raises on any table carrying it;
2. :func:`check_destination` refuses to write the file under ``tt_sim/`` at all;
3. neither cost table contains any energy vocabulary today.
"""

from pathlib import Path

import pytest

from tt_sim.perf.costs import PROVENANCE_RANK, load_costs
from tt_sim.perf.energy_rank import FITTED_PROVENANCE, check_destination

REPO = Path(__file__).resolve().parents[2]
COST_YAMLS = (
    REPO / "tt_sim" / "perf" / "unit_costs.yaml",
    REPO / "tt_sim" / "pe" / "tensix" / "tensix_instruction_costs.yaml",
)


def test_fitted_is_not_a_rankable_provenance():
    """The load-bearing half: a cost entry stamped ``fitted`` cannot be ranked,
    so :meth:`CostTable.weakest_provenance` raises rather than quietly treating
    it as anything. Pasting one of these coefficients into a cost table breaks
    the loader."""
    assert FITTED_PROVENANCE not in PROVENANCE_RANK
    with pytest.raises(KeyError):
        PROVENANCE_RANK[FITTED_PROVENANCE]
    # And the other direction: the vocabulary that IS ranked still is, so this
    # test is asserting a boundary rather than that the dict is empty.
    assert PROVENANCE_RANK["estimated"] < PROVENANCE_RANK["isa_doc"]


def test_the_writer_refuses_the_cost_model_tree_and_allows_perfbench(tmp_path):
    with pytest.raises(ValueError, match="refusing to write"):
        check_destination(REPO / "tt_sim" / "perf" / "fitted_energy_coefficients.yaml")
    with pytest.raises(ValueError, match="refusing to write"):
        check_destination(REPO / "tt_sim" / "pe" / "tensix" / "energy.yaml")
    # The intended home is accepted, or the refusal would be indiscriminate
    # rather than aimed.
    home = REPO / "perfbench" / "energybench" / "fitted_energy_coefficients.yaml"
    assert check_destination(home) == home
    assert check_destination(tmp_path / "anything.yaml")


def test_no_cost_table_carries_energy_vocabulary():
    """A coefficient does not have to arrive labelled ``fitted`` to do damage --
    it could arrive as ``energy_pj`` with a plausible-looking provenance. Nothing
    in either table talks about energy today, and this pins that."""
    forbidden = ("energy", "joule", "watt", "pico_joule", "pj_per", "picojoule")
    for path in COST_YAMLS:
        text = path.read_text().lower()
        found = [word for word in forbidden if word in text]
        assert not found, (
            f"{path.name} mentions {found}: energy has no home in the cost tables"
        )


def test_the_cost_tables_still_load_and_still_rank_everything():
    """The control for the test above: the tables are readable and every entry
    still has a ranked provenance, so a clean run of this file means "no energy
    got in", not "the tables were unreadable"."""
    for arch in ("wormhole", "blackhole"):
        table = load_costs(arch)
        assert table.sections
        for entry in table.entries():
            assert entry.provenance in PROVENANCE_RANK
