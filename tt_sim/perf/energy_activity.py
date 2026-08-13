"""Reduce a tt-sim counter dataset to a per-launch **activity vector**.

This is the simulator half of the ranking-level energy estimator (ROADMAP v2.0
item 5). It reads nothing but the Parquet counter dataset a run already emits
(``TT_SIM_TRACE_COUNTERS``) and writes one row per workload:

    label, arm, inner, launches, sim_cycles, <activity terms...>

**It changes no computed value and no cycle count.** Nothing here is imported by
the simulator; the dependency runs one way, from this module to
:mod:`tt_sim.trace.report`, which is itself a reader.

What an activity vector is for
------------------------------

The energy model is

    E_launch  =  c_launch  +  Σ_j c_j · a_j
    P_board   =  P_static  +  rate · E_launch

where ``a`` is the vector this module emits, ``rate`` is the measured launch
rate from ``perfbench/energybench``, and ``P_static`` is the measured idle
baseline. The coefficients ``c`` are **not set here and are not set anywhere in
``tt_sim/``** -- see :mod:`tt_sim.perf.energy_rank` and the quarantine note in
``perfbench/energybench/README.md``. They will be FITTED to silicon, which makes
them a different kind of number from everything in ``unit_costs.yaml``, and they
are kept physically outside that file for exactly that reason.

The terms
---------

:data:`ACTIVITY_TERMS` is a **fixed, ordered** schema: the design matrix's
columns are these names in this order, so a term added tomorrow shifts no
existing column and every CSV ever written stays readable by name. Each term is
a per-launch quantity -- the module divides the run total by the launch count,
because the counter dataset spans every launch the harness made.

Terms fall into two families, and the distinction matters when reading a fit:

*Volume* terms (``instr_retired``, ``noc_bytes_total``, ``noc_txns``,
``tensix_dispatch``) are counted whether or not the cost model is on. A
coefficient against one of them is "energy per event".

*Occupancy* terms (every ``*_busy_cycles``, ``rv_stall_cycles``,
``tensix_stall_cycles``) exist **only with ``TT_SIM_COST_MODEL=1``** -- the
aggregator emits a counter only when something incremented it, so an un-modelled
run has no such rows at all rather than rows asserting zero. Reading a
coefficient against an occupancy term from a cost-model-off dataset would be
reading a coefficient against a column of zeros, so ``--cost-model`` is recorded
in the CSV, :mod:`tt_sim.perf.energy_rank` drops a zero-spread column from the
design entirely, and one named explicitly with ``--terms`` is refused by the
identifiability gate rather than fitted.

Usage
-----

::

    TT_SIM_COST_MODEL=1 TT_SIM_TRACE_COUNTERS=/tmp/ab ./perfbench/run.sh energybench \\
        -- --arm mm --inner 8 --iters 1
    python3 -m tt_sim.perf.energy_activity --counters /tmp/ab \\
        --label mm-8 --arm mm --inner 8 --launches 1 --cost-model 1 \\
        --out activity.csv --append
"""

import argparse
import csv
import sys
from pathlib import Path

#: Every counter that lands in ``busy_cycles`` is keyed by the backend unit's
#: name as it appears in ``unit_id[3]`` -- the canonical join key documented in
#: ``tt_sim.trace.events.BACKEND_UNIT_ALIASES``. These are the ones with a
#: plausible independent energy signature.
_BUSY_UNITS = ("MATRIX", "SFPU", "PACKER", "UNPACKER", "THCON", "MOVER")

#: The baby RISC-V cores. ``instr_retired`` and ``stall_cycles`` are published
#: under these unit names.
_RV_UNITS = ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")

#: The activity vector's columns, in order. **Append only.** A consumer joins on
#: the name, but the fitting code reports coefficients positionally and a
#: reordering would silently relabel every stored fit.
ACTIVITY_TERMS = (
    # -- volume terms: present with the cost model on or off ----------------
    "instr_retired",
    "tensix_dispatch",
    "noc_bytes_total",
    "noc_txns",
    # -- occupancy terms: present ONLY with TT_SIM_COST_MODEL=1 -------------
    "matrix_busy_cycles",
    "sfpu_busy_cycles",
    "packer_busy_cycles",
    "unpacker_busy_cycles",
    "thcon_busy_cycles",
    "mover_busy_cycles",
    "rv_stall_cycles",
    "tensix_stall_cycles",
    "noc_flight_cycles",
)

#: The columns that identify a row, ahead of the terms.
KEY_COLUMNS = ("label", "arch", "arm", "inner", "launches", "cost_model", "sim_cycles")

CSV_COLUMNS = KEY_COLUMNS + ACTIVITY_TERMS


def _unit_of(key: str) -> str:
    """``"2,1 MATRIX"`` -> ``"MATRIX"``. ``load_counters`` builds the key."""
    return key.rsplit(" ", 1)[-1]


def reduce_counters(
    totals: dict[tuple[str, str], int],
) -> dict[str, int]:
    """Fold ``{(unit_key, counter): value}`` into the fixed activity schema.

    Every term is a plain sum across units, because energy is extensive: two
    cores each retiring a thousand instructions cost what one retiring two
    thousand costs, to the resolution this instrument can ever reach.
    """
    out = dict.fromkeys(ACTIVITY_TERMS, 0)
    for (unit_key, counter), value in totals.items():
        unit = _unit_of(unit_key)
        if counter == "instr_retired" and unit in _RV_UNITS:
            out["instr_retired"] += value
        elif counter == "stall_cycles" and unit in _RV_UNITS:
            out["rv_stall_cycles"] += value
        elif counter == "dispatch_total":
            out["tensix_dispatch"] += value
        elif counter == "tensix_stall_cycles":
            out["tensix_stall_cycles"] += value
        elif counter == "noc_bytes_total":
            out["noc_bytes_total"] += value
        elif counter == "noc_txns_timed":
            out["noc_txns"] += value
        elif counter == "noc_flight_cycles":
            out["noc_flight_cycles"] += value
        elif counter == "busy_cycles" and unit in _BUSY_UNITS:
            out[f"{unit.lower()}_busy_cycles"] += value
    return out


def per_launch(raw: dict[str, int], launches: int) -> dict[str, float]:
    """Divide the run totals by the number of launches they span.

    A launch count of zero is a caller bug rather than a degenerate run -- the
    harness always reports at least one -- so it raises rather than returning a
    vector of infinities that would then be fitted.
    """
    if launches <= 0:
        raise ValueError(f"launches must be positive, got {launches}")
    return {k: v / launches for k, v in raw.items()}


def load_activity(path: Path | str) -> list[dict]:
    """Read an activity CSV back, with the numeric columns typed.

    ``#`` lines are skipped, matching every other dataset in
    ``tt_sim/perf/datasets/``: a file that cannot carry a provenance header is a
    file whose provenance ends up in a commit message.
    """
    rows = []
    with open(path, newline="") as fh:
        body = (line for line in fh if not line.lstrip().startswith("#"))
        for row in csv.DictReader(body):
            typed = dict(row)
            for term in ACTIVITY_TERMS:
                typed[term] = float(row.get(term, 0) or 0)
            for key in ("inner", "launches", "cost_model", "sim_cycles"):
                typed[key] = int(float(row.get(key, 0) or 0))
            rows.append(typed)
    return rows


def write_row(path: Path | str, row: dict, append: bool = False) -> None:
    path = Path(path)
    fresh = not append or not path.exists() or path.stat().st_size == 0
    with open(path, "a" if append else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        if fresh:
            writer.writeheader()
        writer.writerow({k: row.get(k, 0) for k in CSV_COLUMNS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--counters", required=True, help="TT_SIM_TRACE_COUNTERS directory")
    ap.add_argument("--label", required=True, help="workload label, e.g. mm-4096")
    ap.add_argument("--arm", default="", help="energybench arm name")
    ap.add_argument("--arch", default="", help="blackhole | wormhole")
    ap.add_argument("--inner", type=int, default=0)
    ap.add_argument(
        "--launches",
        type=int,
        default=1,
        help="launches the counter dataset spans; the vector is divided by it",
    )
    ap.add_argument("--cost-model", type=int, default=0, choices=(0, 1))
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args(argv)

    from tt_sim.trace.report import load_counters

    directory = Path(args.counters)
    if not directory.is_dir():
        print(f"energy_activity: no counter dataset at {directory}", file=sys.stderr)
        return 2
    totals, span = load_counters(directory)
    if not totals:
        print(
            f"energy_activity: {directory} holds no counter rows -- was "
            "TT_SIM_TRACE_COUNTERS set for the SERVER's environment?",
            file=sys.stderr,
        )
        return 2

    vector = per_launch(reduce_counters(totals), args.launches)
    row = {
        "label": args.label,
        "arch": args.arch,
        "arm": args.arm,
        "inner": args.inner,
        "launches": args.launches,
        "cost_model": args.cost_model,
        "sim_cycles": span,
    }
    row.update(vector)
    write_row(args.out, row, append=args.append)

    width = max(len(t) for t in ACTIVITY_TERMS)
    print(
        f"# activity vector: {args.label} (per launch, over {args.launches} launches)"
    )
    print(f"#   sim cycle span: {span}")
    for term in ACTIVITY_TERMS:
        print(f"  {term:<{width}}  {vector[term]:>16.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
