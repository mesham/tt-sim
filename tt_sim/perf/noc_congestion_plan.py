"""Plans for the NoC congestion experiments, with the confounds asserted.

``noc.congestion`` is the largest ``provenance: unknown`` left in
``tt_sim/perf/unit_costs.yaml``. :mod:`tt_sim.perf.noc_dataset_sweep` showed
that tt-metal's shipped 740-row measured dataset cannot *derive* one, and the
reason is identifiability rather than coarseness: every multi-party row varies
the flow count by resizing a grid, so flow count, path length and link sharing
move together; no core coordinates are recorded; and the single aggregate
scalar per key already contains the issuing core's loop and the endpoint's L1
port arbitration. One equation, three unknowns.

This module builds the experiments that *would* settle it, for a card. It is
the planning half of the harness; ``perfbench/nocbench`` is the executing half
and ``tt_sim.perf.noc_congestion_sweep`` the analysing half. The split exists
because **the invariants are the experiment**: if a parameterisation lets two
of {flow count, path length, shared links} move together it has rebuilt the
very dataset whose unidentifiability is the reason for the work, so every
"held fixed" claim below is checked by :func:`check_invariants` at plan time
and re-checked by the sweep against what came back, rather than being asserted
in a comment.

Run it
------

::

    # on the card, once:
    ./build/nocbench --dump-grid                      # -> nocbench-grid-<arch>.csv
    python3 -m tt_sim.perf.noc_congestion_plan \\
        --grid nocbench-grid-blackhole.csv --out plan.csv
    ./build/nocbench --plan plan.csv                  # -> nocbench-<arch>.csv

``perfbench/nocbench/README.md`` is the version of that written for somebody
with a card and no context.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tt_sim.network.tt_noc import noc_route_links

# ---------------------------------------------------------------------------
# What each experiment would show, DECLARED BEFORE ANYTHING WAS RUN.
# ---------------------------------------------------------------------------
#
# A null result is a real possible outcome here, and the whole harness is worth
# nothing unless a null is distinguishable from a benchmark that measured
# nothing. So each entry says what "no congestion effect" looks like AND what
# would have to also be true for that null to be believable.

HYPOTHESES = {
    "hops": """\
EXPERIMENT 1 -- THE INTERCEPT (one flow, subordinate swept over the grid).

  Held fixed: one flow; the master core; transaction size; transaction count;
  direction; NoC. Varying: the subordinate's coordinate, hence the hop count.

  A structural prediction comes first, because it is the thing that decides
  whether "sweep master x subordinate for a fine-grained hop sweep" is even a
  coherent request. On a UNIDIRECTIONAL torus with dimension-ordered routing a
  request and its reply travel the SAME way round the ring, so the round trip
  costs `grid_x` hops for any two cores in one row, `grid_y` for any two in one
  column, and `grid_x + grid_y` for any pair differing on both axes -- three
  values, whatever the coordinates. The sweep therefore yields:

  * TORUS CONFIRMED: latency is flat along a row (all `grid_x`), flat down a
    column (all `grid_y`), flat over the diagonal cases (all `grid_x+grid_y`),
    and the three levels differ. The row family's spread is then a direct
    measurement of this harness's noise floor at constant predicted latency,
    which every other experiment is read against.
  * TORUS WRONG: latency rises with |dx| along a row. That would mean the NoC
    is not the unidirectional ring tt-sim models (bidirectional, or shortest
    path), and every hop count in the cost model is wrong before congestion is
    reached. This is a cheap, decisive check nobody has run.

  Fitting `cycles = a + b * round_trip_hops` over the three levels gives the
  uncongested line: `b` is the per-hop cost (the ISA docs say ~9 cycles router
  to router) and `a` is the issuing core's own path plus the endpoint, which is
  the ~90-cycle residual the shipped dataset leaves and never separates.""",
    "size": """\
POSITIVE CONTROL -- transaction size, one flow, geometry frozen.

  Held fixed: one flow; both coordinates; transaction count; direction; NoC.
  Varying: bytes per transaction.

  tt-sim DOES model this (the NIU holds its injection link for one flit per
  cycle, `noc.link_bandwidth`), so this series must rise with size against the
  simulator. If it does not, the harness is not measuring anything and every
  flat reading elsewhere is meaningless. This is the control that makes the
  forced null in experiments 2-4 interpretable.""",
    "readport": """\
POSITIVE CONTROL -- two masters READING from one subordinate, sharing that
  subordinate's injection port.

  Held fixed: the subordinate; both masters' forward and round-trip hop counts;
  sizes; counts; direction (read); NoC; ONE data-movement RISC per core.
  Varying: whether the second master is present.

  This replaces the `selfport` control, which was RETRACTED -- see
  docs/plans/cost-model.md, "the self-port control is not a measurement". The
  short version: `selfport` put two flows on one core's two DM RISCs, and
  `noc_async_write_barrier` compares a per-NIU HARDWARE ack counter against a
  per-RISC SOFTWARE one, so with two issuers each RISC's wait is satisfied by
  ANY N acks and both kernels stop at the halfway point. That control reads the
  single-flow time whether the injection port serialises perfectly or not at
  all, which was shown by deleting the mechanism in tt-sim and watching the
  control not move. It is also a configuration that can HANG a card
  (BRISC_WR_CMD_BUF and NCRISC_WR_CMD_BUF are both command buffer 0), and
  `check_invariants` now refuses it.

  A read reaches the same shared resource by the route tt-metal actually
  supports: the payload rides the RETURN leg, so both response streams leave
  the subordinate's single NIU and queue on its one injection port. One DM RISC
  per core means every barrier counts only its own core's responses, so the
  rendezvous, the overlap check and the barriers all mean what they say.

  Sized so the port is the BOTTLENECK, which the retracted control was not: at
  four transactions the timed region is dominated by the fixed round trip and
  the per-transaction term the port touches is a small share of it. The planner
  refuses a readport point carrying less than 64 KiB per flow for that reason.

  * THE FLOWS CONTEND: per-transaction cost roughly DOUBLES when the second
    master joins -- the subordinate's port carries twice the payload and each
    master waits for its own responses, so each sees about 2x. Against tt-sim
    this is the prediction under the model as it stands, because `send_response`
    claims the responder's `_tx_free_cycle`; ablating `claim_injection_port` to
    a no-op must collapse it to ~1x, and if it does not this control is as
    blind as the one it replaces.
  * THE FLOWS DO NOT CONTEND: flat. On tt-sim that would mean the injection
    port is not on this path and the control is unusable. On a card it would
    mean the two flows never overlapped in time, which invalidates experiments
    2 and 3.

  What this control does NOT do is attribute the rise to one mechanism. Two
  response streams leaving one tile share the first router-to-router link out
  of it as well as the port, and on a card those are indistinguishable. That is
  fine for a positive control, whose only job is to show the flows really
  contend; the attribution is experiment 2's job. On tt-sim they ARE
  distinguishable, because tt-sim charges nothing whatever for a
  router-to-router link.

  [ADDENDUM 2026-08-05: tt-sim now charges a router-to-router link its
  occupancy, and the conclusion above survives with a different reason -- the
  control still reads 1.48x, unchanged to the digit. Two response streams
  leaving one NIU are already spaced one occupancy apart by its port, so they
  reach the shared first link one occupancy apart and it is never busy. The
  port and the link remain separable on tt-sim; they are still one reading on
  a card.]""",
    "shared": """\
EXPERIMENT 2 -- THE COEFFICIENT. Two flows, N FIXED AT 2, positioned so their
  payload paths share 0, 1, 2 ... k router-to-router links.

  Held fixed: the flow count (2); flow A entirely (same master, same
  subordinate, same route, at every point); flow B's hop count, both forward
  and round trip; the transaction size and count; the direction; the NoC; and
  -- checked link by link -- EVERY leg-pair overlap other than the one being
  swept, which is held at zero. Varying: the number of links flow A's payload
  and flow B's payload have in common.

  That last clause is the whole design and it is why the placement is searched
  rather than written down: both masters sit in one row and each writes to a
  DIFFERENT row, so the two forward paths overlap in the masters' row by an
  amount set by their separation, while the two write-acknowledgement paths
  (which return along the destinations' rows) do not overlap at all. Sliding
  the second master then moves ONE number. `check_invariants` refuses a plan
  in which anything else moved.

  * CONGESTION IS REAL AND PER-LINK: cycles rise linearly in shared links. The
    slope IS the per-hop congestion coefficient, in cycles per shared link per
    concurrent flow -- the number `noc.congestion` does not have, measured with
    the confounds held rather than fitted out.
  * CONGESTION IS REAL BUT SATURATING: cycles jump between 0 and 1 shared links
    and then flatten. That is a bandwidth split at the first shared link, not a
    per-hop latency, and it should be modelled as a link occupancy rather than
    a per-hop adder.
  * NO CONGESTION EFFECT: flat. Believable ONLY if the size control rose, the
    self-port control doubled, and the two flows' timed regions overlapped (the
    sweep computes the overlap fraction from the raw t0/t1 stamps). Against
    tt-sim it MUST be flat -- the model charges only the issuing NIU's own
    port, and nothing at all for a router-to-router link -- so a non-flat
    simulator reading means the plan is not holding what it claims.

    [ADDENDUM 2026-08-05, and deliberately an addendum: the paragraphs above
    were written before anything ran and are not edited. The final sentence
    described tt-sim as it was, and tt-sim has since changed -- the model now
    charges each router-to-router link its occupancy, so a non-flat simulator
    reading at a saturating size is the expected one and no longer indicts the
    plan. Everything above about the CARD is untouched, and the card's answer
    was SATURATING before the model was built to match it.]

  The size axis is its own control: at 64 B a transaction occupies a link for
  one flit against ~90 cycles of issue loop, so the links are ~1 % busy and
  even a real congestion term cannot show. Only the large sizes, where the
  injection port is saturated, can. An effect that appears at 64 B is evidence
  of something other than link sharing.""",
    "contention": """\
EXPERIMENT 3 -- THE CONTENTION CURVE. N flows into ONE subordinate, every
  master at the SAME hop distance from it.

  Held fixed: the subordinate; every master's forward hop count and round-trip
  hop count; sizes; counts; direction; NoC. Varying: N.

  Stated plainly, because it is the honest limit of this experiment: the number
  of shared links CANNOT be held fixed here. N flows converging on one endpoint
  must share the links adjacent to it; that is a geometric fact, not a defect
  in the plan. So this experiment does not identify a per-link coefficient --
  experiment 2 is the one that does -- and what it produces is the endpoint
  contention curve with the per-flow geometry held still, which is strictly
  more than the shipped dataset's grid-resizing rows offer. The planner
  computes the shared-link count at each N and records it as a covariate, so
  the sweep can report the curve against both N and the sharing it implies
  rather than pretending one of them is absent.

  Any congestion model tt-sim ever gains must reproduce this curve; that is the
  "validate, not derive" role rung 2 was climbed for.""",
    "vc": """\
EXPERIMENT 4 -- VIRTUAL CHANNEL ARBITRATION. TWO writers whose payloads share
  exactly ONE router-to-router link, the second writer's VC swept 0-3.

  Held fixed: both masters, both subordinates, every hop count, the shared-link
  count (1), sizes, counts, the NoC, the direction (write), and the first
  writer's VC (1, which is tt-metal's NOC_UNICAST_WRITE_VC and therefore what
  every other flow in this harness uses). Varying: the second writer's VC.

  This is the one congestion mechanism the ISA docs actually name: "if the two
  packets have the same virtual circuit number, then one packet will wait for
  the other". The experiment therefore has to put two packets on one link and
  ask whether the VC number changes what happens -- so the reading at vc=1
  (both writers on one VC) against vc in {0, 2, 3} is the documented effect,
  with a three-point control built in.

  It is aimed straight at the ONE banked congestion result: a single shared
  link at a saturating transaction size roughly halves each flow's bandwidth on
  Blackhole silicon. If that halving is VC arbitration then it must go away
  when the two writers are put on different VCs; if it is link occupancy it
  must not. Nothing else in this harness separates those.

  * VC ARBITRATION IS THE MECHANISM: vc=1 costs about twice vc in {0, 2, 3}.
  * LINK OCCUPANCY IS THE MECHANISM: all four VCs read the same, and equal to
    the shared-link-1 point of experiment 2.
  * tt-sim models no VCs at all, so it must be flat there, and a non-flat
    simulator reading means the plan is not holding what it claims.

  THIS EXPERIMENT USED TO BE BIDIRECTIONAL AND IT HUNG A CARD. The previous
  design was one master reading from and writing to one subordinate at once,
  with the write's VC swept. On a Blackhole card the first and only
  `direction=BIDIR` point never returned, while all 79 unidirectional flows in
  the same session completed. It is not the VC: every one of those 79 flows
  issued its writes on VC 0. It is not the kernel: tt-sim executes the same
  binary and the same plan (64 x 4096 B, BIDIR, VC 0-3) to completion in 4958
  cycles. tt-metal's own `core_bidirectional` suite disables its entire
  directed-ideal family -- same-kernel AND different-kernel, write-VC sweep
  included -- with `GTEST_SKIP() << "Skipping test"; // Timeout issue
  (#36428)`. The root cause is not established and cannot be established from
  here, so `check_invariants` refuses DIR_BIDIR outright rather than leave a
  configuration in the plan that can hang somebody's card.""",
}


# ---------------------------------------------------------------------------
# The grid.
# ---------------------------------------------------------------------------

#: SoC-physical NoC 0 worker columns and rows, per architecture. Transcribed
#: from the SoC descriptors (``driver/wormhole/soc_descriptor.yaml``,
#: ``driver/blackhole/soc_descriptor.yaml``); ``noc_congestion_plan_test.py``
#: reads those files back and asserts these still match, so the two cannot
#: drift. They are needed because a plan's link arithmetic must happen on the
#: FULL torus -- the non-worker columns (DRAM, PCIe) are routers a packet
#: crosses, and a hop count that skipped them would be wrong.
WORKER_COLUMNS = {
    "wormhole": (1, 2, 3, 4, 6, 7, 8, 9),
    "blackhole": (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16),
}
WORKER_ROWS = {
    "wormhole": (1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
    "blackhole": (2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
}
#: SoC NoC grid (columns x rows), the torus the links live on.
SOC_GRID = {"wormhole": (10, 12), "blackhole": (17, 12)}


class PlanError(RuntimeError):
    """A plan could not be built, or an invariant it claims does not hold."""


@dataclass(frozen=True)
class Grid:
    """A ``--dump-grid`` capture, resolved into SoC-physical NoC 0 space.

    ``noc`` is what ``worker_core_from_logical_core`` returned -- the coordinate
    a kernel addresses, which on a card is usually the *translated* space.
    ``phys`` is the SoC-physical NoC 0 coordinate the link arithmetic needs.
    Keeping both, and carrying both into the plan, is what lets the executor
    check that the plan was built against the machine it is running on.
    """

    arch: str
    grid_x: int
    grid_y: int
    coord_space: str  # "noc0" or "translated"
    #: ``{(logical_x, logical_y): (noc_x, noc_y)}`` exactly as dumped.
    noc: dict[tuple[int, int], tuple[int, int]]
    #: ``{(logical_x, logical_y): (phys_x, phys_y)}`` in SoC-physical NoC 0.
    phys: dict[tuple[int, int], tuple[int, int]]

    @property
    def logical_by_phys(self):
        return {v: k for k, v in self.phys.items()}


def load_grid(path):
    """Parse a ``nocbench --dump-grid`` CSV into a :class:`Grid`.

    A dump may carry two, four or six columns of coordinates. ``phys_x`` /
    ``phys_y``, when present, are each core's own ``NOC_NODE_ID`` read by a
    probe kernel *on that core* -- the SoC-physical NoC 0 coordinate, which is
    the only space the link arithmetic is valid in. Older dumps have only the
    addressed coordinate, and :func:`_resolve_grid` then has to infer.
    """
    arch = None
    rows = []
    header = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            for token in line[1:].split():
                if token.startswith("arch="):
                    arch = token.split("=", 1)[1]
            continue
        cells = line.split(",")
        if header is None:
            header = cells
            continue
        v = [int(c) for c in cells]
        by_name = dict(zip(header, v))
        phys = None
        if "phys_x" in by_name and "phys_y" in by_name:
            phys = (by_name["phys_x"], by_name["phys_y"])
        rows.append(
            (
                (by_name["log_x"], by_name["log_y"]),
                (by_name["noc_x"], by_name["noc_y"]),
                phys,
            )
        )
    if arch not in SOC_GRID:
        raise PlanError(
            f"grid dump names arch={arch!r}; expected one of {sorted(SOC_GRID)}"
        )
    if not rows:
        raise PlanError(f"{path} lists no cores")
    noc_map = {log: noc for log, noc, _p in rows}
    # A device that does not answer NOC_NODE_ID stamps (0, 0) everywhere; that
    # is "unavailable", not "every core is at the origin", and is treated the
    # same as an old dump with no such column at all.
    reported = {log: p for log, _n, p in rows if p is not None and p != (0, 0)}
    phys_map = reported if len(reported) == len(rows) else None
    return _resolve_grid(arch, noc_map, phys_map)


def _resolve_grid(arch, noc_map, phys_map=None):
    """Decide which coordinate space a dump is in, and map it to physical.

    Three spaces occur in practice and they must not be confused, because the
    hop count of a route is a property of the physical torus while the space a
    kernel addresses is a *dense* renumbering of it. Getting this wrong would
    not fail loudly; it would quietly produce shared-link counts for a machine
    that does not exist -- which is exactly what happened on the first
    Blackhole card this was run on, a **harvested** part whose addressed
    columns ``{1..7, 10..14}`` are a subset of the unharvested physical worker
    columns ``{1..7, 10..16}`` and so passed the "looks physical" test below
    while actually being ``{1..7, 12..16}``. Four of the eight shared-link
    counts in that plan were wrong, and one hop count the experiment declared
    fixed was not. The order of the tests here is the fix: a dump that is short
    of an unharvested part's worker count is refused *before* anything is
    inferred from the coordinates it carries.
    """
    grid_x, grid_y = SOC_GRID[arch]
    cols, rows = WORKER_COLUMNS[arch], WORKER_ROWS[arch]
    xs = sorted({c[0] for c in noc_map.values()})
    ys = sorted({c[1] for c in noc_map.values()})

    if phys_map is not None:
        bad = sorted(
            c for c in phys_map.values() if c[0] not in cols or c[1] not in rows
        )
        if bad:
            raise PlanError(
                f"the probe kernels report physical coordinates {bad} that are not worker "
                f"positions on {arch} (columns {list(cols)}, rows {list(rows)}); the "
                f"WORKER_COLUMNS/WORKER_ROWS tables and this part disagree"
            )
        space = "noc0" if phys_map == noc_map else "translated"
        return Grid(arch, grid_x, grid_y, space, dict(noc_map), dict(phys_map))

    # No self-report. The dump is then only interpretable when NOTHING is
    # missing from it: a full house of worker columns and rows can only be the
    # physical layout or a dense renumbering of the whole of it, and both are
    # recoverable. Anything short of that is genuinely ambiguous -- the card
    # this was first run on dumped columns {1..7, 10..14}, which is a legal
    # SUBSET of the physical worker columns {1..7, 10..16} and also a dense
    # renumbering of {1..7, 12..16}, and it was the second. Nothing in the
    # coordinates distinguishes them, so nothing here tries.
    if set(xs) == set(cols) and set(ys) == set(rows):
        return Grid(arch, grid_x, grid_y, "noc0", dict(noc_map), dict(noc_map))
    if len(xs) == len(cols) and len(ys) == len(rows):
        # Dense renumbering of the complete worker set, in order.
        x_map = dict(zip(xs, cols))
        y_map = dict(zip(ys, rows))
        phys = {log: (x_map[nx], y_map[ny]) for log, (nx, ny) in noc_map.items()}
        return Grid(arch, grid_x, grid_y, "translated", dict(noc_map), phys)
    raise PlanError(
        f"grid dump lists {len(xs)}x{len(ys)} worker columns/rows (x={xs}, y={ys}) where an "
        f"unharvested {arch} has {len(cols)}x{len(rows)}, and carries no phys_x/phys_y "
        f"column. Something is missing -- harvesting, or a compute grid narrower than the "
        f"worker grid -- and the space a kernel addresses is renumbered around whatever went "
        f"without recording which. The physical link geometry therefore cannot be recovered "
        f"from these coordinates, and a shared-link count computed for the wrong geometry is "
        f"indistinguishable from a measurement. Re-dump with a nocbench new enough to probe "
        f"NOC_NODE_ID on each core (`--dump-grid` then writes phys_x/phys_y)."
    )


# ---------------------------------------------------------------------------
# Routing, and what a flow occupies.
# ---------------------------------------------------------------------------


def to_noc_space(coord, noc, grid_x, grid_y):
    """A physical NoC 0 coord in ``noc``'s own space.

    NoC 1's origin is the opposite corner, so a tile's NoC 1 coordinate is the
    mirror ``(grid_x - 1 - x, grid_y - 1 - y)`` -- the same convention
    :func:`tt_sim.network.tt_noc.noc_hop_count` documents. Mirroring both
    endpoints negates dx and dy, which is exactly the reversal of routing
    direction that distinguishes the two NoCs, so everything below works on
    either NoC with no special case.
    """
    if noc == 0:
        return coord
    return (grid_x - 1 - coord[0], grid_y - 1 - coord[1])


#: The router-to-router links a packet crosses, in order -- dimension-ordered X
#: then Y on a directional torus, with a link named by the router it leaves and
#: the axis it leaves on.
#:
#: **It is the simulator's own function**, not a copy of it. This module wrote
#: its own for a while, because a hop *order* was something only an experiment
#: planner needed: the network layer knew a hop count and charged a flight time,
#: and no link had an identity to get wrong. Since ``NocLinkRegistry`` gave links
#: identities the two have to agree exactly -- a shared-link count measured on
#: silicon describes a different machine from the modelled one the moment the
#: planner and the NoC disagree about which link is which -- so there is one
#: implementation and this is a name for it.
route_links = noc_route_links


#: Direction codes, matching ``perfbench/nocbench/src/kernels/nocbench_layout.h``.
#: ``DIR_BIDIR`` is refused by :func:`check_invariants` -- the kernel still
#: implements it, but no plan may contain it. See :func:`_refuse_unrunnable`.
DIR_WRITE, DIR_READ, DIR_BIDIR = 0, 1, 2

#: tt-metal's default unicast write virtual channel, from
#: ``tt_metal/hw/inc/internal/dataflow/dataflow_api_common.h``
#: (``#define NOC_UNICAST_WRITE_VC 1``; reads default to the same one). Every
#: flow in this harness that is not deliberately sweeping the VC uses it, so
#: that "the same VC" and "a different VC" are statements about the channel the
#: rest of the machine is on.
NOC_UNICAST_WRITE_VC = 1


@dataclass(frozen=True)
class Flow:
    """One master core moving data to or from one subordinate core."""

    master: tuple[int, int]  # SoC-physical NoC 0
    sub: tuple[int, int]
    direction: int = DIR_WRITE
    noc: int = 0
    vc: int = 0
    proc: int = 0
    num_tx: int = 64
    tx_bytes: int = 4096

    def legs(self, grid_x, grid_y):
        """``{name: (links, carries_payload)}`` for every packet this flow sends.

        A write is a payload-carrying request plus a one-flit acknowledgement
        coming back; a read is a one-flit request plus a payload-carrying
        response. Both legs occupy links, and both are counted -- the whole
        point of the shared-link experiment is that only ONE leg-pair overlap
        is allowed to move, so the ones that must stay at zero have to be
        computed rather than assumed away.
        """
        m = to_noc_space(self.master, self.noc, grid_x, grid_y)
        s = to_noc_space(self.sub, self.noc, grid_x, grid_y)
        out = {}
        if self.direction in (DIR_WRITE, DIR_BIDIR):
            out["write_req"] = (route_links(m, s, grid_x, grid_y), True)
            out["write_ack"] = (route_links(s, m, grid_x, grid_y), False)
        if self.direction in (DIR_READ, DIR_BIDIR):
            out["read_req"] = (route_links(m, s, grid_x, grid_y), False)
            out["read_resp"] = (route_links(s, m, grid_x, grid_y), True)
        return out

    def payload_links(self, grid_x, grid_y):
        links = set()
        for lk, payload in self.legs(grid_x, grid_y).values():
            if payload:
                links |= set(lk)
        return links

    def fwd_hops(self, grid_x, grid_y):
        """Hops on the request leg -- the direction the master addresses."""
        m = to_noc_space(self.master, self.noc, grid_x, grid_y)
        s = to_noc_space(self.sub, self.noc, grid_x, grid_y)
        return len(route_links(m, s, grid_x, grid_y))

    def rt_hops(self, grid_x, grid_y):
        """Hops there and back. On a directional torus this is `grid_x`,
        `grid_y` or `grid_x + grid_y` and nothing else."""
        m = to_noc_space(self.master, self.noc, grid_x, grid_y)
        s = to_noc_space(self.sub, self.noc, grid_x, grid_y)
        return len(route_links(m, s, grid_x, grid_y)) + len(
            route_links(s, m, grid_x, grid_y)
        )


def leg_overlap(a, b, grid_x, grid_y):
    """``{(a_leg, b_leg): shared_link_count}`` for two flows, all pairs."""
    la, lb = a.legs(grid_x, grid_y), b.legs(grid_x, grid_y)
    return {
        (na, nb): len(set(ka) & set(kb))
        for na, (ka, _pa) in la.items()
        for nb, (kb, _pb) in lb.items()
    }


def payload_overlap(a, b, grid_x, grid_y):
    return len(a.payload_links(grid_x, grid_y) & b.payload_links(grid_x, grid_y))


def other_overlap(a, b, grid_x, grid_y):
    """Shared links on every leg pair EXCEPT payload-with-payload."""
    la, lb = a.legs(grid_x, grid_y), b.legs(grid_x, grid_y)
    total = 0
    for _na, (ka, pa) in la.items():
        for _nb, (kb, pb) in lb.items():
            if pa and pb:
                continue
            total += len(set(ka) & set(kb))
    return total


# ---------------------------------------------------------------------------
# Points, plans and the invariant checker.
# ---------------------------------------------------------------------------


@dataclass
class Point:
    """One measurement: a set of concurrent flows, and what it is varying."""

    experiment: str
    label: str
    flows: list[Flow]
    #: Quantities this point is deliberately moving, for the invariant check to
    #: exempt and for the sweep to regress against.
    varying: dict = field(default_factory=dict)


#: The columns of a plan CSV, in order. ``perfbench/nocbench`` reads a subset by
#: name and copies the whole row through to its output, so the geometry that
#: produced a measurement travels with it.
PLAN_COLUMNS = (
    "run",
    "exp",
    "point",
    "flow",
    "n_flows",
    "mst_lx",
    "mst_ly",
    "mst_nx",
    "mst_ny",
    "mst_px",
    "mst_py",
    "sub_lx",
    "sub_ly",
    "sub_nx",
    "sub_ny",
    "sub_px",
    "sub_py",
    "proc",
    "noc",
    "vc",
    "direction",
    "num_tx",
    "tx_bytes",
    "fwd_hops",
    "rt_hops",
    "shared_payload_links",
    "shared_other_links",
)


def _fixed_signature(point, grid):
    """The quantities an experiment claims not to be moving, as one dict."""
    gx, gy = grid.grid_x, grid.grid_y
    flows = point.flows
    # Per-flow scalars are compared as SETS, not multisets: adding a second flow
    # with the same size and the same hop count must not read as "the size
    # moved". `n_flows` is the separate key that says how many there are, so
    # nothing is lost -- an experiment that gave two flows different sizes would
    # show up as a two-element set, which is still a change.
    return {
        "n_flows": len(flows),
        "tx_bytes": frozenset(f.tx_bytes for f in flows),
        "num_tx": frozenset(f.num_tx for f in flows),
        "direction": frozenset(f.direction for f in flows),
        "noc": frozenset(f.noc for f in flows),
        "vc": frozenset(f.vc for f in flows),
        "fwd_hops": frozenset(f.fwd_hops(gx, gy) for f in flows),
        "rt_hops": frozenset(f.rt_hops(gx, gy) for f in flows),
        "shared_payload_links": _shared_payload_total(flows, gx, gy),
        "shared_other_links": _shared_other_total(flows, gx, gy),
        "masters": frozenset(f.master for f in flows),
        "subs": frozenset(f.sub for f in flows),
    }


def _shared_payload_total(flows, gx, gy):
    return sum(
        payload_overlap(a, b, gx, gy) for a, b in itertools.combinations(flows, 2)
    )


def _shared_other_total(flows, gx, gy):
    return sum(other_overlap(a, b, gx, gy) for a, b in itertools.combinations(flows, 2))


def _refuse_unrunnable(point):
    """Refuse a point no card should be asked to run, whatever it would measure.

    The invariant checks below are about *identifiability*. These two are about
    the machine surviving the experiment, and they come from a Blackhole run
    that produced one of each.

    **Two flows on one master core.** A Tensix's two data-movement RISCs are
    separated by *NoC*, never by command buffer: ``BRISC_WR_CMD_BUF`` and
    ``NCRISC_WR_CMD_BUF`` are *both* command buffer 0 ("for large writes", same
    header, both architectures --
    ``tt_metal/hw/inc/internal/tt-1xx/<arch>/noc_nonblocking_api.h``). Two
    kernels issuing on one NoC therefore race on one set of ``NOC_TARG_ADDR`` /
    ``NOC_RET_ADDR`` / ``NOC_PACKET_TAG`` / ``NOC_CTRL`` registers, between the
    ``noc_cmd_buf_ready`` poll and the ``NOC_CTRL_SEND_REQ`` that arms them.
    The measurement is unusable even when it survives: ``noc_async_*_barrier``
    compares a per-NIU *hardware* counter against a per-RISC *software* one
    seeded from it at kernel init (``noc_local_state_init``), so with two
    issuers each RISC's ``==`` is satisfied by ANY N acks and both stop at the
    halfway point -- and an ``==`` against a monotonic counter that two issuers
    are advancing can be **overshot between polls**, which hangs. That was the
    retracted ``selfport`` control; ``plan_readport`` is the replacement.

    **A bidirectional flow.** On a Blackhole card the first and only
    ``DIR_BIDIR`` point never returned, while all 79 unidirectional flows in
    the same session completed. tt-sim runs the identical binary and plan to
    completion, so the root cause is something the simulator does not model and
    cannot be found from here; tt-metal's own ``core_bidirectional`` suite
    disables its whole directed-ideal family with ``GTEST_SKIP() << "Skipping
    test"; // Timeout issue (#36428)``. Refusing is not a diagnosis, it is the
    only responsible thing to emit while there is not one.
    """
    by_master = {}
    for f in point.flows:
        if f.direction == DIR_BIDIR:
            raise PlanError(
                f"experiment {point.experiment!r} point {point.label!r} asks for a "
                f"bidirectional flow ({f.master} <-> {f.sub}). That configuration hung a "
                f"Blackhole card and its cause is not established; no plan may contain it. "
                f"Use two unidirectional flows -- `plan_vc` shows how the virtual-channel "
                f"question is asked without one."
            )
        if f.master in by_master:
            raise PlanError(
                f"experiment {point.experiment!r} point {point.label!r} puts two flows on "
                f"master core {f.master}. A Tensix's two data-movement RISCs share command "
                f"buffer 0 (BRISC_WR_CMD_BUF == NCRISC_WR_CMD_BUF == 0), so two kernels "
                f"issuing on one NoC race on one set of NOC_TARG_ADDR / NOC_PACKET_TAG "
                f"registers and can hang the card; and their barriers compare a per-NIU "
                f"hardware ack counter against a per-RISC software one, so each stops after "
                f"any N acks and the measurement is blind whatever it reads. Put the second "
                f"flow on another core."
            )
        by_master[f.master] = f


def check_invariants(points, grid, varying_keys):
    """Raise unless exactly ``varying_keys`` differ across ``points``.

    This is the guard the whole exercise turns on. tt-metal's shipped dataset is
    unusable for congestion because three quantities move together in it; a plan
    that let two move would be the same mistake in a new file. So every quantity
    an experiment could plausibly confound is computed for each point, and any
    that differs and is not on the declared varying list is a hard error.

    :func:`_refuse_unrunnable` runs first and applies to every point on its own,
    including a single-point plan: those two refusals are about the card, not
    about the experiment, so they must not be skipped by the early return.
    """
    for point in points:
        _refuse_unrunnable(point)
    if len(points) < 2:
        return {}
    sigs = [_fixed_signature(p, grid) for p in points]
    moved = {k for k in sigs[0] if len({s[k] for s in sigs}) > 1}
    unexpected = moved - set(varying_keys)
    if unexpected:
        detail = "; ".join(
            f"{k}: {sorted(str(s[k]) for s in sigs)}" for k in sorted(unexpected)
        )
        raise PlanError(
            f"experiment {points[0].experiment!r} claims to vary {sorted(varying_keys)} but "
            f"these also moved across its points: {detail}. A plan in which two of "
            f"{{flow count, path length, shared links}} move together measures the same "
            f"unidentifiable thing tt-metal's shipped dataset does."
        )
    missing = set(varying_keys) - moved
    if missing:
        raise PlanError(
            f"experiment {points[0].experiment!r} declares it varies {sorted(missing)} but "
            f"those are constant across its points -- the sweep would have no axis to "
            f"regress against."
        )
    return {k: sorted(str(s[k]) for s in sigs) for k in sorted(moved)}


# ---------------------------------------------------------------------------
# The experiments.
# ---------------------------------------------------------------------------


def _workers(grid):
    """Physical worker coords present in the dump, as a set."""
    return set(grid.phys.values())


def plan_hops(
    grid, *, tx_bytes=4096, num_tx=64, direction=DIR_WRITE, noc=0, max_points=8
):
    """Experiment 1: one flow, subordinate swept; hop count is the axis."""
    gx, gy = grid.grid_x, grid.grid_y
    present = _workers(grid)
    cols = sorted({c[0] for c in present})
    rows = sorted({c[1] for c in present})
    master = (cols[0], rows[0])
    if master not in present:
        raise PlanError(
            "the dump's first worker column and row do not intersect at a worker"
        )

    points = []
    # Same row: round trip is grid_x however far apart the two cores are. The
    # spread here is the harness's noise floor.
    for x in cols[1 : 1 + max_points]:
        points.append(
            Point(
                "hops",
                f"row+{(x - master[0]) % gx}",
                [
                    Flow(
                        master,
                        (x, master[1]),
                        direction,
                        noc,
                        NOC_UNICAST_WRITE_VC,
                        0,
                        num_tx,
                        tx_bytes,
                    )
                ],
            )
        )
    # Same column: round trip is grid_y.
    for y in rows[1 : 1 + max_points]:
        points.append(
            Point(
                "hops",
                f"col+{(y - master[1]) % gy}",
                [
                    Flow(
                        master,
                        (master[0], y),
                        direction,
                        noc,
                        NOC_UNICAST_WRITE_VC,
                        0,
                        num_tx,
                        tx_bytes,
                    )
                ],
            )
        )
    # Both axes: round trip is grid_x + grid_y.
    for x, y in zip(cols[1 : 1 + max_points], rows[1 : 1 + max_points]):
        points.append(
            Point(
                "hops",
                f"diag+{x}-{y}",
                [
                    Flow(
                        master,
                        (x, y),
                        direction,
                        noc,
                        NOC_UNICAST_WRITE_VC,
                        0,
                        num_tx,
                        tx_bytes,
                    )
                ],
            )
        )
    check_invariants(points, grid, {"subs", "fwd_hops", "rt_hops"})
    levels = {p.flows[0].rt_hops(gx, gy) for p in points}
    if levels != {gx, gy, gx + gy}:
        raise PlanError(
            f"round-trip hop levels are {sorted(levels)}, expected {sorted({gx, gy, gx + gy})} -- "
            f"the directional-torus claim this experiment tests is not even expressible in "
            f"this plan"
        )
    return points


def plan_size(
    grid, *, sizes=(64, 512, 4096, 16384), num_tx=64, direction=DIR_WRITE, noc=0
):
    """Positive control: one flow, geometry frozen, transaction size swept."""
    present = _workers(grid)
    cols = sorted({c[0] for c in present})
    rows = sorted({c[1] for c in present})
    master, sub = (cols[0], rows[0]), (cols[1], rows[1])
    points = [
        Point(
            "size",
            f"{n}B",
            [Flow(master, sub, direction, noc, NOC_UNICAST_WRITE_VC, 0, num_tx, n)],
        )
        for n in sizes
    ]
    check_invariants(points, grid, {"tx_bytes"})
    return points


#: Minimum payload per flow for the ``readport`` control, in bytes. Below this
#: the timed region is dominated by the fixed round trip rather than by the
#: shared port, and a 1-vs-2 ratio cannot resolve the port at all -- which is
#: precisely how the retracted ``selfport`` control was sized (4 x 8 KiB, ~440
#: of 568 cycles being round trip). 64 KiB is 1024 cycles of injection at
#: Blackhole's 64 B/cycle against a ~260-cycle round trip: port-dominated by
#: about 4:1, which is the smallest ratio worth running.
READPORT_MIN_BYTES_PER_FLOW = 64 * 1024


def plan_readport(grid, *, tx_bytes=8192, num_tx=64, direction=DIR_READ, noc=0):
    """Positive control: one subordinate's injection port, read by one master then two.

    Replaces ``plan_selfport``, which was retracted -- see
    ``HYPOTHESES["readport"]`` and ``docs/plans/cost-model.md``. The two flows
    are on two *different* master cores, one data-movement RISC each, so every
    barrier counts only its own core's responses; they meet at the
    subordinate's single NIU, because a read's payload rides the return leg.
    """
    gx, gy = grid.grid_x, grid.grid_y
    if direction != DIR_READ:
        raise PlanError(
            "the readport control has to be a READ: a write's payload leaves the master, so "
            "two writers into one subordinate would share that subordinate's L1 write port "
            "rather than its injection port, which is a different resource"
        )
    if num_tx * tx_bytes < READPORT_MIN_BYTES_PER_FLOW:
        raise PlanError(
            f"readport would carry {num_tx * tx_bytes} B per flow; it needs at least "
            f"{READPORT_MIN_BYTES_PER_FLOW} B for the shared port rather than the fixed round "
            f"trip to dominate the timed region. Raise --num-tx or --readport-bytes. A control "
            f"sized below this reads the same whether or not the port serialises, which is how "
            f"the `selfport` control it replaces came to be retracted."
        )
    present = _workers(grid)
    cols = sorted({c[0] for c in present})
    rows = sorted({c[1] for c in present})
    sub = (cols[0], rows[0])
    # Both masters must differ from the subordinate on BOTH axes -- so every
    # round trip is grid_x + grid_y -- and be the same forward distance away, so
    # that adding the second flow adds no path length in either direction.
    off_axis = [c for c in sorted(present) if c[0] != sub[0] and c[1] != sub[1]]
    by_dist = {}
    for c in off_axis:
        by_dist.setdefault(len(route_links(c, sub, gx, gy)), []).append(c)
    # Among the equidistant pairs, take the one whose two RESPONSE paths share
    # the fewest router-to-router links. They cannot share none -- both leave
    # the subordinate's NIU on the same first hop, which is exactly why they
    # queue on its port -- but every link beyond the first is a mechanism this
    # control cannot tell from the port on silicon, so minimising them is free
    # precision. On a full Blackhole grid this takes the overlap from 13 to 1.
    best = None
    for d, candidates in sorted(by_dist.items()):
        for a, b in itertools.combinations(candidates, 2):
            fa = Flow(a, sub, direction, noc, NOC_UNICAST_WRITE_VC, 0, num_tx, tx_bytes)
            fb = Flow(b, sub, direction, noc, NOC_UNICAST_WRITE_VC, 0, num_tx, tx_bytes)
            overlap = payload_overlap(fa, fb, gx, gy)
            if best is None or (overlap, d) < best[0]:
                best = ((overlap, d), fa, fb)
    if best is None:
        raise PlanError(
            "no two off-axis masters at equal forward hop distance from the subordinate"
        )
    m0, m1 = best[1].master, best[2].master
    one = [
        Flow(
            m0,
            sub,
            direction,
            noc,
            NOC_UNICAST_WRITE_VC,
            0,
            num_tx,
            tx_bytes,
        )
    ]
    two = [
        Flow(
            m0,
            sub,
            direction,
            noc,
            NOC_UNICAST_WRITE_VC,
            0,
            num_tx,
            tx_bytes,
        ),
        Flow(
            m1,
            sub,
            direction,
            noc,
            NOC_UNICAST_WRITE_VC,
            0,
            num_tx,
            tx_bytes,
        ),
    ]
    points = [Point("readport", "1flow", one), Point("readport", "2flows", two)]
    # The flow count and the second master's identity move. The two response
    # paths leave one tile, so they necessarily share the links next to it --
    # declared, not hidden, and stated in HYPOTHESES["readport"] as the reason
    # this control proves contention without attributing it.
    check_invariants(
        points,
        grid,
        {"n_flows", "masters", "shared_payload_links", "shared_other_links"},
    )
    for p in points:
        if {f.rt_hops(gx, gy) for f in p.flows} != {gx + gy}:
            raise PlanError(
                "readport masters are not all a full grid_x + grid_y round trip away"
            )
        if len({f.proc for f in p.flows}) != 1 or p.flows[0].proc != 0:
            raise PlanError("readport must use one data-movement RISC per core")
    if payload_overlap(two[0], two[1], gx, gy) < 1:
        raise PlanError(
            "the two readport response paths share no link, so they cannot be leaving one "
            "NIU -- the geometry search picked masters the subordinate reaches by different "
            "first hops"
        )
    return points


def _search_shared_placement(grid, *, num_tx=64, direction=DIR_WRITE, noc=0, want=8):
    """``(score, flow_a, {shared_links: flow_b})`` -- the searched two-flow geometry.

    The placement is SEARCHED, not written down, because the constraint that
    matters -- every leg-pair overlap except the payload one stays at zero -- is
    easier to check than to solve. See :data:`HYPOTHESES` for the geometry the
    search is exploiting. Shared by experiment 2 (which sweeps the overlap) and
    experiment 4 (which pins the overlap at 1 and sweeps a virtual channel), so
    the two are the same geometry with one axis swapped.
    """
    gx, gy = grid.grid_x, grid.grid_y
    present = _workers(grid)
    cols = sorted({c[0] for c in present})
    rows = sorted({c[1] for c in present})
    if len(rows) < 3:
        raise PlanError(
            "need at least three worker rows to keep the acknowledgement paths apart"
        )

    best = None
    for y0, y1, y2 in itertools.permutations(rows[:4], 3):
        for ax in cols:
            if (ax, y0) not in present:
                continue
            for d in range(1, gx):
                a_sub = ((ax + d) % gx, y1)
                if a_sub not in present:
                    continue
                a = Flow(
                    (ax, y0), a_sub, direction, noc, NOC_UNICAST_WRITE_VC, 0, num_tx, 0
                )
                by_overlap = {}
                for bx in cols:
                    if bx == ax or (bx, y0) not in present:
                        continue
                    b_sub = ((bx + d) % gx, y2)
                    if b_sub not in present:
                        continue
                    b = Flow(
                        (bx, y0),
                        b_sub,
                        direction,
                        noc,
                        NOC_UNICAST_WRITE_VC,
                        0,
                        num_tx,
                        0,
                    )
                    if len({a.master, a.sub, b.master, b.sub}) != 4:
                        continue
                    if other_overlap(a, b, gx, gy) != 0:
                        continue
                    ov = payload_overlap(a, b, gx, gy)
                    by_overlap.setdefault(ov, b)
                if len(by_overlap) < 2 or 0 not in by_overlap:
                    continue
                score = (len(by_overlap), max(by_overlap))
                if best is None or score > best[0]:
                    best = (score, a, by_overlap)
                if len(by_overlap) >= want and max(by_overlap) >= want - 1:
                    # A contiguous run of `want` overlap values starting at zero
                    # is everything the sweep can use; searching on would only
                    # relabel the same experiment.
                    return best
    if best is None:
        raise PlanError(
            "no placement found in which two flows can be slid to share 0, 1, 2 ... links "
            "while every other leg-pair overlap stays at zero. This is a property of the "
            "worker grid in the dump, not of the request."
        )
    return best


def plan_shared(
    grid, *, tx_bytes=(64, 16384), num_tx=64, direction=DIR_WRITE, noc=0, max_points=8
):
    """Experiment 2: two flows, N fixed at 2, payload-link overlap swept."""
    best = _search_shared_placement(
        grid, num_tx=num_tx, direction=direction, noc=noc, want=max_points
    )
    return _shared_points(best, grid, tx_bytes, num_tx, direction, noc, max_points)


def _shared_points(best, grid, tx_bytes, num_tx, direction, noc, max_points):
    """Turn a searched placement into points, and re-assert what it claims."""
    _score, a, by_overlap = best
    overlaps = sorted(by_overlap)[:max_points]

    points = []
    for size in sorted(tx_bytes):
        for ov in overlaps:
            b = by_overlap[ov]
            # Both flows on the FIRST data-movement RISC of their own core. They
            # are on different cores, so nothing forces them apart, and using
            # one processor for both removes a confound the earlier version
            # carried: BRISC and NCRISC do not run the same issue loop.
            flows = [
                Flow(
                    a.master,
                    a.sub,
                    direction,
                    noc,
                    NOC_UNICAST_WRITE_VC,
                    0,
                    num_tx,
                    size,
                ),
                Flow(
                    b.master,
                    b.sub,
                    direction,
                    noc,
                    NOC_UNICAST_WRITE_VC,
                    0,
                    num_tx,
                    size,
                ),
            ]
            points.append(
                Point(
                    "shared", f"{size}B/share{ov}", flows, {"shared_payload_links": ov}
                )
            )
    # Checked per size, because size is a separate (declared) axis.
    for size in sorted(tx_bytes):
        group = [p for p in points if p.flows[0].tx_bytes == size]
        check_invariants(group, grid, {"shared_payload_links", "masters", "subs"})
        # `masters`/`subs` move by construction -- sliding flow B IS the
        # experiment -- so pin the parts that must not: flow A is identical at
        # every point, and B's hop counts never change.
        first = group[0].flows[0]
        for p in group:
            if (p.flows[0].master, p.flows[0].sub) != (first.master, first.sub):
                raise PlanError("flow A moved between points; only flow B may slide")
            if p.flows[1].fwd_hops(grid.grid_x, grid.grid_y) != group[0].flows[
                1
            ].fwd_hops(grid.grid_x, grid.grid_y):
                raise PlanError("flow B's forward hop count moved between points")
    return points


def plan_contention(
    grid,
    *,
    tx_bytes=16384,
    num_tx=64,
    direction=DIR_WRITE,
    noc=0,
    counts=(1, 2, 3, 4, 6),
):
    """Experiment 3: N flows into one subordinate, every master equidistant."""
    gx, gy = grid.grid_x, grid.grid_y
    present = _workers(grid)
    sub = sorted(present)[len(present) // 2]
    # Masters at one forward distance from `sub`, and differing from it on both
    # axes so every round trip is grid_x + grid_y.
    by_dist = {}
    for c in sorted(present):
        if c == sub or c[0] == sub[0] or c[1] == sub[1]:
            continue
        d = len(route_links(c, sub, gx, gy))
        by_dist.setdefault(d, []).append(c)
    usable = {d: ms for d, ms in by_dist.items() if len(ms) >= max(counts)}
    if not usable:
        raise PlanError(
            f"no forward distance from {sub} has {max(counts)} equidistant masters differing "
            f"on both axes; reduce --contention-counts"
        )
    dist = min(usable)
    masters = usable[dist][: max(counts)]

    points = []
    for n in counts:
        flows = [
            Flow(m, sub, direction, noc, NOC_UNICAST_WRITE_VC, 0, num_tx, tx_bytes)
            for m in masters[:n]
        ]
        points.append(Point("contention", f"N{n}", flows, {"n_flows": n}))
    # `shared_payload_links` is DECLARED to move: N flows into one endpoint
    # must share the links next to it. That is the honest limit of this
    # experiment, not a defect in the plan -- see HYPOTHESES["contention"].
    check_invariants(
        points,
        grid,
        {"n_flows", "masters", "shared_payload_links", "shared_other_links"},
    )
    for p in points:
        hops = {f.fwd_hops(gx, gy) for f in p.flows}
        if len(hops) != 1 or hops != {dist}:
            raise PlanError(
                f"masters are not equidistant from the subordinate: {sorted(hops)}"
            )
        if {f.sub for f in p.flows} != {sub}:
            raise PlanError("contention points do not share one subordinate")
    return points


def plan_vc(grid, *, tx_bytes=16384, num_tx=64, noc=0):
    """Experiment 4: two writers sharing ONE link, the second's write VC swept.

    Unidirectional by construction. The previous design -- one master reading
    from and writing to one subordinate at once -- hung a Blackhole card, and
    :func:`_refuse_unrunnable` now rejects it outright; see
    ``HYPOTHESES["vc"]`` for the evidence and for what this asks instead.
    """
    gx, gy = grid.grid_x, grid.grid_y
    best = _search_shared_placement(
        grid, num_tx=num_tx, direction=DIR_WRITE, noc=noc, want=2
    )
    _score, a, by_overlap = best
    non_zero = sorted(ov for ov in by_overlap if ov > 0)
    if not non_zero:
        raise PlanError(
            "the searched placement offers no point at which the two writers share a link, so "
            "there is nothing for a virtual channel to arbitrate"
        )
    b = by_overlap[non_zero[0]]
    shared_links = non_zero[0]

    points = [
        Point(
            "vc",
            f"vc{vc}",
            [
                Flow(
                    a.master,
                    a.sub,
                    DIR_WRITE,
                    noc,
                    NOC_UNICAST_WRITE_VC,
                    0,
                    num_tx,
                    tx_bytes,
                ),
                Flow(b.master, b.sub, DIR_WRITE, noc, vc, 0, num_tx, tx_bytes),
            ],
            {"vc": vc},
        )
        for vc in range(4)
    ]
    # `vc` is the ONLY thing that moves: same two masters, same two
    # subordinates, same hop counts, same shared-link count, same size, same
    # count, same NoC, same processor. That is a stronger claim than the
    # retracted bidirectional version could make, and it is checked rather than
    # asserted.
    check_invariants(points, grid, {"vc"})
    for p in points:
        if _shared_payload_total(p.flows, gx, gy) != shared_links:
            raise PlanError("the vc points do not all share the same number of links")
        if other_overlap(p.flows[0], p.flows[1], gx, gy) != 0:
            raise PlanError(
                "a vc point has a non-payload leg overlap, so the two writers meet somewhere "
                "other than the one link this experiment is about"
            )
    return points


EXPERIMENTS = {
    "hops": plan_hops,
    "size": plan_size,
    "readport": plan_readport,
    "shared": plan_shared,
    "contention": plan_contention,
    "vc": plan_vc,
}

#: The two the cost model actually needs, per docs/plans/cost-model.md: an
#: intercept and a slope. `size` and `readport` are the controls that make a
#: null reading in `shared` interpretable, so they are not optional either.
MINIMUM = ("hops", "size", "readport", "shared")


# ---------------------------------------------------------------------------
# Emitting.
# ---------------------------------------------------------------------------


def plan_rows(points, grid, first_run=0):
    """``[dict]``, one per flow, ready to be written as a plan CSV."""
    gx, gy = grid.grid_x, grid.grid_y
    by_phys = grid.logical_by_phys
    rows = []
    run = first_run
    for point in points:
        for i, f in enumerate(point.flows):
            if f.master not in by_phys or f.sub not in by_phys:
                raise PlanError(f"flow {f} names a core the grid dump does not contain")
            mlog, slog = by_phys[f.master], by_phys[f.sub]
            others = [o for o in point.flows if o is not f]
            rows.append(
                {
                    "run": run,
                    "exp": point.experiment,
                    "point": point.label,
                    "flow": i,
                    "n_flows": len(point.flows),
                    "mst_lx": mlog[0],
                    "mst_ly": mlog[1],
                    "mst_nx": grid.noc[mlog][0],
                    "mst_ny": grid.noc[mlog][1],
                    "mst_px": f.master[0],
                    "mst_py": f.master[1],
                    "sub_lx": slog[0],
                    "sub_ly": slog[1],
                    "sub_nx": grid.noc[slog][0],
                    "sub_ny": grid.noc[slog][1],
                    "sub_px": f.sub[0],
                    "sub_py": f.sub[1],
                    "proc": f.proc,
                    "noc": f.noc,
                    "vc": f.vc,
                    "direction": f.direction,
                    "num_tx": f.num_tx,
                    "tx_bytes": f.tx_bytes,
                    "fwd_hops": f.fwd_hops(gx, gy),
                    "rt_hops": f.rt_hops(gx, gy),
                    "shared_payload_links": sum(
                        payload_overlap(f, o, gx, gy) for o in others
                    ),
                    "shared_other_links": sum(
                        other_overlap(f, o, gx, gy) for o in others
                    ),
                }
            )
        run += 1
    return rows


def write_plan(path, rows, grid, header_notes=()):
    lines = [
        "# nocbench plan -- built by tt_sim.perf.noc_congestion_plan",
        f"# arch={grid.arch} grid_x={grid.grid_x} grid_y={grid.grid_y} coord_space={grid.coord_space}",
        "# *_lx/_ly logical, *_nx/_ny the coordinate a kernel addresses, *_px/_py SoC-physical NoC 0",
        f"# tt_sim_tensix_coords={tensix_coords(rows)}",
    ]
    lines += [f"# {n}" for n in header_notes]
    lines.append(",".join(PLAN_COLUMNS))
    for r in rows:
        lines.append(",".join(str(r[c]) for c in PLAN_COLUMNS))
    Path(path).write_text("\n".join(lines) + "\n")


def tensix_coords(rows):
    """The ``TT_SIM_TENSIX_COORDS`` value a simulator run of this plan needs.

    tt-sim materialises only the worker tiles named in that variable and warns
    (loudly) about traffic to any other, so a plan that touches nine cores needs
    all nine listed. The coordinates are the ones the wire carries, which is the
    ``*_nx/_ny`` pair.
    """
    coords = sorted(
        {(r["mst_nx"], r["mst_ny"]) for r in rows}
        | {(r["sub_nx"], r["sub_ny"]) for r in rows}
    )
    return ",".join(f"{x}-{y}" for x, y in coords)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Not `required=True`: `--hypotheses` prints the pre-declared predictions
    # and needs no card, and the README tells people to run it that way.
    ap.add_argument("--grid", help="a `nocbench --dump-grid` CSV")
    ap.add_argument("--out", default="nocbench-plan.csv")
    ap.add_argument(
        "--experiments",
        default=",".join(MINIMUM),
        help=f"comma-separated, from {sorted(EXPERIMENTS)}, or 'all'",
    )
    ap.add_argument(
        "--num-tx", type=int, default=64, help="transactions per flow per point"
    )
    ap.add_argument(
        "--tx-bytes", type=int, default=4096, help="bytes per transaction, where fixed"
    )
    ap.add_argument("--noc", type=int, default=0, choices=(0, 1))
    ap.add_argument(
        "--readport-bytes",
        type=int,
        default=8192,
        help="bytes per transaction for the injection-port control; it only "
        "works when the port rather than the fixed round trip is the "
        "bottleneck, so --num-tx times this must be at least "
        f"{READPORT_MIN_BYTES_PER_FLOW} B",
    )
    ap.add_argument(
        "--sizes",
        default="64,512,4096,16384",
        help="transaction sizes for the `size` control",
    )
    ap.add_argument(
        "--shared-sizes",
        default="64,16384",
        help="transaction sizes at which the shared-link sweep is repeated; the "
        "small one is a negative control (idle links) and the large one is where "
        "an effect can show",
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=8,
        help="cap on points per family; lower it to make a simulator run tractable",
    )
    ap.add_argument(
        "--hypotheses",
        action="store_true",
        help="print the pre-declared predictions and exit",
    )
    args = ap.parse_args(argv)

    if args.hypotheses:
        for name in EXPERIMENTS:
            print(HYPOTHESES[name])
            print()
        return 0

    if not args.grid:
        ap.error("--grid is required to build a plan (see --hypotheses)")
    grid = load_grid(args.grid)
    names = (
        sorted(EXPERIMENTS)
        if args.experiments == "all"
        else args.experiments.split(",")
    )
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(
            f"unknown experiment(s) {unknown}; have {sorted(EXPERIMENTS)}",
            file=sys.stderr,
        )
        return 2

    rows, notes, run = [], [], 0
    for name in names:
        kwargs = {"noc": args.noc, "num_tx": args.num_tx}
        if name in ("hops", "contention"):
            kwargs["tx_bytes"] = args.tx_bytes
        if name == "readport":
            kwargs["tx_bytes"] = args.readport_bytes
        if name == "size":
            kwargs["sizes"] = tuple(int(v) for v in args.sizes.split(","))
        if name == "shared":
            kwargs["tx_bytes"] = tuple(int(v) for v in args.shared_sizes.split(","))
        if name in ("hops", "shared"):
            kwargs["max_points"] = args.max_points
        try:
            points = EXPERIMENTS[name](grid, **kwargs)
        except PlanError as exc:
            print(f"cannot plan {name!r}: {exc}", file=sys.stderr)
            return 1
        new = plan_rows(points, grid, first_run=run)
        run += len(points)
        rows += new
        notes.append(f"{name}: {len(points)} points, {len(new)} flows")

    write_plan(args.out, rows, grid, notes)
    print(
        f"wrote {args.out}: {len(rows)} flows over {run} runs ({grid.arch}, {grid.coord_space} coords)"
    )
    print(f"  {'; '.join(notes)}")
    print(f"  TT_SIM_TENSIX_COORDS={tensix_coords(rows)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
