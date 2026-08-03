"""Turning the cycle-cost tables into occupancy the simulator can charge.

This is the consuming half of :mod:`tt_sim.perf.costs` — Phase 5 of
``docs/plans/event-driven-pump.md``, and the first thing in the tree to read
the tables at all. It answers one question per instruction: *how many cycles
does this op occupy its unit for?* The answer is then handed to
:meth:`tt_sim.pe.tensix.backends.backend_base.TensixBackendUnit.occupy_for`,
which Phase 4 left as the socket for exactly this.

**Opt-in.** Nothing here runs unless ``TT_SIM_COST_MODEL`` is truthy. With it
unset every unit's :attr:`cost_model` is ``None``, no table is loaded, and not
one cycle of any existing run changes — which is not a nicety but the
condition of the project's validation strategy: byte-identical replay and a
pinned matmul PCC are how timing regressions are caught, and a cost model that
moved them silently would have burned the instrument it is measured with.

Three policies live here rather than at the call sites, because
``docs/plans/cost-model.md`` is explicit that a consumer should make each of
them once, deliberately:

1. **No entry means no opinion.** An opcode the tables do not cost, or cost
   with no ``occupancy`` field, gets ``None`` — not a silently invented
   1 cycle. ``None`` leaves the same-cycle retire exactly as it is.
2. **A bound is not an equals sign.** ``ATCAS`` is *at least* 15 cycles and
   ``ADDDMAREG`` is "3 or 4"; the model charges the *low* end of both and
   :func:`modelled_occupancy` says so, so a modelled cycle count is a floor
   wherever bounded entries are involved, never a claim of exactness. See
   :data:`BOUND_POLICY`.
3. **Derived is not measured.** Most Tensix occupancies are ``1 /
   throughput_ipc`` rather than a published occupancy column
   (``isa_doc_derived``). Those are charged — arithmetic on a documented
   number is the best available — but :meth:`UnitCostModel.provenance_of`
   keeps the rank reachable so a report can say what a number is worth.
   ``unknown`` and ``estimated`` entries are never charged.
"""

from __future__ import annotations

import math
import os

from tt_sim.perf.costs import SOURCED_PROVENANCE, CycleCost, load_costs

#: How each :class:`~tt_sim.perf.costs.CycleCost` bound turns into the single
#: integer ``occupy_for`` needs. ``at_least`` / ``range`` charge the low end,
#: which makes the model a lower bound on occupancy: the honest direction for
#: an estimator, because over-charging a unit invents back-pressure that the
#: hardware does not have, while under-charging only fails to model a stall
#: that was already not modelled at all.
BOUND_POLICY = {
    "exact": "as stated",
    "at_least": "low end (the model is a floor)",
    "at_most": "as stated (an upper bound charged in full)",
    "approximate": "as stated",
    "range": "low end of the range (the model is a floor)",
}

#: Bounds whose charged value is not a claim about the exact cycle count.
INEXACT_BOUNDS = frozenset({"at_least", "at_most", "approximate", "range"})

_ENV_VAR = "TT_SIM_COST_MODEL"


def _truthy(raw, default=False):
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def cost_model_enabled(env=None):
    """True when ``TT_SIM_COST_MODEL`` opts this run into modelled occupancy.

    Read once, when a unit is constructed, so a driver script that wants the
    model on for part of a run should build its device with the variable set
    rather than toggling it mid-flight.
    """
    return _truthy((env if env is not None else os.environ).get(_ENV_VAR))


def modelled_occupancy(cost: CycleCost | None) -> int | None:
    """The integer number of cycles to charge for ``cost``, or ``None``.

    ``None`` in, ``None`` out: an instruction the tables do not cost keeps the
    simulator's existing same-cycle retire. A fractional cost rounds *up* (a
    unit cannot be busy for two thirds of a cycle), and a bounded cost is
    resolved by :data:`BOUND_POLICY` — never by pretending the bound was not
    written.
    """
    if cost is None:
        return None
    cycles = cost.cycles
    if cost.bound not in BOUND_POLICY:
        raise ValueError(f"no occupancy policy for bound {cost.bound!r}")
    return int(math.ceil(cycles))


class UnitCostModel:
    """Modelled occupancy for one Tensix backend unit's opcodes.

    Built once per unit from :func:`~tt_sim.perf.costs.load_costs`, so a lookup
    on the instruction path is a dict hit rather than a YAML walk. Instances
    are immutable in practice and safe to share between units of the same kind.
    """

    def __init__(self, unit_costs, arch):
        self.arch = arch
        self.unit_name = unit_costs.name
        self._costs = unit_costs
        self._occupancy = {}
        self._inexact = set()
        for name, entry in unit_costs.instructions.items():
            if entry.provenance not in SOURCED_PROVENANCE:
                # ``unknown`` / ``estimated`` carry no numbers worth charging.
                continue
            cycles = modelled_occupancy(entry.occupancy)
            if cycles is None:
                continue
            self._occupancy[name] = cycles
            if entry.occupancy.bound in INEXACT_BOUNDS:
                self._inexact.add(name)
        self._fidelity_scaled = frozenset(
            name
            for name, entry in unit_costs.instructions.items()
            if entry.scales_with == "fidelity_phases"
        )
        self._fidelity = unit_costs.extras.get("fidelity_phases")

    # -- per-instruction ---------------------------------------------------
    def occupancy(self, instruction_name):
        """Cycles ``instruction_name`` occupies the unit, or ``None``."""
        return self._occupancy.get(instruction_name)

    def is_exact(self, instruction_name):
        """False when the charged number came from a ``>=``, ``~`` or range."""
        return instruction_name not in self._inexact

    def provenance_of(self, instruction_name):
        entry = self._costs.instructions.get(instruction_name)
        return entry.provenance if entry is not None else None

    # -- fidelity phases ---------------------------------------------------
    def scales_with_fidelity(self, instruction_name):
        """True for the ops the table marks ``scales_with: fidelity_phases``."""
        return instruction_name in self._fidelity_scaled

    def fidelity_occupancy(self, phase_index):
        """Occupancy of one fidelity-scaled op issued at fidelity phase ``phase_index``.

        The table's flat ``occupancy: 1`` for ``MVMUL`` cannot be the whole
        story — the ISA docs footnote the throughput column with "if multiple
        fidelity phases are in use, then one instruction is required per
        fidelity phase, so the effective IPC decreases" — so this is the one
        cost in the table that has to be computed rather than looked up.

        The computation, entirely from the unit's ``fidelity_phases`` block:

        * ``phase_index`` is the *index* of this instruction's phase (0-3), so
          at least ``phase_index + 1`` phases are in use. That is all a single
          instruction can tell us, and it is exact for the kernels that walk
          the phases in order, which is what tt-metal's matmul MOP emits.
        * ``cycles_per_tile`` gives the whole-tile cost at that phase count
          (LoFi 16, HiFi2 32, HiFi3 48, HiFi4 64).
        * ``mvmuls_per_tile.count`` gives how many instructions one phase of a
          32x32x32 tile matmul takes (16), so the tile needs
          ``phases * count`` instructions in total.

        Dividing gives the per-instruction occupancy. **It comes out 1 at every
        fidelity phase**, and that is the substantive result rather than a
        degenerate one: the fidelity multiplier is already carried by the
        instruction *stream* (four phases means four MVMULs), so scaling each
        instruction by the phase count as well would charge a HiFi4 matmul 2.5x
        what the hardware costs. It is also an independent corroboration of the
        ISA docs' ``throughput_ipc: 1`` from a vendor source that never
        mentions IPC — rung 1 of ``docs/plans/cost-model.md``'s calibration
        ladder, available without silicon.

        Returns ``None`` when the unit has no ``fidelity_phases`` block or the
        phase count is not tabulated, which falls back to the flat entry.
        """
        block = self._fidelity
        if block is None:
            return None
        phases_in_use = phase_index + 1
        phase_name = None
        for name, count in (block.get("phases") or {}).items():
            if count == phases_in_use:
                phase_name = name
                break
        if phase_name is None:
            return None
        cycles_per_tile = (block.get("cycles_per_tile") or {}).get(phase_name)
        per_phase = (block.get("mvmuls_per_tile") or {}).get("count")
        if not cycles_per_tile or not per_phase:
            return None
        return int(math.ceil(cycles_per_tile / (phases_in_use * per_phase)))


_UNIT_MODELS = {}


def unit_cost_model(unit_name, arch):
    """The :class:`UnitCostModel` for ``unit_name`` on ``arch``, cached.

    Returns ``None`` when the model is switched off, so a caller can write
    ``self.cost_model = unit_cost_model("MATH", arch)`` and have the disabled
    case cost one attribute store.
    """
    if not cost_model_enabled():
        return None
    key = (unit_name, arch)
    if key not in _UNIT_MODELS:
        _UNIT_MODELS[key] = UnitCostModel(load_costs(arch).unit(unit_name), arch)
    return _UNIT_MODELS[key]
