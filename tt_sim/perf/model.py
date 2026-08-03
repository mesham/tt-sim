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


# ---------------------------------------------------------------------------
# The baby RISC-V cores.
# ---------------------------------------------------------------------------
#
# Everything above this line is the Tensix coprocessor, whose cost is a
# per-opcode occupancy. The baby cores are a different shape and the tables say
# so: ``riscv.load_latency`` is a *latency* table keyed by address region, and
# the ISA docs are explicit about what that means —
#
#   "A latency of N cycles means that N - 1 independent instructions need to
#    follow the load if the latency is to be entirely hidden."
#
# So a load's cost is not occupancy at all. It is a scoreboard entry: the core
# is single-issue and in-order, it keeps running after the load, and it stalls
# only when something *reads the loaded register* too soon. That is the same
# distinction the SFPU forced (latency 2 is time-to-result, not time-held), read
# off the opposite kind of table, and it is what ROADMAP section I asks for by
# name: "memory-stall back-pressure on L1 / NoC reads".
#
# What *is* occupancy for these cores is the throughput half — "at most one
# store to L1 every five cycles" — and the integer unit's own multiply/divide
# blocking. Those are charged as occupancy; the load table is charged as a
# scoreboard.

#: Canonical address regions of the load-latency table. Integers rather than
#: strings because this is indexed once per RISC-V load, on the hottest path in
#: the simulator. The arch-specific YAML keys are mapped onto them below —
#: Wormhole and Blackhole group the regions differently (Blackhole moves the
#: TDMA registers from the ">= 7" row to the ">= 4" row) and the consumer
#: should not have to know that.
RV_REGION_L1 = 0
RV_REGION_LOCAL_DATA_RAM = 1
RV_REGION_MAILBOX_GROUP = 2
RV_REGION_TENSIX_GPR_CFG = 3
RV_REGION_TDMA = 4
RV_REGION_TILECTRL_PIC_OVERLAY = 5
#: An address the load-latency table does not name. Charged nothing — see
#: :data:`RV_UNNAMED_REGIONS` for which blocks these are and why that matters.
RV_REGION_UNNAMED = 6
RV_REGION_COUNT = 7

RV_REGION_NAMES = (
    "l1",
    "local_data_ram",
    "mailbox_group",
    "tensix_gpr_cfg",
    "tdma",
    "tilectrl_pic_overlay",
    "unnamed",
)

#: The MMIO blocks tt-sim maps into a baby core's address space that the ISA
#: docs' load-latency table does not have a row for, so nothing can be charged
#: for them without inventing a number. Documented here rather than in a
#: comment because one of them is load-bearing: the **NoC NIU registers**
#: (0xFFB20000 / 0xFFB30000) are the target of every ``noc_async_*_barrier``
#: poll in every tt-metal dataflow kernel, i.e. the busiest MMIO load in the
#: tree, and the table's ">= 7" row names "TDMA / tile control / PIC / NoC
#: *overlay*" — a different block. Charging the overlay's cost to the NIUs
#: would be a guess dressed up as a citation.
RV_UNNAMED_REGIONS = (
    "noc niu registers (0xFFB20000 / 0xFFB30000)",
    "mop expander config (0xFFB80000)",
    "riscv instruction ram (0xFFC00000)",
    "tensix instruction push buffers (0xFFE40000-0xFFE60000)",
)

#: Which ``riscv.load_latency`` key supplies each canonical region, per arch.
#: Blackhole's table is not a superset of Wormhole's — it renames rows, splits
#: L1 into a d-cache hit and a miss, and moves TDMA between groups — so the
#: mapping is spelled out per arch rather than guessed by name similarity.
_LOAD_LATENCY_KEYS = {
    "wormhole": {
        RV_REGION_L1: "l1",
        RV_REGION_LOCAL_DATA_RAM: "core_local_data_ram",
        RV_REGION_MAILBOX_GROUP: "mailboxes_pcbufs_ttsync_semaphores",
        RV_REGION_TENSIX_GPR_CFG: "tensix_gprs_and_backend_config",
        RV_REGION_TDMA: "tdma_tilectrl_pic_noc_overlay",
        RV_REGION_TILECTRL_PIC_OVERLAY: "tdma_tilectrl_pic_noc_overlay",
    },
    "blackhole": {
        # Blackhole has an L0 data cache in front of L1, so the table gives two
        # numbers: 2 on a hit, >= 8 on a miss. tt-sim models no such cache and
        # no hit rate is published anywhere, so the pair is treated exactly
        # like the file's other two-ended costs and charged at its **low end**
        # — the same "a modelled count is a floor" policy as `at_least` and
        # `range`. Charging the miss instead would invent a 100 % miss rate,
        # which is both unsourced and the over-charging direction.
        RV_REGION_L1: "l1_dcache_hit",
        RV_REGION_LOCAL_DATA_RAM: "core_local_data_ram",
        RV_REGION_MAILBOX_GROUP: "mailboxes_pcbufs_ttsync_semaphores",
        RV_REGION_TENSIX_GPR_CFG: "tensix_gprs_backend_config_tdma",
        RV_REGION_TDMA: "tensix_gprs_backend_config_tdma",
        RV_REGION_TILECTRL_PIC_OVERLAY: "tilectrl_pic_noc_overlay",
    },
}


def _sourced_cycles(raw, provenance):
    """Cycles to charge for one raw YAML cost value, or ``None``.

    The same three policies as :class:`UnitCostModel`, applied to the plain
    mappings in ``unit_costs.yaml`` (which the loader leaves as raw dicts
    rather than turning into :class:`~tt_sim.perf.costs.CostEntry`): an
    ``unknown`` / ``estimated`` block is charged nothing whatever numbers it
    carries, and a bound is resolved by :data:`BOUND_POLICY`.
    """
    if provenance not in SOURCED_PROVENANCE:
        return None
    return modelled_occupancy(CycleCost.parse(raw))


class RiscvCostModel:
    """What the tables say a baby RISC-V core's memory path costs.

    Three separate things, because the ISA docs publish three separate things:

    * :attr:`load_latency` — cycles from a load issuing to its destination
      register being readable, by address region. A *scoreboard* input, not an
      occupancy: the core keeps issuing and only stalls on a dependent read.
    * :attr:`l1_store_period` / :attr:`l1_coalesced_store_period` — the
      sustained store rate, which *is* an occupancy of the load/store unit.
    * :attr:`multiply` / the divide costs — the integer unit's own blocking,
      the one place the docs say outright that "the next instruction cannot
      enter the unit until the multiply instruction has finished".

    Anything the tables do not name is ``None`` and is charged nothing.
    """

    def __init__(self, sections, arch):
        self.arch = arch
        riscv = sections["riscv"]
        self._table = riscv.get("load_latency") or {}
        self.load_latency = self._load_latencies(riscv, arch)
        #: Cycles a dependent instruction stalls when it issues in the cycle
        #: right after the load. ``latency - 1``, which is the docs' own
        #: statement of the table ("N - 1 independent instructions ... to be
        #: entirely hidden") turned round.
        self.load_use_stall = tuple(
            None if c is None else max(c - 1, 0) for c in self.load_latency
        )
        stores = riscv.get("store_throughput") or {}
        store_prov = stores.get("provenance")
        self.l1_store_period = _sourced_cycles(
            stores.get("l1_period_cycles"), store_prov
        )
        self.l1_coalesced_store_period = _sourced_cycles(
            stores.get("l1_coalesced_period_cycles"), store_prov
        )
        self.other_store_period = _sourced_cycles(
            stores.get("other_regions_period_cycles"), store_prov
        )
        integer = riscv.get("integer_unit") or {}
        int_prov = integer.get("provenance")
        self.multiply = _sourced_cycles(integer.get("multiply"), int_prov)
        self.divide_general = _sourced_cycles(integer.get("divide_general"), int_prov)
        self.divide_trivial = _sourced_cycles(
            integer.get("divide_by_zero_or_one"), int_prov
        )
        self.divide_int_min = _sourced_cycles(
            integer.get("divide_int_min_by_minus_one"), int_prov
        )
        #: The mispredict penalty is sourced (2-cycle bubble on Wormhole, 4 on
        #: Blackhole) and deliberately **not** charged: it is a cost per
        #: *mispredicted* branch and neither the ISA docs nor tt-sim describe
        #: the predictor, so the number of mispredictions is unknowable. Kept
        #: reachable so a report can say the gap is the predictor, not the
        #: table.
        self.branch_mispredict_observed = _sourced_cycles(
            integer.get("branch_mispredict_observed"), int_prov
        )

    @staticmethod
    def _load_latencies(riscv, arch):
        table = riscv.get("load_latency") or {}
        provenance = table.get("provenance")
        keys = _LOAD_LATENCY_KEYS[arch]
        latencies = []
        for region in range(RV_REGION_COUNT):
            key = keys.get(region)
            latencies.append(
                None if key is None else _sourced_cycles(table.get(key), provenance)
            )
        return tuple(latencies)

    def load_latency_bound(self, region):
        """The bound the source printed for ``region``'s latency, or ``None``.

        Every row but the local-data-RAM one is a ``>=``, so a modelled stall
        is a floor. Exposed for reporting rather than used for charging.
        """
        key = _LOAD_LATENCY_KEYS[self.arch].get(region)
        if key is None:
            return None
        cost = CycleCost.parse(self._table.get(key))
        return None if cost is None else cost.bound


_RISCV_MODELS = {}


def riscv_cost_model(arch):
    """The :class:`RiscvCostModel` for ``arch``, cached, or ``None`` when off.

    Same contract as :func:`unit_cost_model`: with ``TT_SIM_COST_MODEL`` unset
    this returns ``None`` and the caller stores one ``None`` attribute, which
    is the whole cost of the model on the RV interpreter's hot path.
    """
    if arch is None or not cost_model_enabled():
        return None
    if arch not in _RISCV_MODELS:
        _RISCV_MODELS[arch] = RiscvCostModel(load_costs(arch).sections, arch)
    return _RISCV_MODELS[arch]


# ---------------------------------------------------------------------------
# The NoC.
# ---------------------------------------------------------------------------
#
# A third shape again, and the simplest of the three: not a per-opcode table
# and not a scoreboard, but *distance x a constant*. The ISA docs publish a
# clean three-term per-hop model -- ~5 cycles NIU to router, 9 cycles router to
# router, ~5 cycles router back to NIU -- and ``unit_costs.yaml``'s own note on
# the section says what a consumer should do with it almost verbatim:
#
#   "NoC requests gain per-hop latency by scheduling the destination's request
#    event at c + hops * latency instead of c + 1"
#
# So the whole model is ``endpoint_cycles + per_hop_cycles * hops``, and the
# only interesting question is what ``hops`` is. That is a *topology* property
# rather than a cost-table one, so it lives with the NoC
# (``tt_sim.network.tt_noc.noc_hop_count``) and this class never sees a
# coordinate.
#
# Bandwidth is a *fourth* shape and arrives with the same class: not a latency
# at all but an **occupancy of a link**. One flit per cycle, 256 bits on
# Wormhole and 512 on Blackhole, so a packet of N flits holds the link it is
# injected on for N cycles and its tail lands N-1 cycles after its head. That
# is what makes a 32-byte semaphore poke cost less than an 8 KiB tile read,
# which the hop term alone cannot express because it never sees a size.
#
# What is deliberately absent is congestion. The docs say it "can negatively
# impact latency" and give no number; ``noc.congestion`` in the table is
# ``provenance: unknown`` for that reason, so this model charges a packet the
# same flight time whether *somebody else's* traffic is on the link or not --
# only the packet's own bytes are charged, and only on the one link tt-sim can
# name without an arbitration policy. That is the honest under-charge -- the
# same direction as every other bound in these files.


class NocCostModel:
    """Flight time of one NoC packet, in cycles, as a function of hop count.

    Two terms, both from ``unit_costs.yaml``'s ``noc.hops`` block:

    * :attr:`endpoint_cycles` -- the fixed NIU->router + router->NIU cost paid
      once per packet however far it travels (~5 + ~5). Both ends are
      ``bound: approximate``, so a modelled flight time is approximate too and
      :attr:`is_exact` says so.
    * :attr:`per_hop_cycles` -- 9 cycles per router-to-router hop, the one
      unqualified integer in the block, restated independently by the L1 page
      ("the latency of each hop is at least 9 cycles").

    Identical on both architectures: Blackhole's NoC page changes the flit
    width (256 -> 512 bits) and nothing else, which the table records in its
    ``arch_overrides``. The arch difference that *does* reach a hop count is
    the torus size, and that comes from the profile rather than from here.

    A term the tables do not source is charged nothing, per this module's first
    policy, so a hypothetical table with no ``router_to_router`` entry yields a
    flat per-packet cost rather than a fabricated distance term.
    """

    def __init__(self, sections, arch):
        self.arch = arch
        hops = (sections.get("noc") or {}).get("hops") or {}
        self._bounds = {}
        niu_to_router = self._hop_term(hops, "niu_to_router")
        router_to_niu = self._hop_term(hops, "router_to_niu")
        self.per_hop_cycles = self._hop_term(hops, "router_to_router")
        #: Cycles every packet pays regardless of distance, or ``None``.
        self.endpoint_cycles = _sum_or_none(niu_to_router, router_to_niu)
        #: The docs describe congestion qualitatively and quantify nothing, so
        #: a packet's flight time here does not depend on link occupancy. Kept
        #: as an attribute so a report can name the gap rather than imply it
        #: was modelled. See ``noc.congestion`` (``provenance: unknown``).
        self.congestion_modelled = False
        noc = sections.get("noc") or {}
        #: Bytes in one flit, or ``None`` when the section is not sourced. The
        #: table records ``flit_bits`` (256 Wormhole, 512 Blackhole) under the
        #: ``noc`` section's own ``isa_doc`` provenance.
        self.flit_bytes = self._flit_bytes(noc)
        #: Flits a link carries per cycle, from the hop table's
        #: ``throughput_flits_per_cycle``. Every entry says 1.
        self.flits_per_cycle = self._flit_rate(hops)
        #: Bytes per cycle one NoC link carries: 32 on Wormhole, 64 on
        #: Blackhole. Not a second source — it is ``flit_bytes`` x
        #: ``flits_per_cycle`` — but it is the form the *other* recorded
        #: bandwidth figure is in, and the two agree: Wormhole's
        #: ``link_bandwidth_gb_per_s: 32`` is exactly 32 B/cycle at the
        #: ``clock`` section's 1 GHz. Two independently recorded fields, one
        #: number, which is why the flit rate is safe to spend as bandwidth.
        self.bytes_per_cycle = (
            None
            if self.flit_bytes is None or self.flits_per_cycle is None
            else self.flit_bytes * self.flits_per_cycle
        )

    def _flit_bytes(self, noc):
        """Bytes per flit, or ``None`` when the section is not sourced.

        ``flit_bits`` is a bare scalar under the section, so the provenance
        that governs it is the section's own — which the Blackhole override
        restates (``bh_noc#performance``) alongside the doubled width.
        """
        if noc.get("provenance") not in SOURCED_PROVENANCE:
            return None
        bits = noc.get("flit_bits")
        return None if not bits else bits // 8

    @staticmethod
    def _flit_rate(hops):
        entry = hops.get("niu_to_router") or {}
        if entry.get("provenance") not in SOURCED_PROVENANCE:
            return None
        return entry.get("throughput_flits_per_cycle")

    @property
    def bandwidth_modelled(self):
        """True when a packet's size reaches its cost at all."""
        return self.bytes_per_cycle is not None

    def packet_flits(self, payload_bytes):
        """Flits a packet carrying ``payload_bytes`` of data is made of.

        At least one, because a packet with no payload — a read request, a
        write ACK, an atomic whose operand rides in the header — is still a
        packet and still occupies the link for a cycle. The header flit itself
        is **not** counted on top: the docs' flit accounting does not say how
        many flits a header takes, and inventing one would be a number with no
        source. That is the under-charging direction, like every other bound in
        these tables.
        """
        if self.flit_bytes is None:
            return None
        return max(1, int(math.ceil(payload_bytes / self.flit_bytes)))

    def serialisation_cycles(self, payload_bytes):
        """Cycles a packet of ``payload_bytes`` holds the link it is injected on.

        The bandwidth term, and deliberately an *occupancy* rather than a
        latency: a link carries :attr:`flits_per_cycle` flits per cycle, so a
        packet made of N flits takes N cycles to push onto the wire and the
        next packet behind it cannot start until it has. ``None`` when the
        table sources no flit width, which charges nothing.
        """
        flits = self.packet_flits(payload_bytes)
        if flits is None or not self.flits_per_cycle:
            return None
        return int(math.ceil(flits / self.flits_per_cycle))

    def tail_cycles(self, payload_bytes):
        """Extra cycles for a packet's last flit to arrive after its first.

        :meth:`serialisation_cycles` minus one, and the reason it is minus one
        rather than the whole thing is that the NoC is *wormhole*-routed: the
        head flit propagates hop by hop and the tail follows one cycle behind
        it, rather than the whole packet being received and re-sent at each
        router. So a packet's own size is paid **once**, whatever the distance,
        and the hop term is untouched by this.
        """
        cycles = self.serialisation_cycles(payload_bytes)
        return None if cycles is None else cycles - 1

    def _hop_term(self, hops, name):
        entry = hops.get(name) or {}
        cycles = _sourced_cycles(entry.get("latency"), entry.get("provenance"))
        if cycles is not None:
            cost = CycleCost.parse(entry.get("latency"))
            self._bounds[name] = cost.bound
        return cycles

    @property
    def is_exact(self):
        """False while any term came from a ``~``, ``>=`` or range.

        Both endpoint terms are written "~5 cycles", so this is False for the
        shipped table: a modelled flight time is an approximation of a
        published approximation, and a report should not print it as though it
        were counted.
        """
        return not (INEXACT_BOUNDS & set(self._bounds.values()))

    def flight_cycles(self, hops):
        """Cycles between a packet leaving one NIU and arriving at another.

        ``hops`` is the number of router-to-router hops, which is 0 for two
        endpoints on the same tile — that packet still pays
        :attr:`endpoint_cycles`, because it still goes NIU -> router -> NIU.
        Returns ``None`` when the tables sourced nothing at all, which leaves
        the caller's existing same-cycle-plus-one delivery untouched.
        """
        total = self.endpoint_cycles
        if self.per_hop_cycles is not None and hops:
            total = (0 if total is None else total) + self.per_hop_cycles * hops
        return total


def _sum_or_none(*values):
    present = [v for v in values if v is not None]
    return sum(present) if present else None


_NOC_MODELS = {}


def noc_cost_model(arch):
    """The :class:`NocCostModel` for ``arch``, cached, or ``None`` when off.

    Same contract as :func:`unit_cost_model` and :func:`riscv_cost_model`: an
    ``NUI`` built without an architecture, or any run without
    ``TT_SIM_COST_MODEL``, stores one ``None`` and the NoC keeps delivering a
    packet on the cycle after it was sent.
    """
    if arch is None or not cost_model_enabled():
        return None
    if arch not in _NOC_MODELS:
        _NOC_MODELS[arch] = NocCostModel(load_costs(arch).sections, arch)
    return _NOC_MODELS[arch]


# ---------------------------------------------------------------------------
# DRAM.
# ---------------------------------------------------------------------------
#
# The fourth shape, and the smallest: one number, paid once per request, at the
# *endpoint*. It is deliberately not part of the flight time above -- a packet's
# flight is what the interconnect costs, and this is what the device on the far
# end costs after the packet has landed. Keeping them separate is what stops the
# two terms double-counting, and it is why the number can be derived at all: see
# the ``derivation`` on ``dram.access_latency``, which subtracts one measured
# end-to-end figure from another of identical shape so that the NoC cancels.
#
# Three things this is not, all of which ROADMAP section I asks for and none of
# which any source quantifies:
#
# * **bank conflicts** -- tt-sim models no DRAM banks, and the ISA docs publish
#   no bank geometry or conflict cost for the DRAM tile;
# * **refresh windows** -- unpublished, and periodic rather than per-request, so
#   it is not even this shape;
# * **occupancy** -- the endpoint does not hold off a second request while it
#   services the first, so this adds latency and no contention. The
#   under-charging direction, like every other bound in these files.
#
# Nor is it size-dependent. DRAM *bandwidth* is well sourced (24 GB/s per
# channel, isa_doc) and deliberately unconsumed: turning a byte count into
# cycles is the same physical serialisation the NoC's per-link bandwidth term
# describes, so it belongs in one place, once, not in two that add up.


class DramCostModel:
    """What a DRAM endpoint's own service time costs, in cycles.

    One attribute worth reading and one worth naming:

    * :attr:`service_cycles` -- cycles between a request landing at a DRAM
      channel and that channel having serviced it, or ``None`` when the tables
      source nothing for this arch (which is Blackhole, on purpose -- see the
      ``access_latency`` note).
    * :attr:`is_exact` -- False for the shipped Wormhole entry, whose
      ``bound: at_least`` records that the derived 99 is the DRAM-versus-L1
      *difference* and therefore a floor under the absolute device latency.

    The provenance is worth checking rather than assuming: this is the file's
    only ``vendor_source_derived`` entry, which is arithmetic on two vendor
    measurements — weaker than a published number and stronger than a guess.
    :attr:`provenance` keeps it reachable so a report can say so.
    """

    def __init__(self, sections, arch):
        self.arch = arch
        entry = (sections.get("dram") or {}).get("access_latency") or {}
        self.provenance = entry.get("provenance")
        # An ``unknown`` entry carries no ``cycles`` at all — the convention
        # guarantees it and a test enforces it — so this is also the Blackhole
        # path, and it lands on ``None`` twice over.
        raw = entry if "cycles" in entry else None
        cost = CycleCost.parse(raw)
        self.service_cycles = _sourced_cycles(raw, self.provenance)
        #: The bound the entry carries, or ``None``. ``at_least`` here means
        #: the same thing it means everywhere else in this module: charged at
        #: the low end, so a modelled cycle count is a floor.
        self.bound = None if cost is None else cost.bound
        #: Named so a report can say the gaps are gaps rather than imply they
        #: were modelled. All three are unquantified by every available source.
        self.bank_conflicts_modelled = False
        self.refresh_modelled = False
        self.occupancy_modelled = False

    @property
    def is_exact(self):
        return self.bound not in INEXACT_BOUNDS


_DRAM_MODELS = {}


def dram_cost_model(arch):
    """The :class:`DramCostModel` for ``arch``, cached, or ``None`` when off.

    Same contract as the three above, plus one of its own: it also returns
    ``None`` when the arch's table sources no latency at all, so a Blackhole
    DRAM tile behaves exactly as it did before this landed rather than
    borrowing Wormhole's number.
    """
    if arch is None or not cost_model_enabled():
        return None
    if arch not in _DRAM_MODELS:
        model = DramCostModel(load_costs(arch).sections, arch)
        _DRAM_MODELS[arch] = model if model.service_cycles else None
    return _DRAM_MODELS[arch]
