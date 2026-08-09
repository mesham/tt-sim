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
        self._ipc_groups = {
            name: entry.ipc_group
            for name, entry in unit_costs.instructions.items()
            if entry.ipc_group is not None
        }
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
        # -- the unpacker's throttled data phase (UNPACK only by data) -------
        # ``l1_bandwidth.throttle_modes`` maps the ``THCON_SEC[n].REG2_
        # Throttle_mode`` config value to a fetch rate in bytes per cycle
        # (16 / 32 / 64, ``isa_doc``); Blackhole adds ``blackhole_throttle``
        # via the arch override (x8 = 128 B/cycle, the x4 "2x" upgrade for
        # datums of two bytes and up, and the default modes used when
        # ``THCON_SEC0_REG1_ovrd_default_throttle_mode`` is clear). The PACK
        # unit also carries an ``l1_bandwidth`` block but no ``throttle_modes``,
        # so every attribute here stays ``None`` for it and for every other
        # unit, and :meth:`unpack_data_phase_cycles` charges nothing.
        self._throttle_rates = None
        self._tileize_mode_value = None
        self._bh_throttle = None
        bandwidth = unit_costs.extras.get("l1_bandwidth") or {}
        if bandwidth.get("provenance") in SOURCED_PROVENANCE:
            modes = bandwidth.get("throttle_modes") or {}
            rates = {
                entry["throttle_mode_value"]: entry["bytes_per_cycle"]
                for entry in modes.values()
            }
            if rates:
                self._throttle_rates = rates
                forced = bandwidth.get("tileize_forced_mode")
                if forced in modes:
                    self._tileize_mode_value = modes[forced]["throttle_mode_value"]
                bh = bandwidth.get("blackhole_throttle") or {}
                if bh.get("provenance") in SOURCED_PROVENANCE:
                    self._bh_throttle = bh
                    x8 = bh.get("x8") or {}
                    if x8:
                        rates[x8["throttle_mode_value"]] = x8["bytes_per_cycle"]

    # -- per-instruction ---------------------------------------------------
    def occupancy(self, instruction_name):
        """Cycles ``instruction_name`` occupies the unit, or ``None``."""
        return self._occupancy.get(instruction_name)

    def is_exact(self, instruction_name):
        """False when the charged number came from a ``>=``, ``~`` or range."""
        return instruction_name not in self._inexact

    @property
    def has_ipc_groups(self):
        """True when this unit's throughput limit is published per group.

        False for every unit but Blackhole's Configuration Unit, and read on
        the issue path so a unit without groups never pays for the concept:
        one attribute read and the whole-unit hold as before.
        """
        return bool(self._ipc_groups)

    def ipc_group(self, instruction_name):
        """The throughput group ``instruction_name`` contends for, or ``None``.

        ``None`` — the answer for every instruction of every ungrouped unit,
        and for an untabulated opcode of a grouped one — means "charge this
        against the whole unit", which is what
        :meth:`~tt_sim.pe.tensix.backends.backend_base.TensixBackendUnit.occupy_for`
        did unconditionally before groups existed and is the conservative
        answer: a whole-unit hold refuses strictly more than a per-group one,
        so an instruction whose group is unknown is never let through on a
        guess. Grouping is only ever transcribed from a published "IPC group"
        column — see the field's note in ``tensix_instruction_costs.yaml``.
        """
        return self._ipc_groups.get(instruction_name)

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

    # -- the unpacker's data phase -----------------------------------------
    def unpack_data_phase_cycles(
        self,
        transfer_bytes,
        throttle_mode,
        datum_bytes,
        tileize=False,
        default_throttle_overridden=True,
    ):
        """Cycles one UNPACR's L1 fetch takes at the throttle in effect, or ``None``.

        The second half of the only genuinely non-constant cost in the Tensix
        table (the first is the fixed address phase, which is the ``UNPACR``
        entry's ordinary ``occupancy``): "execution proceeds in a pipelined
        fashion, with the primary bottleneck being the fetching of bytes from
        L1", at a documented rate selected by the ``Throttle_mode`` config
        field — so the charge is ``ceil(transfer_bytes / rate)``, computed at
        issue from the size and the config in effect, which is why this is a
        method rather than a table lookup.

        The rate selection is transcribed from ``UNPACR_Regular.md``, in the
        pseudocode's own order:

        * ``tileize`` (``DiscontiguousInputRows``) forces the mode the table
          names in ``tileize_forced_mode`` — "tileize always runs at x4,
          regardless of Throttle_mode". The other forced modes the doc lists
          (compressed data, ``UpsampleZeroes``, BFP2) force modes of unpacks
          tt-sim rejects before moving a datum, so they never reach this.
        * On Blackhole with ``THCON_SEC[0].REG1_ovrd_default_throttle_mode``
          clear (``default_throttle_overridden=False``), the config mode is
          ignored: "8-bit modes use x8, others use x4".
        * Otherwise the mode is the config value ("0 means x1, 1 means x2, and
          2 means x4"; Blackhole adds 3 = x8, "illegal on Wormhole" — an
          untabulated mode charges nothing rather than guessing).
        * Blackhole upgrades x4 to 128 B/cycle for datums of two bytes and up
          ("upgrade to x4 '2x'").

        What is deliberately **not** here: the two unpackers' 3x3 interference
        table and the joint 80 B/cycle L1 ceiling. Both are shared constraints
        over two *simultaneously streaming* units, published as sustained
        rates with no per-transfer arbitration rule, so each unpacker is
        charged its own uncontended rate — the floor — and the ceiling is
        recorded unconsumed in the table (``joint_bandwidth``).
        """
        rates = self._throttle_rates
        if not rates or not transfer_bytes or transfer_bytes <= 0:
            return None
        bh = self._bh_throttle
        mode = throttle_mode
        if tileize and self._tileize_mode_value is not None:
            mode = self._tileize_mode_value
        elif bh is not None and not default_throttle_overridden:
            default = bh.get("default_mode") or {}
            mode = default.get(
                "byte_datums_value" if datum_bytes == 1 else "wider_datums_value",
                mode,
            )
        rate = rates.get(mode)
        if rate is None:
            return None
        if bh is not None:
            wide = bh.get("x4_wide") or {}
            if mode == wide.get("throttle_mode_value") and datum_bytes >= wide.get(
                "min_datum_bytes", math.inf
            ):
                rate = wide["bytes_per_cycle"]
        return int(math.ceil(transfer_bytes / rate))


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
#: Tile control / debug / status, the PIC, **both NIU register blocks** and the
#: NoC overlay — one row of the load-latency table on both architectures, and
#: named as six separate entries within that row's one cell.
RV_REGION_TILECTRL_PIC_NOC = 5
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
    "tilectrl_pic_noc",
    "unnamed",
)

#: The MMIO blocks tt-sim maps into a baby core's address space that the ISA
#: docs' load-latency table does not have a row for, so nothing can be charged
#: for them without inventing a number.
#:
#: **The NoC NIU register block used to head this list and does not any more**
#: (2026-08-06). It was recorded here as the load-bearing gap — every
#: ``noc_async_*_barrier`` in every tt-metal dataflow kernel polls a NIU
#: counter, which makes it the busiest MMIO load in the tree — on the reading
#: that the ">= 7" row named "TDMA / tile control / PIC / NoC *overlay*", a
#: different block. It does not: that row's cell lists "NoC 0 configuration
#: and command" and "NoC 1 configuration and command" as their own entries
#: next to the overlay's, on both architectures, each linking to the NIU
#: register block's own page. The number was in the table all along; what was
#: wrong was this file's key name for the row. See ``tt_sim/pe/rv/cost.py``.
#:
#: What is left is genuinely unnamed. None of the three appears in any row of
#: either architecture's load-latency table, and two of the three are not
#: really load targets at all (the instruction push buffers are written, and
#: NCRISC's IRAM is marked "not accessible by Load/Store Unit" in the memory
#: map), which is why nothing in the tree loads from them in any measurable
#: quantity.
RV_UNNAMED_REGIONS = (
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
        RV_REGION_TDMA: "tdma_tilectrl_pic_noc0_noc1_overlay",
        RV_REGION_TILECTRL_PIC_NOC: "tdma_tilectrl_pic_noc0_noc1_overlay",
    },
    "blackhole": {
        # Blackhole has an L0 data cache in front of L1, so the table gives two
        # numbers: 2 on a hit, >= 8 on a miss. This mapping names the row an
        # L1 load is charged **when its line is resident** in the per-core L0
        # line model (:attr:`RiscvCostModel.l0_lines` and friends, consumed by
        # ``tt_sim/pe/rv/cost.py``); a load whose line is not resident is
        # charged the miss row instead. The residency test is licensed by the
        # published geometry alone (``riscv.l0_data_cache``: 64 bytes, 4 lines
        # of 16, ``isa_doc``) — no hit *rate* is published anywhere, and none
        # is invented: the model tracks the four line tags and answers
        # per-load. Before 2026-08-06 every L1 load was charged this hit row
        # outright, the low end of the two-ended pair; silicon reads the miss
        # row on any working set the L0 cannot hold (`rv_load_chase` 8.098,
        # corroborated by `rv_load_indep` 1.742 via the docs' throughput
        # formula), which is what made the per-line model worth building.
        RV_REGION_L1: "l1_dcache_hit",
        RV_REGION_LOCAL_DATA_RAM: "core_local_data_ram",
        RV_REGION_MAILBOX_GROUP: "mailboxes_pcbufs_ttsync_semaphores",
        RV_REGION_TENSIX_GPR_CFG: "tensix_gprs_backend_config_tdma",
        RV_REGION_TDMA: "tensix_gprs_backend_config_tdma",
        RV_REGION_TILECTRL_PIC_NOC: "tilectrl_pic_noc0_noc1_overlay",
    },
}

#: The L1 load-latency row an access reaches when the L0 data cache **cannot**
#: hold its working set, or ``None`` on an architecture with no L0 cache (where
#: L1 has a single row and the question does not arise).
#:
#: Two consumers, both of which know something a bare address does not carry:
#: ``tt_sim/perf/riscv_bench_sweep`` compares individual benchmark probes whose
#: working sets are known exactly, and a probe whose working set exceeds
#: ``riscv.l0_data_cache.capacity_bytes`` reaches this row whatever the
#: cache's (unpublished) organisation; and, since 2026-08-06,
#: :attr:`RiscvCostModel.l1_load_miss_latency` feeds it to the per-core L0
#: line model in ``tt_sim/pe/rv/cost.py``, which charges it to a load whose
#: line is not among the tracked line tags. Both mappings live here so the two
#: can never name rows the other does not have.
_L1_DCACHE_MISS_KEYS = {
    "wormhole": None,
    "blackhole": "l1_dcache_miss",
}


def l1_dcache_miss_key(arch):
    """The ``riscv.load_latency`` row for an L1 load that misses the L0 cache.

    ``None`` where the architecture publishes no L0 data cache.
    """
    return _L1_DCACHE_MISS_KEYS.get(arch)


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
      Where the tables publish an L0 data cache (Blackhole), the L1 row is a
      hit/miss pair: :attr:`l0_lines` / :attr:`l0_line_bytes` carry the
      published geometry and :attr:`l1_load_miss_latency` the miss row, so the
      consumer can answer residency per load instead of picking one row for
      every L1 access.
    * :attr:`l1_store_period` / :attr:`l1_coalesced_store_period` — the
      sustained store rate, which *is* an occupancy of the load/store unit.
    * :attr:`load_slots` / :attr:`load_slot_cycles` — the sustained *load*
      rate, which is an occupancy of a queue rather than of the unit: the
      docs give "four such loads every N - 1 cycles" for a load latency of
      N >= 5, and one per cycle below that. Expressed here as what it is —
      a fixed number of in-flight slots, each held for N - 1 cycles — so a
      stream of mixed latencies has an answer at all. See
      :attr:`load_slot_cycles` for why that shape is the published one
      rather than a generalisation of it.
    * :attr:`multiply` / the divide costs — the integer unit's own blocking,
      the one place the docs say outright that "the next instruction cannot
      enter the unit until the multiply instruction has finished". On an
      architecture whose multiply *pipelines* instead of blocking, the
      published stage split becomes :attr:`multiply_latency` — a scoreboard
      input like the load table, not an occupancy.

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
        # -- the L0 data-cache line model (Blackhole only by data) ---------
        # Engaged only when the tables publish all three of: an L0 geometry
        # (``riscv.l0_data_cache``: 64 bytes, 4 lines of 16, isa_doc), a hit
        # row *and* a miss row for L1. Wormhole publishes none of them — its
        # L1 has a single ``>= 8`` row — so every attribute is ``None`` there
        # and ``cost.py`` never builds the tag array. What the geometry
        # licenses is exactly a residency test: a line among the last
        # ``l0_lines`` distinct lines loaded can be resident, anything else
        # cannot (the capacity argument is one-sided, and the model takes the
        # generous side of every unpublished property — see ``cost.py`` for
        # the replacement-policy discussion).
        self.l0_lines = None
        self.l0_line_bytes = None
        self.l1_load_miss_latency = None
        l0 = riscv.get("l0_data_cache") or {}
        miss_key = _L1_DCACHE_MISS_KEYS.get(arch)
        if l0.get("provenance") in SOURCED_PROVENANCE and miss_key is not None:
            miss = _sourced_cycles(
                self._table.get(miss_key), self._table.get("provenance")
            )
            lines = l0.get("lines")
            line_bytes = l0.get("line_bytes")
            hit = self.load_latency[RV_REGION_L1]
            if miss is not None and hit is not None and lines and line_bytes:
                self.l0_lines = lines
                self.l0_line_bytes = line_bytes
                self.l1_load_miss_latency = miss
        self._load_throughput(riscv)
        integer = riscv.get("integer_unit") or {}
        int_prov = integer.get("provenance")
        self.multiply = _sourced_cycles(integer.get("multiply"), int_prov)
        #: Cycles from a multiply issuing to its result being readable, or
        #: ``None`` where the tables publish only a blocking occupancy.
        #: Blackhole's page splits the pipeline — "exactly one cycle in EX1,
        #: and then exactly one cycle in EX2" — so its multiply has occupancy
        #: 1 and result latency 1 + 1 = 2, and a dependent chain pays the
        #: latency (silicon: ``rv_mul_dep`` 1.985). Wormhole publishes no
        #: ``multiply_ex2`` (its multiply *blocks* the integer unit for two
        #: cycles, which :attr:`multiply` already charges as occupancy), so
        #: this stays ``None`` there and nothing changes.
        multiply_ex2 = _sourced_cycles(integer.get("multiply_ex2"), int_prov)
        self.multiply_latency = (
            None
            if self.multiply is None or multiply_ex2 is None
            else self.multiply + multiply_ex2
        )
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

    def _load_throughput(self, riscv):
        """The sustained-load rate, turned into slots and slot occupancies.

        The entry is one sentence — "throughput of sustained loads is one per
        cycle if the load latency is less than five cycles. Otherwise, when the
        load latency is ``N`` cycles, the throughput of sustained loads is four
        such loads every ``N - 1`` cycles" — and the shape it is stored in here
        is a **rate limiter with four slots, each held ``N - 1`` cycles**, which
        is that sentence and not a generalisation of it:

        * For a stream of one latency it *is* the sentence: four slots, each
          freed ``N - 1`` cycles after it was taken, admits four loads every
          ``N - 1`` cycles and no more.
        * The four is also the number the same page's table prints in its
          "Maximum loads in flight" column ("4 (in aggregate across all of
          these regions)"), so the two published statements are one mechanism
          seen twice, and ``N - 1`` is what Little's law reads off them
          together: four loads in flight sustaining four per ``N - 1`` cycles
          is a residency of exactly ``N - 1``. That is why
          :data:`riscv.load_latency.max_loads_in_flight` is **not** separately
          charged — charging both would bill one queue twice. See
          ``docs/plans/cost-model.md``.
        * A stream of mixed latencies is the case the sentence does not cover,
          and the slot form answers it in the only way that reduces to the
          sentence on every uniform stream.

        Loads whose latency is below ``one_per_cycle_if_latency_under`` take no
        slot at all: the doc gives them one per cycle, which a single-issue
        core already cannot beat, so the fast rows (core-local RAM, an L0
        d-cache hit) are charged nothing by this term.
        """
        #: How many loads may be in flight at the sustained rate, or ``None``
        #: when the entry is not sourced (and then nothing is charged).
        self.load_slots = None
        #: Per region, the cycles one load holds a slot, or ``None`` where the
        #: region's latency is below the "one per cycle" threshold (or unnamed).
        self.load_slot_cycles = (None,) * RV_REGION_COUNT
        #: The same, for an L1 load that missed the L0 data cache.
        self.l1_miss_slot_cycles = None
        throughput = riscv.get("load_throughput") or {}
        if throughput.get("provenance") not in SOURCED_PROVENANCE:
            return
        slots = throughput.get("else_loads_per_window")
        under = throughput.get("one_per_cycle_if_latency_under")
        offset = throughput.get("else_window_cycles_offset")
        if not slots or under is None or offset is None:
            return

        def window(latency):
            if latency is None or latency < under:
                return None
            return max(latency + offset, 0) or None

        self.load_slots = slots
        self.load_slot_cycles = tuple(window(c) for c in self.load_latency)
        self.l1_miss_slot_cycles = window(self.l1_load_miss_latency)

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
        packet and still occupies the link for a cycle.

        The header flit is **not** counted on top, and the reason has changed.
        It used to be that the docs' flit accounting did not say how many flits
        a header takes. It does, in the first sentence of the NoC page: "Each
        packet consists of one or more flits (exactly one header flit, followed
        by up to 256 data flits)" (``wh_noc``/``bh_noc``), and the same page
        adds that "the amount of *useful* throughput depends on the ratio of
        header flits to data flits". So the number exists and is 1, at
        ``isa_doc``. What has not happened is the change: adding it costs every
        packet in the tree a cycle, which **doubles** the modelled link
        occupancy of a 32-byte semaphore poke while adding 0.39 % to a 64 KiB
        transfer, and it is a whole-tree timing perturbation for a term rung 2
        measured as 3–6 % of the bandwidth gap it was hoped to explain. It
        wants its own change and its own guard run. See ``docs/plans/
        cost-model.md``, "Is the L1 read shortfall sourceable?".
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
# * **device occupancy** -- how long the array itself is unavailable to a second
#   request. ``service_cycles`` is a latency and says nothing about the re-issue
#   interval behind it, so reading one as the other would assert a throughput no
#   source supports. The under-charging direction, like every other bound here.
#
# What *is* held off, since 2026-08-09, is the **channel**: a GDDR6 channel
# carries one transfer at a time, so a request arriving while the previous one
# is still streaming waits for it. That needed no new number --
# ``channel_serialisation`` is already in the table at ``isa_doc_derived`` and
# already spent as a latency -- and it is a floor twice over: it is the shortest
# any endpoint can possibly be busy (the bytes have to cross the bus), and it is
# charged only where the rate is published, which is Wormhole and not Blackhole.
# Before it, a tt-sim Wormhole DRAM channel sustained 32 B/cycle, the NoC link's
# rate, against the 24 the ISA docs publish for the channel.
#
# Size dependence is the *fifth* shape and arrives with the same class, as of
# 2026-08-04. It used to be argued away: DRAM bandwidth is well sourced
# (24 GB/s per channel, isa_doc) and was deliberately unconsumed on the ground
# that turning a byte count into cycles is the same physical serialisation the
# NoC's per-link bandwidth term already describes, so it belonged in one place,
# once, not in two that add up. That argument confused two queues for one. The
# NoC link and the GDDR6 channel are separate hardware at separate rates -- 32
# B/cycle against 24 on Wormhole -- in series, and a transfer streaming through
# both runs at the slower. Charging the *sum* would indeed double-bill; what
# :meth:`DramCostModel.channel_excess_cycles` returns is the **excess** of the
# channel's serialisation over the link's, so the round trip's size-dependent
# cost is the slower of the two exactly once.
#
# Rung 2 is what settled it: tt-metal's measured dataset has a Wormhole DRAM
# read sustaining 24.38 B/cycle against a modelled 32, and 24.38 is not a
# fraction of 32 -- it is ``dram.bandwidth.per_channel_gb_per_s`` to within
# 2 %. Nothing was fitted: the number was already in the table at ``isa_doc``,
# and all this consumes is a unit conversion into bytes per cycle.


class DramCostModel:
    """What a DRAM endpoint's own service time costs, in cycles.

    One attribute worth reading and one worth naming:

    * :attr:`service_cycles` -- cycles between a request landing at a DRAM
      channel and that channel having serviced it, or ``None`` when the tables
      source nothing for this arch. Both shipped arches source one (99
      Wormhole, 126 Blackhole); the ``None`` path is what a third arch, or the
      base entry's ``unknown``, would land on.
    * :attr:`is_exact` -- False for both shipped entries, whose
      ``bound: at_least`` records that the derived figure is the DRAM-versus-L1
      *difference* and therefore a floor under the absolute device latency.

    The provenance is worth checking rather than assuming: these are the
    file's only ``vendor_source_derived`` entries, arithmetic on two vendor
    measurements — weaker than a published number and stronger than a guess.
    :attr:`provenance` keeps it reachable so a report can say so.

    Plus one term that is not a latency at all:

    * :attr:`channel_bytes_per_cycle` -- the GDDR6 channel's own rate, 24 on
      Wormhole and ``None`` on Blackhole, which publishes no per-channel
      bandwidth. Spent through :meth:`channel_excess_cycles`, never as an
      absolute serialisation, because the NoC link has already charged its own.
    """

    def __init__(self, sections, arch):
        self.arch = arch
        dram = sections.get("dram") or {}
        entry = dram.get("access_latency") or {}
        self.provenance = entry.get("provenance")
        # An ``unknown`` entry carries no ``cycles`` at all — the convention
        # guarantees it and a test enforces it — so an unsourced arch lands on
        # ``None`` twice over rather than borrowing another arch's number.
        raw = entry if "cycles" in entry else None
        cost = CycleCost.parse(raw)
        self.service_cycles = _sourced_cycles(raw, self.provenance)
        #: The bound the entry carries, or ``None``. ``at_least`` here means
        #: the same thing it means everywhere else in this module: charged at
        #: the low end, so a modelled cycle count is a floor.
        self.bound = None if cost is None else cost.bound
        #: Named so a report can say the gaps are gaps rather than imply they
        #: were modelled. Both are unquantified by every available source.
        self.bank_conflicts_modelled = False
        self.refresh_modelled = False
        #: The part of endpoint occupancy that is **not** modelled, and the
        #: reason :attr:`occupancy_modelled` below is not the whole story: how
        #: long the DRAM array itself is unavailable to a second request. That
        #: is the device's re-issue interval, and no source publishes it or the
        #: pipelining depth it implies -- :attr:`service_cycles` is a *latency*,
        #: and reading a latency as an occupancy would assert a throughput of
        #: one request per 99 cycles, which is 0.3 B/cycle against a channel
        #: the same page publishes at 24. So it stays a named gap.
        self.device_occupancy_modelled = False
        #: Bytes the DRAM channel moves per cycle, or ``None``. Provenance is
        #: checked rather than assumed, and that check is load-bearing: the
        #: arch overrides deep-merge, so Wormhole's 24 is still *present* under
        #: Blackhole's ``unknown`` override and reading it without looking
        #: would launder one arch's published figure into another's gap.
        channel = dram.get("channel_serialisation") or {}
        self.channel_bytes_per_cycle = (
            channel.get("bytes_per_cycle")
            if channel.get("provenance") in SOURCED_PROVENANCE
            else None
        )
        #: Whether a second request is held off while the first is being
        #: serviced -- and it means the **channel data bus** only, because that
        #: is the only part of an endpoint's occupancy any source sizes. Where
        #: the channel rate is sourced this is the same
        #: :meth:`channel_serialisation_cycles` the latency term already
        #: spends, held as a resource rather than added as a delay; where it is
        #: not (Blackhole, whose ``dram.bandwidth`` is ``unknown``) the
        #: endpoint stays contention-free and this reads False. See
        #: :attr:`device_occupancy_modelled` for the half that is still a gap.
        self.occupancy_modelled = self.channel_bytes_per_cycle is not None

    def channel_serialisation_cycles(self, payload_bytes):
        """Cycles the channel itself needs to move ``payload_bytes``, or ``None``.

        The raw ``ceil(N / rate)``, and it is spent twice on two different
        axes — one number, not two, exactly as the NoC link's occupancy is:

        * as a **latency**, through :meth:`channel_excess_cycles`, which charges
          only the excess over what the link already billed so a single
          transfer's size cost is ``ceil(N / rate)`` once rather than twice;
        * as an **occupancy**, since 2026-08-09: the channel carries one
          transfer at a time, so a request arriving while it is busy waits.
          That charge is zero for an isolated request and bites only on a
          stream, which is why it moves a sustained rate and not a latency.
        """
        rate = self.channel_bytes_per_cycle
        if not rate:
            return None
        return int(math.ceil(payload_bytes / rate))

    def channel_excess_cycles(self, payload_bytes, link_cycles):
        """How much slower the channel is than the link, for these bytes.

        The whole of the size term, and the reason it is a *difference*: the
        channel and the NoC link are two stages in series, so a transfer
        streaming through both is limited by the slower one, not by their sum.
        The link's serialisation (``link_cycles``, what
        :meth:`NocCostModel.serialisation_cycles` already charged the packet)
        is therefore subtracted out, leaving ``max(0, channel - link)`` and a
        round trip whose size-dependent cost is ``ceil(N / rate)`` exactly
        once.

        Zero — never negative — when the link is the slower of the two, which
        is what makes this safe to add unconditionally: an architecture whose
        NoC is slower than its DRAM gets no charge from here at all, rather
        than a refund it did not earn.

        ``None`` for ``link_cycles`` means the NoC bandwidth term is not
        modelled either, so there is nothing to be the excess *of* and the
        honest answer is to charge nothing.
        """
        channel = self.channel_serialisation_cycles(payload_bytes)
        if channel is None or link_cycles is None:
            return 0
        return max(0, channel - link_cycles)

    @property
    def is_exact(self):
        return self.bound not in INEXACT_BOUNDS


_DRAM_MODELS = {}


def dram_cost_model(arch):
    """The :class:`DramCostModel` for ``arch``, cached, or ``None`` when off.

    Same contract as the three above, plus one of its own: it also returns
    ``None`` when the arch's table sources no latency at all, so an arch
    without a derivation of its own keeps an instantaneous DRAM rather than
    borrowing another arch's number.
    """
    if arch is None or not cost_model_enabled():
        return None
    if arch not in _DRAM_MODELS:
        model = DramCostModel(load_costs(arch).sections, arch)
        _DRAM_MODELS[arch] = model if model.service_cycles else None
    return _DRAM_MODELS[arch]


# ---------------------------------------------------------------------------
# The Mover.
# ---------------------------------------------------------------------------
#
# The XMOV entry in the Tensix table is 1 cycle, and that 1 is the *issue*
# cost only: "The thread issuing an XMOV instruction will be automatically
# stalled until the mover is able to start work, at which point XMOV will
# execute in a single cycle - the mover proceeds with the task in the
# background" (wh_xmov). The interesting half is how long the background task
# runs, and that is bandwidth-derived from ``unit_costs.yaml``'s ``mover``
# section -- whose numbers the ISA doc publishes as *measured*, with an ideal
# and a contended column per transfer kind. "Stalled until the mover is able
# to start work" is precisely what a unit occupancy models, so the transfer
# duration is charged as mover-unit occupancy and the existing issue-refusal
# machinery is the whole delivery mechanism.
#
# The ideal column is charged and the contended one is not: the page gives no
# rule for when the L1-port contention it mentions applies, so the ideal rate
# is the floor -- the same reasoning that charges every ``at_least`` at its
# low end.


class MoverCostModel:
    """Transfer duration of one Mover command, in cycles, by transfer kind.

    ``kind`` is a key of ``mover.transfer`` in ``unit_costs.yaml``:
    ``l1_to_l1`` (the memcpy modes, ``XMOV_L1_TO_L1`` and ``XMOV_L1_TO_L0``),
    ``l1_memset`` (``XMOV_L0_TO_L1``) or ``non_l1_memset``
    (``XMOV_L0_TO_L0``). The mapping from an XMOV mode to a kind lives with
    the mover backend, which owns the mode enum; this class only prices bytes.
    """

    def __init__(self, sections, arch):
        self.arch = arch
        transfer = (sections.get("mover") or {}).get("transfer") or {}
        #: ``{kind: (bits moved per period, cycles per period)}`` from the
        #: ideal column: "Eight 128b reads and eight 128b writes every 11
        #: cycles" is 8 x 128 = 1024 bits copied per 11 cycles for the memcpy
        #: modes; one 128-bit write per cycle for the memsets.
        self._rates = {}
        for kind, block in transfer.items():
            if not isinstance(block, dict):
                continue
            if block.get("provenance") not in SOURCED_PROVENANCE:
                continue
            ideal = block.get("ideal") or {}
            bits_each = ideal.get("bits_each")
            per_cycles = ideal.get("per_cycles")
            moves = ideal.get("reads") or ideal.get("writes")
            if not (bits_each and per_cycles and moves):
                continue
            self._rates[kind] = (moves * bits_each, per_cycles)

    @property
    def transfer_modelled(self):
        """True when at least one transfer kind has a sourced rate."""
        return bool(self._rates)

    def transfer_cycles(self, kind, payload_bytes):
        """Cycles the mover is busy moving ``payload_bytes`` as ``kind``.

        ``ceil(bits * cycles_per_period / bits_per_period)`` — the sustained
        rate applied fractionally, which charges a transfer smaller than one
        period less than a whole period. That is the floor reading of a rate
        statement; rounding up to whole read/write bursts would charge a
        16-byte memcpy the full 11 cycles the doc never claims for it.
        ``None`` — charge nothing — for an unpriced kind or an empty transfer.
        """
        rate = self._rates.get(kind)
        if rate is None or not payload_bytes or payload_bytes <= 0:
            return None
        bits_per_period, period_cycles = rate
        return int(math.ceil(payload_bytes * 8 * period_cycles / bits_per_period))


_MOVER_MODELS = {}


def mover_cost_model(arch):
    """The :class:`MoverCostModel` for ``arch``, cached, or ``None`` when off.

    Same contract as the other four: ``None`` with the model off or when the
    table prices no transfer at all, so the mover backend stores one ``None``
    and every XMOV keeps its same-cycle memcpy.
    """
    if arch is None or not cost_model_enabled():
        return None
    if arch not in _MOVER_MODELS:
        model = MoverCostModel(load_costs(arch).sections, arch)
        _MOVER_MODELS[arch] = model if model.transfer_modelled else None
    return _MOVER_MODELS[arch]
