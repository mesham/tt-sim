"""The cycle-approximate performance model (ROADMAP.md section I).

Two halves. ``costs.py`` is the data — per-unit, per-instruction cycle counts
with the provenance of every number attached — and ``model.py`` turns those
into the occupancy the simulator charges, behind the ``TT_SIM_COST_MODEL``
opt-in. Nothing in the execution path pays for either unless that variable is
set; the only consumer today is the Tensix matrix unit (Phase 5 of
``docs/plans/event-driven-pump.md``). See ``docs/plans/cost-model.md``.
"""
