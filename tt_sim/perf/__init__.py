"""The cycle-approximate performance model (ROADMAP.md section I).

Two halves. ``costs.py`` is the data — per-unit, per-instruction cycle counts
with the provenance of every number attached — and ``model.py`` turns those
into the cycles the simulator charges, behind the ``TT_SIM_COST_MODEL``
opt-in. Nothing in the execution path pays for either unless that variable is
set. The consumers today are five Tensix backend units (matrix, vector, ThCon,
packer, sync) and the baby RISC-V cores' load/store path — Phase 5 of
``docs/plans/event-driven-pump.md``. See ``docs/plans/cost-model.md``.
"""
