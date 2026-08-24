# `srcarow` — an unconfigured unpacker, and the stop that catches it

An FP32 outer-product SFPU kernel from the nekbone team, delivered 2026-08-24,
kept for two reasons: it is the **only** kernel we have that exercises this
path, and its one-line control is the **only** case known to trip
`unpacker.py`'s SrcA row refusal.

## The experiment

`the-one-line.diff` is the whole of it — `binary_op_init_common` present or
absent in the three compute kernels:

| build | `Dest_cntx*_address` | unpacker 0 lands at | outcome |
|---|---|---|---|
| `src/` as shipped | 64 | row 4 | **PASS**, all 4096 values, and bit-exact on an n300 |
| `src/` + `control_kernels/` copied over | **0** | row 0 | **refused**, by design |

`logs/` holds the probe output for both, captured with
`evidence/srca_addr_probe` (a `sitecustomize.py` shim; see the collaboration
record).

## Why this is a regression test and not a bug report

The refusal was reported to us as a tt-sim limit blocking a configuration that
silicon accepts. It was not. The kernel that hit it had **never configured its
unpack hardware**, and on silicon that same kernel ran to completion returning
a constant `1.0f` for every element. Hardware accepting the configuration was
not evidence the configuration was valid — it was the same defect, failing
quietly instead of loudly.

So the control arm is the interesting half: it asserts that an unconfigured
unpack **keeps** stopping. Making it run — by scoping the 4-row subtraction to
the context path — was implemented, measured, and reverted: it would have
turned a caught bug into plausible numbers, which is the failure mode the stop
exists to prevent. See `docs/session_nekbone_collab_resume.md`.

## Using it

Build and run like any other optest (`find_package(TT-Metalium)`), `nelt = 4`,
one core, ~50 s warm. `run_sim.sh` targets Blackhole; Wormhole also passes.

**Drive it with a non-uniform field if you extend it.** The shipped validation
uses `u ≡ 1.0`, and that uniformity is exactly what let a constant-`1.0f` pack
fault masquerade as a passthrough for three days. The team's card runs used
`u[i] = 1 + (i % 7) * 0.125` — dyadic rationals, so the result stays bit-exact
against a double-precision reference.
