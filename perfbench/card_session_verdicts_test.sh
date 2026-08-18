#!/usr/bin/env bash
# Regression test for perfbench/card_session_verdicts.sh.
#
# The fixtures under testdata/card-session-blackhole-2026-08-09/ are the files a
# real Blackhole card returned on 2026-08-09, unmodified. That session printed
#
#   VERDICT nocread: MEANINGFUL -- no initiator limit -- this RETIRES the
#                    credit-limit term, which is a result
#   VERDICT cmdbuf:  MEANINGFUL -- CMD_BUF_AVAIL at rest = 0x00000000 -- the
#                    number the ISA docs withhold
#   VERDICT noc-epoch: MEANINGFUL -- the detector reported no reproducing
#                    per-core epoch, which retires the (11,2) observation
#
# on data where `outstanding_max` read 72 in all 129 rows including bursts of
# four requests, where CMD_BUF_AVAIL was byte-identical at rest and mid-burst in
# all 129 rows, and where the clock-epoch detector had never run at all -- its
# report file held one line of ModuleNotFoundError. The (11,2) epoch it
# "retired" is real: +323438586 cycles, reproduced over five runs.
#
# So the test asserts the fixed logic calls each of those degenerate, on that
# session's own bytes. It costs nothing at a card and it is the only proof
# available without one.
#
#   perfbench/card_session_verdicts_test.sh
#
# SPDX-License-Identifier: Apache-2.0

set -u

PB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARD="$PB/testdata/card-session-blackhole-2026-08-09"
# shellcheck source=card_session_verdicts.sh
. "$PB/card_session_verdicts.sh"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/verdicts_test.XXXXXX")" || exit 2
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0

check() { # description, expected-status, actual "STATUS|text"
  local want="$2" got="${3%%|*}" text="${3#*|}"
  if [ "$got" = "$want" ]; then
    pass=$((pass + 1))
    printf 'ok   %-58s %s\n' "$1" "$got"
  else
    fail=$((fail + 1))
    printf 'FAIL %-58s expected %s, got %s\n     %s\n' "$1" "$want" "$got" "$text"
  fi
}

check_says() { # description, substring, actual
  if case "$3" in *"$2"*) true ;; *) false ;; esac; then
    pass=$((pass + 1))
    printf 'ok   %-58s says "%s"\n' "$1" "$2"
  else
    fail=$((fail + 1))
    printf 'FAIL %-58s does not mention "%s"\n     %s\n' "$1" "$2" "$3"
  fi
}

echo "== the 2026-08-09 Blackhole card session, graded again"

r="$(nocread_verdict "$CARD/nocread.out" "$CARD/nocread.blackhole.csv")"
check "nocread on the returned CSV" DEGENERATE "$r"
check_says "nocread names the impossible reading" "cannot exceed the burst" "$r"

r="$(cmdbuf_verdict "$CARD/nocread.blackhole.csv")"
check "cmdbuf on the returned CSV" DEGENERATE "$r"
check_says "cmdbuf says the control did not move" "control did not move" "$r"

# Both report files hold exactly one line of ModuleNotFoundError.
r="$(noc_verdict "$CARD/noc.report.txt" 1)"
check "noc on a report that is a traceback" FAILED "$r"
r="$(noc_epoch_verdict "$CARD/noc-epoch.report.txt" 1)"
check "noc-epoch on a report that is a traceback" FAILED "$r"
check_says "noc-epoch retires nothing" "in either direction" "$r"

# With tt_sim absent the probe must say the analysis is deferred, not write an
# error file and grade it.
r="$(noc_verdict "$TMP/absent.report.txt" 0)"
check "noc with no tt_sim on the box" DEFERRED "$r"
r="$(noc_epoch_verdict "$TMP/absent.report.txt" 0)"
check "noc-epoch with no tt_sim on the box" DEFERRED "$r"
check_says "noc-epoch defers without a claim" "NEITHER confirmed NOR retired" "$r"

# The primary rv run failed phase Q and nothing else, and every complaint is at
# n <= 16 -- the small-burst scatter riscvbench/README.md documents on silicon.
r="$(rv_verdict "$CARD/rv.out" rv)"
check "rv primary, phase Q only at n <= 16" MEANINGFUL "$r"
check_says "rv names the README's exemption" "cascade failing" "$r"

# The --blocks 8 cross-check failed R, C, Q, F and G. It is graded on its own
# file now, and it is still SUSPECT -- which is why the cross-check runs at the
# same parameters as the primary from here on.
r="$(rv_verdict "$CARD/rv-cross.out" rv-cross)"
check "rv --blocks 8 cross-check, five phases failed" SUSPECT "$r"

echo
echo "== the 13:41 re-run: a BOUNDED verdict its own dist sweep refutes"

# The second card run of 2026-08-09. The kernel fix worked -- inflight_rest is 0
# in every row and both instruments agree -- and the benchmark then printed
# BOUNDED AT 13. It is not a limit: 13 comes from three rows at one transaction
# size (4096 B) while every other row reads 2-3, and cycles_per_tx is flat to
# 1.5% across 13 hops. A credit limit K caps the rate at round_trip/K, so the
# rate MUST rise with distance. The peak is service time, not a ceiling.
R2="$PB/testdata/card-session-blackhole-2026-08-09-run2"
r="$(nocread_verdict "$R2/nocread.out" "$R2/nocread.blackhole.csv")"
check "nocread on the re-run, BOUNDED AT 13" DEGENERATE "$r"
check_says "nocread applies the falsifying dist test" "MUST rise with distance" "$r"

echo
echo "== synthetic cases: the checks must still pass a healthy run"

cat > "$TMP/healthy.csv" <<'EOF'
# nocreadbench arch=blackhole grid=12x10 num_tx=64 repeats=1
# every row is one timed burst; cycles_per_tx = cycles / num_tx
experiment,repeat,point,mst_x,mst_y,mst_node_x,mst_node_y,num_src,src0_x,src0_y,hops,num_tx,tx_bytes,dst_stride,src_stride,cycles,cycles_per_tx,outstanding_max,outstanding_end,samples,cmdbuf_avail_rest,cmdbuf_avail_busy,outstanding_rest,outstanding_delta,inflight_max,inflight_rest,trid,cmdbuf_avail_max,cmdbuf_ovfl_rest,cmdbuf_ovfl_end
burst,0,0,0,0,1,2,1,14,2,13,4,64,0,0,417,104.250,3,1,4,0x00000000,0x00000001,0,3,3,0,0,0x00000002,0x00000000,0x00000000
burst,0,1,0,0,1,2,1,14,2,13,64,64,0,0,3237,50.578,12,4,64,0x00000000,0x00000001,0,12,11,0,0,0x00000003,0x00000000,0x00000000
dist,0,2,0,0,1,2,1,2,2,1,64,64,0,0,3200,50.000,12,4,64,0x00000000,0x00000001,0,12,11,0,0,0x00000003,0x00000000,0x00000000
dist,0,3,0,0,1,2,1,14,2,13,64,64,0,0,7680,120.000,12,4,64,0x00000000,0x00000001,0,12,11,0,0,0x00000003,0x00000000,0x00000000
EOF
printf '  VERDICT: BOUNDED AT 11 -- the initiator holds at most this many\n' > "$TMP/healthy.out"
r="$(nocread_verdict "$TMP/healthy.out" "$TMP/healthy.csv")"
check "nocread on a healthy NRB2 CSV" MEANINGFUL "$r"
r="$(cmdbuf_verdict "$TMP/healthy.csv")"
check "cmdbuf on a healthy NRB2 CSV" MEANINGFUL "$r"

# The case the 2026-08-09 re-run hit: a LARGE baseline. The counter is live
# hardware state, so on that card it sat at ~71 before a request was issued and
# read 76 during a burst of 64. The raw max exceeding the burst is EXPECTED
# there and is exactly what the rest sample exists to remove -- applying the
# "cannot exceed the burst" law to the raw column instead of the delta rejects
# every healthy run. Delta 5 of a 64-burst is a real, modest occupancy.
sed 's/,3237,50.578,12,4,64,/,3237,50.578,76,72,64,/; s/,0x00000001,0,12,11,0,0,/,0x00000001,71,5,5,0,0,/' \
  "$TMP/healthy.csv" > "$TMP/baseline.csv"
r="$(nocread_verdict "$TMP/healthy.out" "$TMP/baseline.csv")"
check "nocread when the counter has a large rest baseline" MEANINGFUL "$r"
# ... and the law still bites when the BASELINED quantity is impossible.
sed 's/,0x00000001,71,5,5,0,0,/,0x00000001,71,99,5,0,0,/' \
  "$TMP/baseline.csv" > "$TMP/baseline-bad.csv"
r="$(nocread_verdict "$TMP/healthy.out" "$TMP/baseline-bad.csv")"
check "nocread when the DELTA exceeds the burst" DEGENERATE "$r"

# The same CSV with the occupancy columns flat. Every value is legal, in range
# and correctly formatted -- and nothing moved.
sed 's/,0,3,3,0,0,/,0,0,0,0,0,/; s/,0,12,11,0,0,/,0,0,0,0,0,/; s/0x00000001,/0x00000000,/; s/0x00000002,/0x00000000,/; s/0x00000003,/0x00000000,/' \
  "$TMP/healthy.csv" > "$TMP/flat.csv"
r="$(nocread_verdict "$TMP/healthy.out" "$TMP/flat.csv")"
check "nocread when both in-flight instruments read zero" DEGENERATE "$r"
r="$(cmdbuf_verdict "$TMP/flat.csv")"
check "cmdbuf when rest, busy and peak all agree" DEGENERATE "$r"

# The all-ones sentinel: the register was not in the build at all.
sed 's/0x00000000/0xFFFFFFFF/g' "$TMP/healthy.csv" > "$TMP/absent.csv"
r="$(cmdbuf_verdict "$TMP/absent.csv")"
check "cmdbuf on the register-absent sentinel" DEGENERATE "$r"

# An NRB1-schema CSV whose counter IS in range still cannot support NO
# INITIATOR LIMIT, because that verdict was reached from a raw counter.
cat > "$TMP/nrb1.csv" <<'EOF'
# nocreadbench arch=blackhole grid=12x10 num_tx=64 repeats=1
# every row is one timed burst; cycles_per_tx = cycles / num_tx
experiment,repeat,point,mst_x,mst_y,mst_node_x,mst_node_y,num_src,src0_x,src0_y,hops,num_tx,tx_bytes,dst_stride,src_stride,cycles,cycles_per_tx,outstanding_max,outstanding_end,samples,cmdbuf_avail_rest,cmdbuf_avail_busy
burst,0,1,0,0,1,2,1,14,2,13,64,64,0,0,3237,50.578,60,60,64,0x00000000,0x00000000
EOF
printf '  VERDICT: NO INITIATOR LIMIT -- in-flight requests track the burst\n' > "$TMP/nrb1.out"
r="$(nocread_verdict "$TMP/nrb1.out" "$TMP/nrb1.csv")"
check "nocread refuses NO INITIATOR LIMIT from an NRB1 CSV" DEGENERATE "$r"

echo
echo "== phase Q is exempted by the README's threshold, not blanket-suppressed"

{ echo "TTRVBENCH_VALID_R: yes"
  echo "TTRVBENCH_VALID_Q: no (1 checks failed)"
  echo "  NOT MONOTONE: q/t1/q_nop thread 0: n=32 -> 90 cycles, n=64 -> 88 cycles"
} > "$TMP/q_big.out"
r="$(rv_verdict "$TMP/q_big.out" rv)"
check "rv, phase Q complaint at n = 64" SUSPECT "$r"

{ echo "TTRVBENCH_VALID_R: no (1 checks failed)"
  echo "TTRVBENCH_VALID_Q: no (1 checks failed)"
  echo "  NOT MONOTONE: q/t1/q_nop thread 0: n=1 -> 14 cycles, n=2 -> 13 cycles"
} > "$TMP/q_and_r.out"
r="$(rv_verdict "$TMP/q_and_r.out" rv)"
check "rv, phase Q plus another phase" SUSPECT "$r"

# A pair is evidence about its SMALLER burst: either point may be the one that
# moved, and only the smaller can be inside the documented scatter band. This
# is the 2026-08-09 22:24 `rv-cross` line, which was graded SUSPECT "at n = 32"
# on a pair whose n = 32 point had not moved at all.
{ echo "TTRVBENCH_VALID_R: yes"
  echo "TTRVBENCH_VALID_Q: no (1 checks failed)"
  echo "  NOT MONOTONE: q/t1/q_loop_adddmareg thread 1: n=16 -> 70 cycles, n=32 -> 48 cycles"
} > "$TMP/q_pair.out"
r="$(rv_verdict "$TMP/q_pair.out" rv)"
check "rv, phase Q pair is graded on its smaller burst" MEANINGFUL "$r"

echo
echo "== the remaining probes"

{ echo "probe variant unit thr cyc/block cyc/instr R2"
  echo "ttbench_a t1 FPU 1 64.00 1.000 1.0000"
  echo "TTBENCH_VALID_A: yes"
} > "$TMP/tensix_flat.out"
r="$(tensix_verdict "$TMP/tensix_flat.out" tensix)"
check "tensix when every probe reads 1.000" DEGENERATE "$r"

{ echo "probe variant unit thr cyc/block cyc/instr R2"
  echo "ttbench_a t1 FPU 1 128.00 2.000 1.0000"
  echo "TTBENCH_VALID_A: yes"
  echo "TTBENCH_VALID_B: no (3 checks failed)"
} > "$TMP/tensix_b.out"
r="$(tensix_verdict "$TMP/tensix_b.out" tensix)"
check "tensix when a phase other than A fails its gate" SUSPECT "$r"

{ echo "probe variant unit thr cyc/block cyc/instr R2"
  echo "ttbench_a t1 FPU 1 128.00 2.000 1.0000"
  echo "TTBENCH_VALID_A: yes"
} > "$TMP/tensix_ok.out"
r="$(tensix_verdict "$TMP/tensix_ok.out" tensix-warm)"
check "tensix-warm now carries a validity gate too" MEANINGFUL "$r"

printf 'phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles\nq,t1,1,a,-,1,1,1,1,40\nq,t1,1,a,-,1,1,2,1,40\n' > "$TMP/qdrain_flat.csv"
r="$(rv_qdrain_verdict "$TMP/qdrain_flat.csv")"
check "rv-qdrain when the sweep did not sweep" DEGENERATE "$r"
printf 'phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles\nq,t1,1,a,-,1,1,1,1,40\nq,t1,1,a,-,1,1,2,1,61\n' > "$TMP/qdrain_ok.csv"
r="$(rv_qdrain_verdict "$TMP/qdrain_ok.csv")"
check "rv-qdrain on a sweep that swept" COLLECTED "$r"

printf 'phase,variant,probe\ng,t1,g_1536\n' > "$TMP/gset1.csv"
cp "$TMP/gset1.csv" "$TMP/gset2.csv"
: > "$TMP/gset.out"
r="$(rv_gset_verdict "$TMP/gset.out" "$TMP/gset1.csv" "$TMP/gset2.csv")"
check "rv-gset when both gsets built the same footprint" DEGENERATE "$r"
printf 'phase,variant,probe\ng,t1,g_1792\n' > "$TMP/gset2.csv"
r="$(rv_gset_verdict "$TMP/gset.out" "$TMP/gset1.csv" "$TMP/gset2.csv")"
check "rv-gset when the two footprints differ" COLLECTED "$r"

# Phase G grew to five compile-time sets on 2026-08-09 (4608 and 5632 B), so the
# session now runs four gsets and the verdict takes all four. A define that
# fails to reach the build makes them ALL identical, which is why the check is
# pairwise rather than adjacent: an adjacent-only check catches that by luck of
# ordering, and a duplicate anywhere is the same bug.
printf 'phase,variant,probe\ng,t1,g_1152\n' > "$TMP/gset3.csv"
printf 'phase,variant,probe\ng,t1,g_1408\n' > "$TMP/gset4.csv"
r="$(rv_gset_verdict "$TMP/gset.out" "$TMP/gset1.csv" "$TMP/gset2.csv" "$TMP/gset3.csv" "$TMP/gset4.csv")"
check "rv-gset over four distinct gsets" COLLECTED "$r"
check_says "rv-gset names the five measured footprints" "4608, 5120, 5632, 6144 and 7168" "$r"
cp "$TMP/gset1.csv" "$TMP/gset4.csv"   # a duplicate at the far END of the list
r="$(rv_gset_verdict "$TMP/gset.out" "$TMP/gset1.csv" "$TMP/gset2.csv" "$TMP/gset3.csv" "$TMP/gset4.csv")"
check "rv-gset catches a duplicate that is not adjacent" DEGENERATE "$r"

echo
echo "== tensix-rdcfg: the distance is graded by controls that are not themselves"

# Every check below is exercised in BOTH directions -- once with a fixture that
# passes it and once with a fixture that trips it -- because this suite has
# already lost a card run to a validity check that could never succeed.
#
# The fixtures are the `TTBENCH_*:` tag lines the benchmark prints, not a
# scraped table: the previous grader read a five-field table out of the prose
# beside a seven-field summary, and a table that gains a column silently changes
# what it reads.
#
# WHAT IS GRADED NOW. A Blackhole card ran the C12 construction on 2026-08-12
# and measured all three arms at 2.9682 cycles/pair -- identical to four decimal
# places -- because BlackholeA0 ConfigurationUnit.md tabulates RDCFG's ">= 2"
# under a column headed **Latency** with IPC 1, so a busy-condition was never
# going to see it. The reading is now the smallest producer-to-consumer
# separation at which the value RDCFG wrote becomes visible.
_mk_rdcfg() { # file, dmin, reps, stale-ctl, fresh-ctl, n_mark, n_other, counts, dep
  { echo "probe          variant unit   thr    cyc/block  cyc/instr      R^2"
    echo "loop_overhead  t1      -      1        64.00      0.000   1.0000"
    echo "TTBENCH_CFGLAT_OCC: 0.998 0.998 0.998"
    echo "TTBENCH_CFGLAT_STALLCOST: 1.970"
    echo "TTBENCH_CFGLAT_COND: C12 CFGEXU 0x1000"
    echo "TTBENCH_CFGLAT_DIFF: 0.000 0.000"
    echo "TTBENCH_DEP_DIFF: ${9:-0.0000}"
    echo "TTBENCH_DEP_PARTS: 2.0000 1.0000 1.0000"
    echo "TTBENCH_VIS_CONTROLS: $4 $5 $6 $7"
    echo "TTBENCH_VIS_DMIN: $2 $3"
    echo "TTBENCH_VIS_COUNTS: $8"
    echo "TTBENCH_VALID_A: yes"
  } > "$1"
}
# The C12 liveness control, which lives in its own run because it needs a t3
# launch and t3 series are contended by construction.
_mk_c12() { # file, d(t1), d(t3)
  { echo "TTBENCH_C12_LIVE: $2 $3 t3"; echo "TTBENCH_VALID_A: yes"; } > "$1"
}

# THE READING THE PROBE EXISTS FOR. Both controls moved, no marker and no
# unexplained observation, no mixture, and the value RDCFG wrote is invisible at
# separation 1 and visible at 2 in every repetition. That is exactly what
# ConfigurationUnit.md's ">= 2 cycles" predicts.
_mk_rdcfg "$TMP/rdcfg_ok.out" 2 64 stale-ok fresh-ok 0 0 "0 64 64 64"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_ok.out")"
check "tensix-rdcfg when the value appears at separation 2" MEANINGFUL "$r"
check_says "tensix-rdcfg refuses to call it provenance" "CORROBORATION" "$r"
check_says "tensix-rdcfg reports it as a lower bound" "LOWER BOUND" "$r"

# THE SAME RUN, with the C12 liveness control beside it in both of its states.
# Neither changes the distance; both change what the 0.0000 of slots 22-25 MEANS,
# and that is the whole reason the control exists.
_mk_c12 "$TMP/c12_live.out" 1.970 41.200
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_ok.out" "$TMP/c12_live.out")"
check "tensix-rdcfg with a C12 control that moved" MEANINGFUL "$r"
check_says "tensix-rdcfg says the busy-condition was blind, not absent" "structurally invisible" "$r"
_mk_c12 "$TMP/c12_dead.out" 1.970 1.968
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_ok.out" "$TMP/c12_dead.out")"
check "tensix-rdcfg with a C12 control that did not move" MEANINGFUL "$r"
check_says "tensix-rdcfg says C12 observed nothing" "say nothing about RDCFG at all" "$r"
# The C12 control has its own run and its own gate, because it needs a t3
# launch and t3 series are contended by construction. Two untrustworthy slopes
# differenced are not evidence for either explanation -- and this fires on real
# output: the tt-sim validation run at --blocks 1 came back nonlinear at t3.
{ _mk_c12 "$TMP/c12_ungated.out" 1.970 41.200
  sed -i 's/TTBENCH_VALID_A: yes/TTBENCH_VALID_A: no (2 checks failed)/' "$TMP/c12_ungated.out"; }
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_ok.out" "$TMP/c12_ungated.out")"
check "tensix-rdcfg when the C12 run failed its own gate" MEANINGFUL "$r"
check_says "tensix-rdcfg refuses the C12 reading" "still open" "$r"

# CONTROL 2 tripped: the two no-RDCFG arms did not read back their own seeds, so
# a STALE observation is not representable and no count means anything. The
# distance is left at the MEANINGFUL value so the check cannot be passing for
# the wrong reason.
_mk_rdcfg "$TMP/rdcfg_nostale.out" 2 64 stale-FAILED fresh-ok 0 0 "0 64 64 64"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_nostale.out")"
check "tensix-rdcfg when a stale reading is not representable" SUSPECT "$r"
check_says "tensix-rdcfg names the stale control" "STALE control failed" "$r"

# CONTROL 3 tripped: the two far arms did not agree on one value distinct from
# both seeds, so a FRESH observation is not representable either.
_mk_rdcfg "$TMP/rdcfg_nofresh.out" 2 64 stale-ok fresh-FAILED 0 0 "0 64 64 64"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_nofresh.out")"
check "tensix-rdcfg when a fresh reading is not representable" SUSPECT "$r"
check_says "tensix-rdcfg names the fresh control" "FRESH control failed" "$r"

# CONTROL 4a tripped: some repetition's consumer never ran, so the sequence is
# not doing what it says.
_mk_rdcfg "$TMP/rdcfg_mark.out" 2 64 stale-ok fresh-ok 3 0 "0 64 64 64"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_mark.out")"
check "tensix-rdcfg when the consumer did not always run" SUSPECT "$r"
_mk_rdcfg "$TMP/rdcfg_other.out" 2 64 stale-ok fresh-ok 0 2 "0 64 64 64"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_other.out")"
check "tensix-rdcfg on an unexplained observation" SUSPECT "$r"

# CONTROL 4b tripped: a MIXTURE at one separation. The RISC-V front end did not
# deliver the sequence at one instruction per cycle, so the separation in issue
# slots is not the separation in cycles and the threshold is not a latency.
_mk_rdcfg "$TMP/rdcfg_mixed.out" 3 64 stale-ok fresh-ok 0 0 "0 37 64 64"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_mixed.out")"
check "tensix-rdcfg on a mixture at one separation" SUSPECT "$r"
check_says "tensix-rdcfg names the delivery rate" "one instruction per cycle" "$r"

# THE NULL, and it is a distinct outcome rather than a smaller MEANINGFUL: the
# value is already visible in the very next issue slot, so this construction
# cannot resolve anything below 1 and the documented ">= 2" is unreached. It is
# not refuted, and the verdict has to say so.
_mk_rdcfg "$TMP/rdcfg_one.out" 1 64 stale-ok fresh-ok 0 0 "64 64 64 64"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_one.out")"
check "tensix-rdcfg when the value is visible immediately" DEGENERATE "$r"
check_says "tensix-rdcfg says the quantity is unreached, not refuted" "UNREACHED" "$r"

# The threshold is outside the swept range: never fresh at any separation, while
# the far-separation control shows it does become visible eventually.
_mk_rdcfg "$TMP/rdcfg_never.out" 0 64 stale-ok fresh-ok 0 0 "0 0 0 0"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_never.out")"
check "tensix-rdcfg when no swept separation is enough" DEGENERATE "$r"

# THE TIMING FORM MOVED, which would make it the measurement and a better one --
# a dependent consumer costing more than an independent one IS the latency in
# cycles. It contradicts the documented absence of an interlock, so the verdict
# has to say that too rather than quietly preferring the number.
_mk_rdcfg "$TMP/rdcfg_interlock.out" 2 64 stale-ok fresh-ok 0 0 "0 64 64 64" 2.0100
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_interlock.out")"
check "tensix-rdcfg when the dependence pair moves" MEANINGFUL "$r"
check_says "tensix-rdcfg flags the contradiction with the doc" "contradicts" "$r"

# CONTROL 1: the gate condemns the whole phase, so no slope in the run is usable
# -- and the visibility sweep shares its launch with them.
{ _mk_rdcfg "$TMP/rdcfg_bad.out" 2 64 stale-ok fresh-ok 0 0 "0 64 64 64"
  sed -i 's/TTBENCH_VALID_A: yes/TTBENCH_VALID_A: no (2 checks failed)/' "$TMP/rdcfg_bad.out"; }
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_bad.out")"
check "tensix-rdcfg when phase A failed its gate" SUSPECT "$r"

# The sweep was not asked for: zero repetitions is "did not run", not "ran and
# found nothing".
_mk_rdcfg "$TMP/rdcfg_noreps.out" 0 0 stale-ok fresh-ok 0 0 "0 0 0 0"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_noreps.out")"
check "tensix-rdcfg when the sweep did not run" UNCLEAR "$r"

# A binary that predates the visibility probe prints no VIS tag. It must not be
# graded by the C12 pair it does print -- that pair measured 0.0000 on a card
# for a reason the ISA documentation gives in the instruction's own Performance
# section.
{ _mk_rdcfg "$TMP/rdcfg_old.out" 2 64 stale-ok fresh-ok 0 0 "0 64 64 64"
  sed -i '/^TTBENCH_VIS_DMIN:/d' "$TMP/rdcfg_old.out"; }
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_old.out")"
check "tensix-rdcfg on a binary predating the visibility probe" UNCLEAR "$r"

# Wormhole: the benchmark drops the C12 slots and says so, because Wormhole's
# fifteen condition bits name no unit RDCFG runs on. Nothing is missing there --
# WormholeB0/RDCFG.md blocks the issuing thread, so the ">= 2" is an occupancy.
{ echo "note: dropping probes 22-25 and 28-29 (the C12 slots) on wormhole."
  echo "TTBENCH_VALID_A: yes"; } > "$TMP/rdcfg_wh.out"
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_wh.out")"
check "tensix-rdcfg on wormhole, where the condition does not exist" SKIPPED "$r"
check_says "tensix-rdcfg says probe 14 already has it there" "OCCUPANCY" "$r"

echo
echo "== dram: a flat measurement is only a result when the CONTROL moved"

_dram_hdr='# dramratebench arch=blackhole magic=0x44524231 grid=13x10 clock_mhz=1350 banks=8
arm,repeat,point,num_readers,num_tx,tx_bytes,bytes_per_reader,total_bytes,max_cycles,min_cycles,agg_bytes_per_cycle,agg_gb_per_s,per_reader_bytes_per_cycle,distinct_banks,distinct_dram_cores,tags_ok,max_barrier_spins,measured_readers'

# The headline case: one channel flat at ~24 B/cycle while the fan-out control
# scales nearly linearly. This is what silicon should look like if the
# endpoint-occupancy term shipped on 2026-08-09 is right.
{ echo "$_dram_hdr"
  echo "onechan,0,0,1,256,4096,1048576,1048576,44000,44000,23.8309,32.171,23.8309,1,1,1,0,1"
  echo "fanchan,0,1,1,256,4096,1048576,1048576,44000,44000,23.8309,32.171,23.8309,1,1,1,0,1"
  echo "onechan,0,2,12,256,4096,1048576,12582912,525000,520000,23.9674,32.356,1.9973,1,1,12,900,12"
  echo "fanchan,0,3,12,256,4096,1048576,12582912,46000,44000,273.5416,369.28,22.7951,8,8,12,850,12"
} > "$TMP/dram_good.csv"
r="$(dram_verdict "$TMP/dram.out" "$TMP/dram_good.csv")"
check "dram: control scales, one channel flat" MEANINGFUL "$r"
check_says "dram says the endpoint saturated" "endpoint is what cost the difference" "$r"
check_says "dram refuses to call it provenance" "never provenance" "$r"

# The nocread mistake, replayed in this probe's own shape: BOTH arms flat. The
# one-channel curve is exactly as flat as the good case above, and means nothing.
sed 's/^fanchan,0,3,12,.*$/fanchan,0,3,12,256,4096,1048576,12582912,520000,515000,24.1979,32.667,2.0165,8,8,12,850,12/' \
  "$TMP/dram_good.csv" > "$TMP/dram_flatcontrol.csv"
r="$(dram_verdict "$TMP/dram.out" "$TMP/dram_flatcontrol.csv")"
check "dram when the fan-out control did NOT move" DEGENERATE "$r"
check_says "dram names the control rather than the measurement" "CONTROL did not move" "$r"

# One reader read something other than its bank's tag. Every rate in the file is
# from an endpoint that may not be the one its row names.
sed 's/,1,1,12,900,12$/,1,1,11,900,12/' "$TMP/dram_good.csv" > "$TMP/dram_badtag.csv"
r="$(dram_verdict "$TMP/dram.out" "$TMP/dram_badtag.csv")"
check "dram when a reader missed its target bank" DEGENERATE "$r"
check_says "dram says the endpoint is not the one named" "endpoint the plan names" "$r"

# Nobody waited at the barrier in any multi-reader point, so the readers ran one
# after another. That produces a flat aggregate from an experiment that did not
# happen, which is the failure mode hardest to see in the numbers alone.
sed 's/,900,12$/,0,12/; s/,850,12$/,0,12/' "$TMP/dram_good.csv" > "$TMP/dram_noverlap.csv"
r="$(dram_verdict "$TMP/dram.out" "$TMP/dram_noverlap.csv")"
check "dram when no reader ever waited at the barrier" DEGENERATE "$r"
check_says "dram names the overlap failure" "did not overlap" "$r"

# The evidenced negative: the control moved AND one channel KEPT UP with it. If
# silicon reads this, the term shipped on 2026-08-09 is wrong.
sed 's/^onechan,0,2,12,.*$/onechan,0,2,12,256,4096,1048576,12582912,46000,44000,273.5416,369.28,22.7951,1,1,12,900,12/' \
  "$TMP/dram_good.csv" > "$TMP/dram_refutes.csv"
r="$(dram_verdict "$TMP/dram.out" "$TMP/dram_refutes.csv")"
check "dram when one channel scales too" MEANINGFUL "$r"
check_says "dram states the term is refuted" "NOT serialised by it" "$r"

# THE CASE AN ABSOLUTE THRESHOLD GETS WRONG, and these are tt-sim's own numbers:
# Wormhole with the cost model on, four readers, onechan x2.21 against fanchan
# x3.97. The concentrated arm reaches 56% of what the same readers reach fanned
# out -- the endpoint plainly costing something -- and a rule of "onechan must
# stay under x1.5" calls that a refutation of the very term it is evidence for.
# The discriminator is the RATIO of the two scalings.
{ echo "$_dram_hdr"
  echo "onechan,0,0,1,16,512,8192,8192,936,936,8.7521,0.000,8.7521,1,1,1,0,1"
  echo "fanchan,0,1,1,16,512,8192,8192,936,936,8.7521,0.000,8.7521,1,1,1,0,1"
  echo "onechan,0,4,4,16,512,8192,32768,1696,1552,19.3208,0.000,4.8302,1,1,4,21,4"
  echo "fanchan,0,5,4,16,512,8192,32768,944,936,34.7119,0.000,8.6780,4,4,4,21,4"
} > "$TMP/dram_underpowered.csv"
r="$(dram_verdict "$TMP/dram.out" "$TMP/dram_underpowered.csv")"
check "dram on tt-sim's own Wormhole cost-model numbers" MEANINGFUL "$r"
check_says "dram reads a partial bound as a bound" "endpoint is what cost the difference" "$r"

# One reader count only -- the shape a simulator run has, because only the tiles
# in TT_SIM_TENSIX_COORDS exist. There is no control to move.
{ echo "$_dram_hdr"
  echo "onechan,0,0,1,16,512,8192,8192,1400,1400,5.8514,7.899,5.8514,1,1,1,0,1"
  echo "fanchan,0,1,1,16,512,8192,8192,1400,1400,5.8514,7.899,5.8514,1,1,1,0,1"
} > "$TMP/dram_single.csv"
r="$(dram_verdict "$TMP/dram.out" "$TMP/dram_single.csv")"
check "dram with a single reader count (the --sim shape)" DEGENERATE "$r"
check_says "dram explains the simulator case" "TT_SIM_TENSIX_COORDS" "$r"

# A schema this session does not know how to grade must say so rather than
# grading what it can find.
printf 'a,b,c\n1,2,3\n' > "$TMP/dram_alien.csv"
r="$(dram_verdict "$TMP/dram.out" "$TMP/dram_alien.csv")"
check "dram on an unknown schema" UNCLEAR "$r"

echo
echo "== build provenance: which tt-metal a build tree was made against"

# The failure this replays: TT_METAL_HOME was changed between sessions while
# src/build/CMakeCache.txt still held the old checkout's paths. cmake reused the
# cache, the binary compiled against one tt-metal and ran against another, and
# the session died at rc=127 on a symbol lookup AFTER the board reset. The
# pre-flight had said "ok build/dramratebench is current" -- it checked that a
# build existed, which is a different claim.
#
# No tt-metal and no cmake are needed to test it: the check reads a stamp or a
# CMakeCache, and both are text.
# shellcheck source=build_provenance.sh
. "$PB/build_provenance.sh"

BPT="$TMP/bp"
mkdir -p "$BPT"

# Two "checkouts". METAL_A is a real git repo so the revision half of the check
# has something to read; METAL_B is a plain directory, which is the case where
# only the path can be compared.
METAL_A="$BPT/tt-metal-a"
METAL_B="$BPT/tt-metal-b"
mkdir -p "$METAL_A/build/lib" "$METAL_B/build/lib"
( cd "$METAL_A" && git init -q . && git -c user.email=t@t -c user.name=t commit -q \
    --allow-empty -m one ) >/dev/null 2>&1
REV_A="$(git -C "$METAL_A" rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"

bp_tree() { # <name> [stamped-root] [stamped-rev] -- a src dir with a build tree
  local dir="$BPT/$1"
  mkdir -p "$dir/build"
  # Every build tree has a cache, and it names tt-metal's dependency configs.
  printf 'CMAKE_CACHEFILE_DIR:INTERNAL=%s/build\numd_DIR:PATH=%s/build/lib/cmake/umd\n' \
    "$dir" "${2:-$METAL_A}" > "$dir/build/CMakeCache.txt"
  if [ -n "${3:-}" ]; then
    printf 'root=%s\nrev=%s\n' "${2:-$METAL_A}" "$3" > "$dir/build/$BP_STAMP_NAME"
  fi
  printf '%s\n' "$dir"
}

bp_out=""
bp_try() { # <src-dir> [mode] -- with TT_METAL_HOME already exported
  bp_out="$(bp_check_build "$@" 2>&1)" && bp_rc=0 || bp_rc=1
}

check_rc() { # description, expected rc, actual rc
  if [ "$3" = "$2" ]; then
    pass=$((pass + 1)); printf 'ok   %-58s rc=%s\n' "$1" "$3"
  else
    fail=$((fail + 1)); printf 'FAIL %-58s want rc=%s, got rc=%s\n     %s\n' "$1" "$2" "$3" "$bp_out"
  fi
}

unset TT_METAL_RUNTIME_ROOT
export TT_METAL_HOME="$METAL_A"

t="$(bp_tree matching "$METAL_A" "$REV_A")"
bp_try "$t"
check_rc "a tree built against the configured tt-metal" 0 "$bp_rc"

# THE DEFECT. Same tree, TT_METAL_HOME moved.
export TT_METAL_HOME="$METAL_B"
bp_try "$t"
check_rc "the same tree once TT_METAL_HOME moved" 1 "$bp_rc"
check_says "the refusal names the tt-metal it was built against" "$METAL_A" "$bp_out"
check_says "the refusal names the tt-metal now configured" "$METAL_B" "$bp_out"
check_says "the refusal tells the operator what to remove" "rm -rf $t/build" "$bp_out"

# A tree from before the stamp existed is still checkable: cmake caches the
# config dir of every tt-metal dependency TT::Metalium pulls in. Waving legacy
# trees through would exempt exactly the trees a card box already has.
legacy="$(bp_tree legacy "$METAL_A")"
bp_try "$legacy"
check_rc "an unstamped tree, read from its CMakeCache, mismatching" 1 "$bp_rc"
export TT_METAL_HOME="$METAL_A"
bp_try "$legacy"
check_rc "an unstamped tree, read from its CMakeCache, matching" 0 "$bp_rc"

# `/x/tt-metal` and `/x/tt-metal/` are one checkout. A path-string compare that
# called them different would refuse every correct session instead.
export TT_METAL_HOME="$METAL_A/"
bp_try "$t"
check_rc "a trailing slash is not a different checkout" 0 "$bp_rc"
export TT_METAL_HOME="$METAL_A"

# TT_METAL_RUNTIME_ROOT is what the CMakeLists actually prefer, so it must be
# what is checked -- otherwise the check reads a variable cmake ignored.
export TT_METAL_RUNTIME_ROOT="$METAL_B"
bp_try "$t"
check_rc "TT_METAL_RUNTIME_ROOT wins over TT_METAL_HOME, as cmake does" 1 "$bp_rc"
unset TT_METAL_RUNTIME_ROOT

# Same checkout, moved on. A rebuild picks the new headers up, so it warns; with
# the build skipped nothing picks it up and the old binary meets the new
# library, which is the rc=127 failure one release apart instead of one path.
moved="$(bp_tree moved "$METAL_A" 0000000000)"
bp_try "$moved"
check_rc "the same checkout at a different rev, rebuilding" 0 "$bp_rc"
check_says "a rev change warns and says the rebuild covers it" "rebuild below" "$bp_out"
bp_try "$moved" skip-build
check_rc "the same checkout at a different rev, --skip-build" 1 "$bp_rc"
check_says "the --skip-build refusal names the remedy" "without --skip-build" "$bp_out"

# Nothing to check against is a failure, not a pass by absence.
unset TT_METAL_HOME
bp_try "$t"
check_rc "no TT_METAL_HOME and no TT_METAL_RUNTIME_ROOT" 1 "$bp_rc"
export TT_METAL_HOME="$METAL_A"

# A tree that does not exist yet cannot be stale.
bp_try "$BPT/never-built"
check_rc "no build tree yet" 0 "$bp_rc"

# The stamp is what makes the revision half work, so record-then-check must
# round-trip -- including the revision, which no CMakeCache carries.
fresh="$BPT/fresh"
mkdir -p "$fresh/build"
bp_record_build "$fresh"
check_says "a recorded stamp names the checkout" "root=$METAL_A" "$(cat "$fresh/build/$BP_STAMP_NAME")"
check_says "a recorded stamp names the revision" "rev=$REV_A" "$(cat "$fresh/build/$BP_STAMP_NAME")"
bp_try "$fresh"
check_rc "a freshly recorded tree checks clean" 0 "$bp_rc"
export TT_METAL_HOME="$METAL_B"
bp_try "$fresh"
check_rc "a freshly recorded tree refuses the other checkout" 1 "$bp_rc"

# Every card runner must actually call it. A library nothing sources is a
# library that fixes nothing, and the list of runners is the thing that was
# wrong before: the check existed nowhere rather than in some places.
for r in dramratebench/run_card.sh dramratebench/run_card_write.sh \
         nocbench/run_card.sh nocevbench/run_card.sh nocreadbench/run_card.sh \
         mechbench/run_card.sh riscvbench/run_card.sh tensixbench/run_card.sh \
         energybench/run_card.sh run_card_session.sh run_paper_session.sh \
         run.sh mechbench/run_sim.sh nocevbench/run_sim.sh; do
  if grep -q 'build_provenance.sh' "$PB/$r" \
     && grep -qE 'bp_(check|require)_build' "$PB/$r" \
     && grep -q 'bp_record_build' "$PB/$r"; then
    pass=$((pass + 1)); printf 'ok   %-58s sources and calls it\n' "$r"
  else
    fail=$((fail + 1)); printf 'FAIL %-58s does not check its build provenance\n' "$r"
  fi
done

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
