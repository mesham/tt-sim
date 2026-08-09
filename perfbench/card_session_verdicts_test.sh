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
echo "== tensix-rdcfg: the difference is graded by a control that is not itself"

# The summary table this parses is seven fields wide; the latency table is five.
# Both are written by the same run, so a check that matched loosely would read
# one as the other.
_mk_rdcfg() { # file, setdma_bare, setdma_stall, rdcfg_bare, difference
  { echo "probe          variant unit   thr    cyc/block  cyc/instr      R^2"
    echo "loop_overhead  t1      -      0        64.00      0.000   1.0000"
    echo "SETDMAREG      t1      THCON  0       $2.00      $2   1.0000"
    echo "RDCFG          t1      CFG    0       $4.00      $4   1.0000"
    echo "SETDMA_STALL   t1  THCON-LAT  0       $3.00      $3   1.0000"
    echo "RDCFG_STALL    t1    CFG-LAT  0       $3.00      $3   1.0000"
    echo "phase A: RDCFG latency, as (RDCFG + STALLWAIT) - (SETDMAREG + the same STALLWAIT)"
    echo "  variant thr   rdcfg/pair    base/pair   difference"
    echo "  t1      0          9.000        7.000       $5"
    echo "TTBENCH_VALID_A: yes"
  } > "$1"
}

# tt-sim: the stall is vacuous, so SETDMAREG+STALLWAIT costs the same as
# SETDMAREG bare. The difference could be anything and would still mean nothing.
_mk_rdcfg "$TMP/rdcfg_vacuous.out" 1.000 1.000 1.000 +2.000
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_vacuous.out")"
check "tensix-rdcfg when the STALLWAIT is free" DEGENERATE "$r"
check_says "tensix-rdcfg names the vacuous stall" "STALLWAIT is free" "$r"

# The baseline op does not cost what the measured op costs bare, so part of the
# difference is occupancy. Not a latency, whatever its size.
_mk_rdcfg "$TMP/rdcfg_confounded.out" 1.000 7.000 3.500 +2.000
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_confounded.out")"
check "tensix-rdcfg when the two bare occupancies differ" SUSPECT "$r"
check_says "tensix-rdcfg says the difference is not latency" "OCCUPANCY" "$r"

# A live stall and a real difference clear of the half-cycle floor: the reading
# the probe exists for.
_mk_rdcfg "$TMP/rdcfg_ok.out" 1.000 7.000 1.000 +2.000
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_ok.out")"
check "tensix-rdcfg on a live stall with a difference" MEANINGFUL "$r"
check_says "tensix-rdcfg refuses to call it provenance" "CORROBORATION" "$r"

# A live stall and a SUB-FLOOR difference. 0.381 is what tt-sim actually reads:
# the STALLWAIT instruction costs cycles there, but its condition is answered
# "satisfied" (TRISC_CFG is mapped on neither arch), so the residue is fit noise.
# Half a cycle per pair cannot tell a latency documented ">= 2" from none, so
# this is the NULL. Quoting 0.381 as a lower bound would be the measurement
# grading itself.
_mk_rdcfg "$TMP/rdcfg_subfloor.out" 1.000 4.020 1.000 +0.381
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_subfloor.out")"
check "tensix-rdcfg on the sub-floor difference tt-sim reads" DEGENERATE "$r"
check_says "tensix-rdcfg names the floor" "half-cycle floor" "$r"
_mk_rdcfg "$TMP/rdcfg_zero.out" 1.000 7.000 1.000 +0.000
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_zero.out")"
check "tensix-rdcfg on a live stall with a zero difference" DEGENERATE "$r"

# The gate condemns the whole phase, so no slope in the run is usable.
{ _mk_rdcfg "$TMP/rdcfg_bad.out" 1.000 7.000 1.000 +2.000
  sed -i 's/TTBENCH_VALID_A: yes/TTBENCH_VALID_A: no (2 checks failed)/' "$TMP/rdcfg_bad.out"; }
r="$(tensix_rdcfg_verdict "$TMP/rdcfg_bad.out")"
check "tensix-rdcfg when phase A failed its gate" SUSPECT "$r"

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
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
