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
r="$(rv_gset_verdict "$TMP/gset1.csv" "$TMP/gset2.csv" "$TMP/gset.out")"
check "rv-gset when both gsets built the same footprint" DEGENERATE "$r"
printf 'phase,variant,probe\ng,t1,g_1792\n' > "$TMP/gset2.csv"
r="$(rv_gset_verdict "$TMP/gset1.csv" "$TMP/gset2.csv" "$TMP/gset.out")"
check "rv-gset when the two footprints differ" COLLECTED "$r"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
