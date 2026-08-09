#!/usr/bin/env bash
# The card session's verdict logic, as pure functions of the files a probe left
# behind. Sourced by run_card_session.sh; exercised by card_session_verdicts_test.sh
# against the CSVs a real card actually returned.
#
# WHY THIS IS A SEPARATE FILE
# ---------------------------
# The 2026-08-09 Blackhole session printed two MEANINGFUL verdicts on controls
# that never moved:
#
#   nocread  outstanding_max = 72 in all 129 rows -- at num_tx = 4 as well as
#            128 -- and the session read that as "no initiator limit", which it
#            reported as RETIRING the credit-limit term.
#   cmdbuf   cmdbuf_avail_rest and cmdbuf_avail_busy both 0x00000000 in all 129
#            rows, and the session read that as "the number the ISA docs
#            withhold" because its only check was "not 0xFFFFFFFF".
#
# Both are the same bug: a check that a value is PRESENT, or in the right
# FORMAT, standing in for a check that the thing being measured RESPONDED. A
# session cannot be trusted to catch that at the card unless the check can be
# run away from the card, against files, on demand -- so the checks live here
# and have a test.
#
# THE RULE EVERY FUNCTION BELOW OBEYS
# -----------------------------------
# A probe may report MEANINGFUL only if something in its own output moved off
# the value it would have had if the instrument had measured nothing. Presence
# of a file, presence of a column, a value that merely is not a known sentinel,
# and the ABSENCE of a failure string are all disqualified as evidence.
#
# Each function echoes one line, `STATUS|text`. Statuses are the session's:
# MEANINGFUL, COLLECTED, DEGENERATE, DEFERRED, SKIPPED, FAILED, SUSPECT, UNCLEAR.
#
# SPDX-License-Identifier: Apache-2.0

# --------------------------------------------------------------- small helpers

# Column index of a named CSV field, 1-based, or empty. The header is the first
# line that is not a `#` comment. Looking columns up by name is what lets one
# verdict grade both the NRB1 and NRB2 nocreadbench schemas.
_csv_col() { # file, column-name
  awk -F, -v want="$2" '
    /^#/ { next }
    { for (i = 1; i <= NF; i++) { gsub(/^[ \t]+|[ \t\r]+$/, "", $i); if ($i == want) { print i; exit } } exit }
  ' "$1" 2>/dev/null
}

# Does a report file hold a report, or a Python traceback? A `.report.txt` whose
# whole content is a ModuleNotFoundError still reads as a result to anything
# that greps it for the absence of a word, which is exactly how the card
# session turned a failed report generator into two MEANINGFUL verdicts.
_report_is_broken() { # file
  [ -s "$1" ] || return 0
  grep -qE 'Traceback \(most recent call last\)|ModuleNotFoundError|Error while finding module|^[A-Za-z_.]*Error:' "$1"
}

# ------------------------------------------------------------------- nocread
#
# Three checks, cheapest and most decisive first.
#
#  1. A count of read requests in flight cannot exceed the burst that produced
#     it. `outstanding_max > num_tx` in any row means the column is not an
#     occupancy, whatever else is true. This needs no baseline and no new
#     column, so it condemns an NRB1 file as readily as an NRB2 one -- which is
#     the point: the 2026-08-09 CSV is caught by this line alone.
#  2. If the CSV carries NRB2's `outstanding_delta` and `inflight_max`, the
#     occupancy must be non-zero somewhere.
#  3. Only then is the benchmark's own printed VERDICT allowed to speak.
nocread_verdict() { # out-file, csv
  local out="$1" csv="$2"
  if [ ! -s "$csv" ]; then
    echo "FAILED|no CSV; send nocread.out"
    return
  fi
  local c_max c_ntx c_delta c_infl c_infl_rest impossible
  c_max="$(_csv_col "$csv" outstanding_max)"
  c_ntx="$(_csv_col "$csv" num_tx)"
  if [ -z "$c_max" ] || [ -z "$c_ntx" ]; then
    echo "UNCLEAR|the CSV has no outstanding_max/num_tx columns; the schema is not one this session knows how to grade"
    return
  fi
  c_delta="$(_csv_col "$csv" outstanding_delta)"
  # Which column can the "cannot exceed the burst" law be applied to? On an NRB2
  # CSV the raw counter is EXPECTED to exceed the burst -- that is the whole
  # point of the rest sample, and on the 2026-08-09 card the baseline was ~71,
  # so a raw check would reject every healthy run of a 64-request burst. The law
  # belongs to the baselined quantity. Only when there is no rest column (NRB1,
  # where a raw counter was all the probe recorded) does it apply to the max.
  local c_law law_name
  if [ -n "$c_delta" ]; then c_law="$c_delta"; law_name="outstanding_delta (max minus this point's own rest sample)"
  else c_law="$c_max"; law_name="outstanding_max, and this CSV predates the rest-sample columns"; fi
  impossible="$(awk -F, -v m="$c_law" -v n="$c_ntx" '
      /^#/ { next }
      !hdr { hdr = 1; next }
      $m + 0 > $n + 0 { bad++; if (worst == "" || $m + 0 > worst) { worst = $m + 0; at = $n + 0 } }
      END { if (bad) printf "%d %d %d", bad, worst, at }' "$csv")"
  if [ -n "$impossible" ]; then
    # shellcheck disable=SC2086
    set -- $impossible
    echo "DEGENERATE|$law_name reads $2 during a burst of $3 requests, in $1 row(s). A count of requests in flight cannot exceed the burst that produced it, so the column is not an occupancy -- it is a live register read with no baseline subtracted. Nothing about an initiator limit can be concluded from this file, in either direction"
    return
  fi
  c_infl="$(_csv_col "$csv" inflight_max)"
  c_infl_rest="$(_csv_col "$csv" inflight_rest)"
  if [ -n "$c_delta" ] && [ -n "$c_infl" ]; then
    local moved
    moved="$(awk -F, -v d="$c_delta" -v f="$c_infl" -v r="${c_infl_rest:-0}" '
        /^#/ { next }
        !hdr { hdr = 1; next }
        { if ($d + 0 > 0) dmoved++; if ($f + 0 > 0) fmoved++; if (r > 0 && $r + 0 != 0) dirty++ }
        END { printf "%d %d %d", dmoved + 0, fmoved + 0, dirty + 0 }' "$csv")"
    # shellcheck disable=SC2086
    set -- $moved
    if [ "$3" -gt 0 ]; then
      echo "DEGENERATE|$3 point(s) saw read requests already in flight BEFORE issuing anything, and the kernel starts after a barrier. The status block is not being read the way the probe assumes"
      return
    fi
    if [ "$1" = 0 ] && [ "$2" = 0 ]; then
      echo "DEGENERATE|neither the per-trid occupancy nor the trid-independent RD_REQ_SENT - RD_RESP_RECEIVED pair left zero in any point. EXPECTED against tt-sim, whose responses resolve inside the pump that issued them; on a card this is a broken run"
      return
    fi
    if [ "$1" = 0 ] || [ "$2" = 0 ]; then
      echo "DEGENERATE|the two in-flight instruments disagree about whether anything was in flight at all ($1 point(s) by the per-trid counter, $2 by the trid-independent pair). The one reading zero is not watching these reads"
      return
    fi
  fi
  # Only now is the benchmark's own verdict evidence.
  if grep -q "VERDICT: DEGENERATE" "$out" 2>/dev/null; then
    echo "DEGENERATE|the outstanding-request counter never moved. EXPECTED against tt-sim (its NIU queue is unbounded); on a card this is a broken run"
  elif grep -q "VERDICT: BOUNDED AT" "$out" 2>/dev/null; then
    # The benchmark PRINTS the falsifying test next to this verdict but cannot
    # run it -- it needs the whole `dist` sweep. Apply it here. A credit limit K
    # caps the rate at round_trip/K, so cycles_per_tx MUST rise with hops. If
    # the dist rows are flat, the peak occupancy is service time (Little's law:
    # it tracks transaction SIZE, not the burst) and is not what caps the rate.
    # On 2026-08-09 this branch awarded MEANINGFUL to "BOUNDED AT 13" whose 13
    # came from 3 rows at one transaction size while every other row read 2-3,
    # against dist rows flat to 1.5% over 13 hops.
    local c_hops c_cyc dist_spread
    c_hops="$(_csv_col "$csv" hops)"; c_cyc="$(_csv_col "$csv" cycles_per_tx)"
    dist_spread="$(awk -F, -v h="$c_hops" -v c="$c_cyc" '
        /^#/ { next }
        !hdr { hdr = 1; next }
        $1 == "dist" { v = $c + 0; if (lo == "" || v < lo) lo = v; if (v > hi) hi = v;
                       if (mh == "" || $h + 0 < mh) mh = $h + 0; if ($h + 0 > xh) xh = $h + 0 }
        END { if (lo != "" && lo > 0 && xh > mh) printf "%.1f %d %d", 100 * (hi - lo) / lo, mh, xh }' "$csv")"
    if [ -n "$dist_spread" ]; then
      # shellcheck disable=SC2086
      set -- $dist_spread
      if awk -v s="$1" 'BEGIN { exit !(s < 5) }'; then
        echo "DEGENERATE|the benchmark printed $(grep -m1 'VERDICT: BOUNDED AT' "$out" | sed 's/^ *//; s/ -- .*//'), but its own falsifying test refutes it: cycles_per_tx varies by only $1% across hops $2 to $3, and a credit limit K caps the rate at round_trip/K, which MUST rise with distance. The peak occupancy is service time tracking transaction size, not a ceiling. No initiator credit limit caps this rate"
        return
      fi
      echo "MEANINGFUL|$(grep -m1 'VERDICT: BOUNDED AT' "$out" | sed 's/^ *//') -- K is read off a register against each point's own rest sample, AND cycles_per_tx rises $1% across hops $2 to $3 as a credit limit requires"
      return
    fi
    echo "UNCLEAR|the benchmark printed BOUNDED AT, but the CSV has no dist rows to run its own falsifying test against. A peak occupancy alone is not a credit limit -- it is service time unless the rate rises with distance"
  elif grep -q "VERDICT: NO INITIATOR LIMIT" "$out" 2>/dev/null; then
    if [ -z "$c_delta" ]; then
      echo "DEGENERATE|the benchmark printed NO INITIATOR LIMIT, but this CSV predates the rest-sample columns, so that verdict was reached by comparing a RAW counter against the burst length. It is unsupported; rebuild nocreadbench and retake"
    else
      echo "MEANINGFUL|no initiator limit -- in-flight requests track the burst length against each point's own rest sample. This RETIRES the credit-limit term, which is a result"
    fi
  else
    echo "UNCLEAR|no VERDICT line found; read nocread.out by hand"
  fi
}

# -------------------------------------------------------------------- cmdbuf
#
# CMD_BUF_AVAIL is an occupancy, not a count of free slots: nothing in tt-metal
# reads it on Wormhole or Blackhole, the only register descriptor for it in the
# tree is Quasar's, its reset default there is 0x00000000, and its neighbour is
# CMD_BUF_OVFL. So zero AT REST is the correct reading and says nothing. Only a
# sample taken while a command buffer actually holds an entry can say anything,
# and the previous kernel took its "busy" sample after the issue loop had ended.
cmdbuf_verdict() { # csv
  local csv="$1"
  [ -s "$csv" ] || { echo "SKIPPED|needs the nocread probe to have run first"; return; }
  local c_rest c_busy c_max
  c_rest="$(_csv_col "$csv" cmdbuf_avail_rest)"
  c_busy="$(_csv_col "$csv" cmdbuf_avail_busy)"
  c_max="$(_csv_col "$csv" cmdbuf_avail_max)"
  if [ -z "$c_rest" ] || [ -z "$c_busy" ]; then
    echo "UNCLEAR|the CSV has no cmdbuf_avail columns; the kernel may not have been built for this part"
    return
  fi
  local stats
  stats="$(awk -F, -v r="$c_rest" -v b="$c_busy" -v x="${c_max:-0}" '
      /^#/ { next }
      !hdr { hdr = 1; next }
      { rows++
        if ($r == "0xFFFFFFFF") absent++
        if ($b != $r || (x > 0 && $x != $r)) moved++
        first = (first == "") ? $r : first }
      END { printf "%d %d %d %s", rows + 0, absent + 0, moved + 0, first }' "$csv")"
  # shellcheck disable=SC2086
  set -- $stats
  local rows="$1" absent="$2" moved="$3" first="$4"
  if [ "$rows" = 0 ]; then
    echo "UNCLEAR|the CSV has no data rows"
  elif [ "$absent" = "$rows" ]; then
    echo "DEGENERATE|reads the all-ones 'register absent' sentinel in every row, so the READ FAILED. EXPECTED against tt-sim, which models no command buffer. On a Blackhole card this means the build did not see CMD_BUF_AVAIL and the reading must be retaken -- do not send this as a result"
  elif [ "$moved" = 0 ]; then
    echo "DEGENERATE|CMD_BUF_AVAIL reads $first at rest and the identical value in every in-loop sample, in all $rows rows. The control did not move, so this says nothing about the command-buffer depth -- which was the whole point. Note that a register whose reset default is 0 and whose neighbour is CMD_BUF_OVFL is an OCCUPANCY: zero at rest is CORRECT and is not a depth"
  elif [ -z "$c_max" ]; then
    echo "SUSPECT|CMD_BUF_AVAIL differs between the rest and busy samples, but this CSV predates the in-loop cmdbuf_avail_max column, so the peak occupancy was never sampled. Rebuild nocreadbench and retake before quoting anything"
  else
    echo "MEANINGFUL|CMD_BUF_AVAIL moved off its rest value ($first) in $moved sample set(s). Its peak is a LOWER BOUND on the depth: only cmdbuf_ovfl_end leaving cmdbuf_ovfl_rest proves the buffer was driven to its limit"
  fi
}

# -------------------------------------------------------- tensixbench (both)
#
# The movement check here was already right -- a peak cyc/instr at the 1.000
# floor is the null and is caught. What was missing is the validity gate: the
# `tensix` probe grepped only `TTBENCH_VALID_A` while running every phase, and
# `tensix-warm` had no validity check at all.
_tensix_peak() { # out-file
  awk 'NF==7 && $1!="probe" && $1!="loop_overhead" && $6+0==$6 {
         if ($6+0 > m) m = $6+0 } END { printf "%.3f", m+0 }' "$1"
}

_tensix_failed_phases() { # out-file
  sed -n 's/^TTBENCH_VALID_\([A-Z]\): no.*/\1/p' "$1" 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
}

tensix_verdict() { # out-file, label
  local out="$1" label="${2:-tensix}" peak bad
  peak="$(_tensix_peak "$out")"
  if ! grep -q '^TTBENCH_VALID' "$out" 2>/dev/null; then
    echo "UNCLEAR|no TTBENCH_VALID lines in $label.out; the run did not reach its own gate"
    return
  fi
  if awk -v p="$peak" 'BEGIN { exit !(p <= 1.0005) }'; then
    echo "DEGENERATE|every probe reads 1.000 (peak cyc/instr $peak). EXPECTED against tt-sim with the cost model off (nothing back-pressures the issuing core); on a card it means the instrument measured nothing"
    return
  fi
  bad="$(_tensix_failed_phases "$out")"
  if [ -n "$bad" ]; then
    echo "SUSPECT|phase(s) $bad failed their linearity/monotonicity checks (peak cyc/instr $peak)"
    return
  fi
  if [ "$label" = tensix-warm ]; then
    echo "MEANINGFUL|peak cyc/instr $peak, every phase gated, fit starting at n=64 so the cold first burst is out. Diff these slopes against the tensix probe's -- the difference IS the warm-up bias the roadmap wants shrunk"
  else
    echo "MEANINGFUL|peak cyc/instr $peak is above 1.000 and every phase passed its gate; this is the second sample rung 3 needed"
  fi
}

# --------------------------------------------------------- riscvbench (both)
#
# Two fixes over the session that ran on 2026-08-09.
#
# 1. It greped the aggregate `TTRVBENCH_VALID:` line out of a file holding TWO
#    concatenated runs, so either run's failure condemned both. Each run now
#    writes its own .out and is graded on its own.
# 2. Phase Q. riscvbench/README.md documents that phase Q's cascade fails the
#    zero-tolerance monotonicity gate at small n ON SILICON -- "one cold,
#    once-only burst" scatters by 6-23 cycles -- and that is where the phase-S
#    measured tolerance came from. So a phase-Q-only failure whose every
#    complaint is at n <= 16 is EXPECTED and is not a reason to distrust the
#    run. This is not a blanket suppression: a phase-Q complaint at n > 16, or
#    any other phase failing, is still SUSPECT. The threshold is the README's
#    own ("n <= 16"), not a number picked to make this run pass.
_rv_failed_phases() { # out-file
  sed -n 's/^TTRVBENCH_VALID_\([A-Z]\): no.*/\1/p' "$1" 2>/dev/null | tr '\n' ' ' | sed 's/ $//'
}

# Largest burst length named by a phase-Q monotonicity complaint, or 0.
_rv_q_worst_n() { # out-file
  sed -n 's/.*NOT MONOTONE: q\/.*n=[0-9]* -> [0-9]* cycles, n=\([0-9]*\) ->.*/\1/p' "$1" 2>/dev/null |
    awk 'BEGIN { m = 0 } { if ($1 + 0 > m) m = $1 + 0 } END { print m }'
}

rv_verdict() { # out-file, label
  local out="$1" label="${2:-rv}" bad qn
  [ -s "$out" ] || { echo "FAILED|no $label.out"; return; }
  if grep -q "ALL of them read" "$out"; then
    echo "DEGENERATE|the live-instrument check says all four above-one-cycle probes read ~1.0. On a card treat this as a broken run -- note riscvbench still EXITS 0, the check is advisory"
    return
  fi
  if ! grep -q '^TTRVBENCH_VALID' "$out"; then
    echo "UNCLEAR|no TTRVBENCH_VALID lines in $label.out; the run did not reach its own gate"
    return
  fi
  bad="$(_rv_failed_phases "$out")"
  if [ -z "$bad" ]; then
    echo "MEANINGFUL|every phase passed its validity gate"
    return
  fi
  if [ "$bad" = Q ]; then
    qn="$(_rv_q_worst_n "$out")"
    if [ "${qn:-0}" -gt 0 ] && [ "${qn:-0}" -le 16 ]; then
      echo "MEANINGFUL|every phase passed except Q, whose complaints are all at n <= $qn. riscvbench/README.md documents phase Q's cascade failing the zero-tolerance monotonicity gate at small n on silicon -- one cold, once-only burst scatters by 6-23 cycles -- so this is expected here and the phase-Q rows are still readable"
      return
    fi
    echo "SUSPECT|phase Q failed its gate at n = ${qn:-?}, above the small-burst scatter the README accounts for. Read the NOT MONOTONE lines"
    return
  fi
  echo "SUSPECT|phase(s) $bad failed their gates; read the TTRVBENCH_VALID_<L> lines in $label.out"
}

# The cross-check is a REPEATABILITY control, so it must be the same experiment
# twice. Comparing a --blocks 32 run against a --blocks 8 one is not a
# cross-check: riscvbench/README.md puts the minimum usable block count "above 8
# and at or below 32", so the short run is expected to fail its gates and did --
# R, C, Q, F and G on 2026-08-09, against the primary's phase Q alone.
rv_cross_verdict() { # cross-out, primary-csv, cross-csv
  local out="$1" a="$2" b="$3" base worst
  base="$(rv_verdict "$out" rv-cross)"
  case "$base" in MEANINGFUL\|*) ;; *) echo "$base"; return ;; esac
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then
    echo "COLLECTED|the cross-check run passed its own gates, but one of the two CSVs is missing so repeatability was not compared"
    return
  fi
  # Same schema, same plan, same block count: join on every key column and take
  # the largest relative disagreement in `cycles`.
  worst="$(awk -F, '
      /^#/ { next }
      $10 + 0 != $10 { next }                     # the header row
      NR == FNR { k = $1 "/" $2 "/" $3 "/" $6 "/" $7 "/" $8; a[k] = $10 + 0; next }
      { k = $1 "/" $2 "/" $3 "/" $6 "/" $7 "/" $8
        if (!(k in a)) { missing++; next }
        n++
        d = a[k] - ($10 + 0); if (d < 0) d = -d
        base = a[k]; if (base <= 0) base = 1
        r = d / base; if (r > w) { w = r; wk = k } }
      END { printf "%.4f %d %d %s", w + 0, n + 0, missing + 0, wk }' "$a" "$b")"
  # shellcheck disable=SC2086
  set -- $worst
  if [ "${2:-0}" = 0 ]; then
    echo "COLLECTED|the cross-check passed its own gates but shares no comparable rows with the primary; repeatability was not established"
    return
  fi
  echo "MEANINGFUL|repeatability control: $2 rows compared against the primary at the SAME parameters, largest disagreement $(awk -v x="$1" 'BEGIN { printf "%.1f%%", x * 100 }') (${4:-?}), ${3:-0} rows unmatched"
}

# ---------------------------------------------------------------- rv-gset
#
# COLLECTED is the honest status -- the cliff's shape is an analysis-box
# question -- but "the file exists" is not evidence that anything was measured.
# The two gsets compile DIFFERENT intermediate footprints, so if their rows are
# identical the build did not vary and neither will the cliff.
rv_gset_verdict() { # csv1, csv2, out-file
  local a="$1" b="$2" out="$3"
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then
    echo "FAILED|one or both gset runs produced no CSV"
    return
  fi
  if [ -s "$out" ] && grep -q '^TTRVBENCH_VALID_G: no' "$out"; then
    echo "SUSPECT|a gset run failed phase G's validity gate; read rv-gset.out"
    return
  fi
  if cmp -s "$a" "$b"; then
    echo "DEGENERATE|gsets 1 and 2 produced byte-identical CSVs, so the two builds compiled the same footprint and the fetch cliff has nothing to be read off"
    return
  fi
  echo "COLLECTED|gsets 1 and 2 written (6144 and 7168 B bodies) and their rows differ; with gset 0's 5120 B from the rv probe they bracket the fetch cliff. The cliff's shape is read at the analysis box. 4608/5632 B are NOT built -- see the README"
}

# --------------------------------------------------------------- rv-qdrain
#
# Also COLLECTED, and also not on the strength of the file existing: a burst
# sweep whose cycle column never changes swept nothing.
rv_qdrain_verdict() { # csv
  local csv="$1" distinct
  [ -s "$csv" ] || { echo "FAILED|no CSV; send rv-qdrain.out"; return; }
  # `n++` over a seen-set rather than `length(c)`: length() on an array is a
  # gawk extension and a card box running mawk must reach the same verdict.
  distinct="$(awk -F, '
      /^#/ { next }
      $10 + 0 != $10 { next }
      !($10 in c) { c[$10] = 1; n++ }
      END { print n + 0 }' "$csv")"
  if [ "${distinct:-0}" -le 1 ]; then
    echo "DEGENERATE|every row of the burst sweep reports the same cycle count, so the sweep did not sweep"
    return
  fi
  echo "COLLECTED|queue-depth sweep written, $distinct distinct cycle counts; it is the knee hunt, so whether it says anything is an analysis-box question. Its longest burst is n=1024 -- if the depth is still growing there the sweep is a LOWER BOUND and needs the compile-time extension the README describes"
}

# -------------------------------------------------------------- rv-pairs
rv_pairs_verdict() { # out-file, arch
  local out="$1" arch="$2" base
  base="$(rv_verdict "$out" rv-pairs)"
  case "$base" in MEANINGFUL\|*) ;; *) echo "$base"; return ;; esac
  if [ "$arch" = wormhole ]; then
    echo "MEANINGFUL|FIRST EVER on Wormhole, every phase gated. Prediction: the store-coalescing pair is IDENTICAL here (5.2x apart on Blackhole). Either outcome settles the cross-arch question"
  else
    echo "MEANINGFUL|second Blackhole sample of the coalescing/multiply/divide probes, every phase gated. The cross-arch comparison these feed waits on the Wormhole follow-on"
  fi
}

# ----------------------------------------------------------- noc / noc-epoch
#
# Both of these read a report that a Python module writes. On 2026-08-09 that
# module was not importable on the card box, so both `.report.txt` files
# contained one line of ModuleNotFoundError -- and because both checks tested
# for the ABSENCE of a string ("INVALID", "clock-epoch skew"), both passed. The
# noc-epoch one then asserted, from that absence, that the (11,2) clock epoch
# was retired. It was not: rerun at the analysis box, the detector found (11,2)
# holding a +323438586-cycle epoch reproduced over five runs. A verdict must
# never conclude anything from a string being missing from a file it has not
# first established is a report.
noc_verdict() { # report-file, have_tt_sim
  local rep="$1" have="$2"
  if [ "$have" != 1 ]; then
    echo "DEFERRED|the CSV is collected and complete; the report generator needs tt_sim/, which is not on this box. Run \`python3 -m tt_sim.perf.noc_congestion_sweep --measured <csv>\` at the analysis box -- until then this probe has NO verdict"
    return
  fi
  if _report_is_broken "$rep"; then
    echo "FAILED|the report generator did not produce a report (missing, empty, or a Python traceback). Nothing can be concluded from its contents"
    return
  fi
  if ! grep -q '  RESULT:' "$rep"; then
    echo "FAILED|the report has no RESULT: line, so it did not reach a verdict"
    return
  fi
  if grep -q 'RESULT: INVALID' "$rep"; then
    echo "DEGENERATE|$(grep -m1 'RESULT: INVALID' "$rep" | sed 's/^ *//'). EXPECTED against tt-sim, which models no link congestion; on a card it means the experiment was forced flat"
  elif grep -q 'RESULT: CONGESTION MEASURED' "$rep"; then
    echo "MEANINGFUL|$(grep -m1 'RESULT: CONGESTION MEASURED' "$rep" | sed 's/^ *//') -- shared-link sweep at 64/512/2048/8192/16384 B; read the report"
  elif grep -q 'RESULT: NO CONGESTION EFFECT' "$rep"; then
    echo "MEANINGFUL|$(grep -m1 'RESULT: NO CONGESTION EFFECT' "$rep" | sed 's/^ *//') -- the controls DID move and the flows DID overlap, so this is a negative result rather than a flat one"
  else
    echo "UNCLEAR|$(grep -m1 '  RESULT:' "$rep" | sed 's/^ *//')"
  fi
}

noc_epoch_verdict() { # report-file, have_tt_sim
  local rep="$1" have="$2"
  if [ "$have" != 1 ]; then
    echo "DEFERRED|both congestion CSVs are collected; the clock-epoch detector needs tt_sim/, which is not on this box. Pool them at the analysis box with \`python3 -m tt_sim.perf.noc_congestion_sweep --measured noc.<arch>.csv noc-epoch.<arch>.csv\`. The (11,2) epoch is NEITHER confirmed NOR retired until that runs"
    return
  fi
  if _report_is_broken "$rep"; then
    echo "FAILED|the clock-epoch detector did not produce a report (missing, empty, or a Python traceback). It has said nothing about any per-core epoch, in either direction"
    return
  fi
  if ! grep -q '  RESULT:' "$rep"; then
    echo "FAILED|the report has no RESULT: line, so the detector did not run to completion"
    return
  fi
  if grep -q 'clock-epoch skew' "$rep"; then
    echo "MEANINGFUL|$(grep -m1 'clock-epoch skew' "$rep" | sed 's/^ *//')"
  else
    echo "COLLECTED|two runs pooled and the detector named no reproducing per-core epoch in THIS pair. That is not a retirement: the detector only ever accepts an offset that reproduces, so its silence over one pair is weaker evidence than the five-run reproduction already on record. Pool this pair with the banked runs at the analysis box"
  fi
}
