#!/usr/bin/env bash
# Guards for driver/sim_procs.sh -- both the library it has always been and the
# command line added for manual runs.
#
# No simulator is started. A tt-sim server is, for the matcher's purposes, a
# python process whose argv contains `-m driver.<arch>.server `, so a `sleep`
# launched under exactly that argv is indistinguishable from the real thing --
# which is the point: the matcher must be tested against what it actually reads
# rather than against a real server, whose lifecycle would make the timing
# assertions flaky.
#
# The properties that matter are the SAFETY ones, and they are asserted in both
# directions: a live run's server must survive `reap`, an orphan must not, and
# nothing that is not a server may ever be killed.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/sim_procs.sh"
FAILED=0
PIDS=""

ok() { printf '  ok   %s\n' "$1"; }
bad() {
  printf '  FAIL %s\n' "$1"
  FAILED=1
}
check() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

cleanup() {
  local p
  for p in $PIDS; do kill -9 "$p" 2>/dev/null; done
}
trap cleanup EXIT INT TERM

# A stand-in server: argv shaped exactly like the real one, including the
# trailing space the matcher anchors on. $1 is the tag, or empty for untagged.
#
# Sets FAKE_PID rather than echoing it, and both streams go to /dev/null.
# Neither is tidiness; both are the same trap, met twice while writing this.
# `x="$(fake_server ...)"` would run the function in a SUBSHELL, so (a) the
# substitution would not return until the backgrounded sleeper closed its end
# of the pipe, 600 seconds later, and (b) the append to PIDS would happen in
# the subshell and be lost, leaving the parent's cleanup with nothing to kill.
# A test for process hygiene that leaks processes is worth less than no test.
FAKE_PID=""
fake_server() {
  local tag="${1:-}"
  if [ -n "$tag" ]; then
    python3 -c 'import time; time.sleep(600)' -m "driver.blackhole.server" --run-tag "$tag" >/dev/null 2>&1 &
  else
    python3 -c 'import time; time.sleep(600)' -m "driver.blackhole.server" >/dev/null 2>&1 &
  fi
  FAKE_PID="$!"
  PIDS="$PIDS $FAKE_PID"
}

# /proc rather than `kill -0`: signalling a process this user does not own
# fails with EPERM, so `kill -0 1` reports init as dead and the "we did not
# kill pid 1" assertion passes for the wrong reason -- or, as it did here,
# fails for one.
alive() { [ -d "/proc/$1" ] && echo yes || echo no; }

echo "sim_procs_test: library"

# Sourcing must define functions and do nothing else -- every consumer in the
# tree relies on that, and the CLI must not have changed it.
out="$(bash -c ". '$SCRIPT'; sim_procs_init srctest; echo \"TAG=\$TT_SIM_RUN_TAG\"")"
case "$out" in
  TAG=ttsim-run.srctest.*) ok "sourcing defines the library and sets a tag" ;;
  *) bad "sourcing produced '$out'" ;;
esac
case "$out" in
  *PID*ARCH*) bad "sourcing ran the CLI" ;;
  *) ok "sourcing does not run the CLI" ;;
esac

echo "sim_procs_test: list"

fake_server "ttsim-run.demo.$$.$(awk '{print $22}' "/proc/$$/stat")"
tagged_live="$FAKE_PID"
fake_server ""
untagged="$FAKE_PID"
sleep 0.2

listing="$("$SCRIPT" list 2>/dev/null)"
case "$listing" in *"$tagged_live"*) ok "list finds a tagged server" ;; *) bad "list missed $tagged_live" ;; esac
case "$listing" in *"$untagged"*) ok "list finds an untagged server" ;; *) bad "list missed $untagged" ;; esac
# The owner of `tagged_live` is this test process, which is alive, so it must
# read as `live` -- the state that `reap` is required to leave alone.
case "$(echo "$listing" | grep " $tagged_live \|^$tagged_live ")" in
  *live*) ok "a server whose owner is alive reads as live" ;;
  *) bad "tagged server did not read as live" ;;
esac
case "$(echo "$listing" | grep "^$untagged ")" in
  *untagged*) ok "a server with no tag reads as untagged" ;;
  *) bad "untagged server did not read as untagged" ;;
esac

echo "sim_procs_test: reap"

# An orphan: a syntactically valid tag naming an owner that cannot be running.
# pid 0 never names a live process, so its start time can never match.
fake_server "ttsim-run.gone.0.999999"
orphan="$FAKE_PID"
sleep 0.2

"$SCRIPT" reap >/dev/null 2>&1
sleep 0.2
check "reap kills an orphan" "$(alive "$orphan")" "no"
check "reap SPARES a live run's server" "$(alive "$tagged_live")" "yes"
check "reap leaves an untagged server alone" "$(alive "$untagged")" "yes"

echo "sim_procs_test: kill"

# Refusing a pid that is not a server is the guard that keeps this from
# becoming the sledgehammer it replaced.
"$SCRIPT" kill 1 >/dev/null 2>&1
check "kill refuses a non-server pid" "$?" "1"
check "  ...and pid 1 is obviously still alive" "$(alive 1)" "yes"

self_out="$("$SCRIPT" kill $$ 2>&1)"
check "kill refuses this test process" "$(alive $$)" "yes"
case "$self_out" in *"not a tt-sim server"*) ok "  ...and says why" ;; *) bad "  ...without explaining: $self_out" ;; esac

"$SCRIPT" kill "$untagged" >/dev/null 2>&1
sleep 0.2
check "kill removes a named server" "$(alive "$untagged")" "no"

echo "sim_procs_test: run"

# The wrapper's contract: the command sees a tag, and the exit status is the
# command's own rather than the wrapper's.
run_out="$("$SCRIPT" run demo -- sh -c 'echo "TAG=$TT_SIM_RUN_TAG"; exit 7' 2>/dev/null)"
rc=$?
check "run propagates the command's exit status" "$rc" "7"
case "$run_out" in
  TAG=ttsim-run.demo.*) ok "run exports a tag the command can see" ;;
  *) bad "run did not export a tag: $run_out" ;;
esac

"$SCRIPT" run 2>/dev/null >/dev/null
check "run without a command is a usage error" "$?" "2"
"$SCRIPT" run has.dots -- true 2>/dev/null >/dev/null
check "run rejects a dotted label (it delimits the tag)" "$?" "2"

echo ""
if [ "$FAILED" = 0 ]; then
  echo "sim_procs_test: PASS"
else
  echo "sim_procs_test: FAIL"
fi
exit "$FAILED"
