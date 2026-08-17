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
# Hermetic: tag adoption reads the environment, so this file must not inherit a
# tag from a shell that happens to be inside a `sim_procs.sh run` itself.
unset TT_SIM_RUN_TAG
FAILED=0
PIDS=""

ASSERTIONS=0
ok() {
  ASSERTIONS=$((ASSERTIONS + 1))
  printf '  ok   %s\n' "$1"
}
bad() {
  ASSERTIONS=$((ASSERTIONS + 1))
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

echo "sim_procs_test: tag propagation"

# THE PATH THE TAG TAKES
# ----------------------
#   sim_procs.sh run <label>            mints and exports TT_SIM_RUN_TAG
#     -> a runner script                sources this file, calls sim_procs_init
#        -> the tt-metal host binary     inherits the environment
#           -> driver/<arch>/run.sh      turns the variable into `--run-tag`
#              -> the server             carries it in argv, where cleanup looks
#
# It used to break at the second hop: every runner calls sim_procs_init with its
# own label, and that MINTED unconditionally, so the wrapper's tag was gone one
# process later and its cleanup matched nothing. Both directions are asserted
# below -- adopted when the enclosing run is alive, minted when it is not.

# Hop 2, adoption: a runner's own label must not displace a live enclosing tag.
live_tag="ttsim-run.outer.$$.$(awk '{print $22}' "/proc/$$/stat")"
out="$(TT_SIM_RUN_TAG="$live_tag" bash -c ". '$SCRIPT'; sim_procs_init inner; echo \"TAG=\$TT_SIM_RUN_TAG\"" 2>/dev/null)"
check "sim_procs_init keeps a live enclosing tag" "$out" "TAG=$live_tag"

# ...and the other direction three times over: a tag whose owner is gone, one
# that does not parse, and none at all must each mint a fresh one. Adopting a
# dead tag would make every server it starts instantly reapable as an orphan.
out="$(TT_SIM_RUN_TAG="ttsim-run.gone.0.999999" bash -c ". '$SCRIPT'; sim_procs_init inner; echo \"TAG=\$TT_SIM_RUN_TAG\"" 2>/dev/null)"
case "$out" in
  TAG=ttsim-run.inner.*) ok "sim_procs_init mints over a DEAD enclosing tag" ;;
  *) bad "sim_procs_init adopted a dead tag: $out" ;;
esac
out="$(TT_SIM_RUN_TAG="not-a-tag" bash -c ". '$SCRIPT'; sim_procs_init inner; echo \"TAG=\$TT_SIM_RUN_TAG\"" 2>/dev/null)"
case "$out" in
  TAG=ttsim-run.inner.*) ok "sim_procs_init mints over an unparseable tag" ;;
  *) bad "sim_procs_init kept junk: $out" ;;
esac
out="$(bash -c "unset TT_SIM_RUN_TAG; . '$SCRIPT'; sim_procs_init inner; echo \"TAG=\$TT_SIM_RUN_TAG\"" 2>/dev/null)"
case "$out" in
  TAG=ttsim-run.inner.*) ok "sim_procs_init mints when nothing is inherited" ;;
  *) bad "sim_procs_init produced '$out' with no inherited tag" ;;
esac

# Hops 1+2 together, through the real CLI: what a runner script that calls
# sim_procs_init sees under `run` must still be the wrapper's tag.
runner="$(mktemp -d)"
cat >"$runner/runner.sh" <<RUNNER
. '$SCRIPT'
sim_procs_init runner_label
echo "SEEN=\$TT_SIM_RUN_TAG"
RUNNER
run_out="$("$SCRIPT" run wrapper -- bash "$runner/runner.sh" 2>/dev/null)"
case "$run_out" in
  SEEN=ttsim-run.wrapper.*) ok "run's tag survives a runner that calls sim_procs_init" ;;
  *) bad "the runner replaced the wrapper's tag: $run_out" ;;
esac

# Hop 4: driver/<arch>/run.sh is what puts the tag where cleanup can read it.
# A `python3` shim on PATH prints the argv the real server would have had, so
# this asserts the forwarding without starting a simulator.
shim="$(mktemp -d)"
cat >"$shim/python3" <<'SHIM'
#!/usr/bin/env bash
echo "ARGV: $*"
SHIM
chmod +x "$shim/python3"
for arch in wormhole blackhole; do
  argv="$(PATH="$shim:$PATH" NNG_SOCKET_ADDR=stub TT_SIM_RUN_TAG="ttsim-run.fwd.1.2" \
    bash "$HERE/$arch/run.sh" 2>/dev/null)"
  case "$argv" in
    *"--run-tag ttsim-run.fwd.1.2"*) ok "driver/$arch/run.sh forwards the tag into argv" ;;
    *) bad "driver/$arch/run.sh dropped the tag: $argv" ;;
  esac
  argv="$(PATH="$shim:$PATH" NNG_SOCKET_ADDR=stub bash -c "unset TT_SIM_RUN_TAG; exec bash '$HERE/$arch/run.sh'" 2>/dev/null)"
  case "$argv" in
    *--run-tag*) bad "driver/$arch/run.sh invented a tag: $argv" ;;
    *) ok "driver/$arch/run.sh adds no tag when there is none" ;;
  esac
done
rm -rf "$shim" "$runner"

echo "sim_procs_test: run cleans up after itself"

# The consequence of all of the above, at the process level. The wrapper starts
# a stand-in server carrying whatever tag it was given; when the wrapper exits,
# that server must be gone. A second stand-in, tagged to a different LIVE run
# (this test process), must survive -- `run` cleans up its own and nothing else.
fake_server "ttsim-run.bystander.$$.$(awk '{print $22}' "/proc/$$/stat")"
bystander="$FAKE_PID"
# Both streams of the backgrounded sleeper go to /dev/null inside the wrapper:
# holding this command substitution's pipe open would block it for 600 seconds,
# which is trap (a) from the fake_server comment above, met through the CLI.
owned="$("$SCRIPT" run wraptest -- bash -c \
  'python3 -c "import time; time.sleep(600)" -m driver.blackhole.server --run-tag "$TT_SIM_RUN_TAG" >/dev/null 2>&1 &
   echo $!
   sleep 1' 2>/dev/null)"
PIDS="$PIDS $owned"
sleep 0.3
check "run kills the server started under its own tag" "$(alive "$owned")" "no"
check "  ...and spares another live run's server" "$(alive "$bystander")" "yes"

echo ""
echo "sim_procs_test: $ASSERTIONS assertions"
if [ "$FAILED" = 0 ]; then
  echo "sim_procs_test: PASS"
else
  echo "sim_procs_test: FAIL"
fi
exit "$FAILED"
