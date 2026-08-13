#!/usr/bin/env bash
# Shared simulator-process bookkeeping for the test tooling (optests/diff.sh,
# driver/*/tests/*.sh). Source it as a library, or run it as a command -- see
# COMMAND LINE at the end of this header.
#
# WHY THIS EXISTS
# ---------------
# These scripts do not start the simulator server themselves: they run a
# tt-metal host binary, and UMD spawns <TT_METAL_SIMULATOR>/run.sh from inside
# it with UV_PROCESS_DETACHED (umd device/simulation/rtl_sim_communicator.cpp),
# i.e. in its own session. The server is therefore neither a child of the
# script nor in its process group, so it cannot be cleaned up by pid or killpg.
# The scripts used to solve that with
#
#     pkill -9 -f 'driver\.(wormhole|blackhole)\.server( |$)'
#
# which kills *every* simulator on the machine — including one another's, and
# anything a second terminal or a concurrent agent is running. That silently
# corrupts results (a run whose server vanished mid-flight looks like a
# legitimate, wrong measurement) rather than failing loudly.
#
# What the script *does* control is the server's command line: run.sh forwards
# $TT_SIM_RUN_TAG as `--run-tag <tag>`, so the tag lands in the server's
# /proc/<pid>/cmdline. Cleanup then targets only the servers this invocation
# caused to be started.
#
# The tag is `ttsim-run.<label>.<owner-pid>.<owner-starttime>`, where the owner
# is the script process and the start time is field 22 of /proc/<pid>/stat.
# Carrying the start time makes "is the owner still running?" exact rather than
# a pid-reuse guess, which is what lets a *later* run tell an orphan left by a
# crashed or timed-out earlier run (owner gone) from a server belonging to a
# run happening right now (owner alive) — the orphan recovery the broad pkill
# provided, and which must not be lost.
#
# Matching walks /proc rather than using `pkill -f`, and never considers this
# script or its ancestors. `pkill -f` matches any process whose command line
# merely *mentions* the pattern, so a shell invoked with the pattern in its
# argv is a candidate — that is not hypothetical, it killed the calling shell
# while this file was being written. Every kill here additionally requires the
# process to be an actual `-m driver.<arch>.server` invocation.
#
# USAGE
#     . "$REPO/driver/sim_procs.sh"
#     sim_procs_init diff          # label, no dots; conventionally the script name
#     trap 'sim_kill_own_servers' EXIT INT TERM
#     ...
#     sim_kill_own_servers         # between runs, to clear a timed-out leftover
#
# ESCAPE HATCH
#     TT_SIM_KILL_ALL_SERVERS=1    restore the old sledgehammer: kill every
#                                  tt-sim server on the machine at startup.
#                                  Prefer `sim_procs.sh run` (below), which
#                                  makes a manual run reapable in the first
#                                  place, over reaching for this.
#
# COMMAND LINE
# ------------
# The library above only helps a script that sources it. A human at a prompt --
# or another team running tt-sim their own way -- had nothing but the
# sledgehammer, which kills concurrent runs and turns a vanished server into a
# plausible wrong measurement rather than a loud failure. That is not
# hypothetical: it is how an hours-long run was lost on 2026-08-13, and how the
# compiler team's report came to say orphans "must be reaped by PID".
#
#     driver/sim_procs.sh list          what is running, and whose
#     driver/sim_procs.sh reap          kill ONLY orphans (owner gone)
#     driver/sim_procs.sh run <label> -- <cmd>...
#                                       tag a manual run so its servers are
#                                       reapable, and clean them up on exit
#     driver/sim_procs.sh kill <pid>... kill named servers, after checking each
#                                       really is a tt-sim server
#
# `run` is the one that closes the gap. Anything started under it carries a tag
# and is cleaned up when it ends, however it ends -- so a manual run stops
# being the thing only a sledgehammer can clear. `list` and `reap` never touch
# a server whose owning run is still alive, so they are safe to use while
# somebody else is working on the same machine.

SIM_RUN_TAG=""

_sim_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

# Field $2 of /proc/$1/stat, counting from the state field (i.e. the real field
# number minus 2). Splitting after the ") " that ends comm survives a command
# name containing spaces or parens. Empty if the process is gone.
_sim_stat_field() {
  local stat rest
  stat="$(cat "/proc/$1/stat" 2>/dev/null)" || return 0
  rest="${stat#*) }"
  printf '%s' "$rest" | awk -v i="$2" '{print $i}'
}

_sim_starttime() { _sim_stat_field "$1" 20; }

# This shell and every process above it, so cleanup can never target itself,
# the calling shell, or the terminal it was launched from.
_sim_ancestry() {
  local pid=$$ hops=0
  while [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null && [ "$hops" -lt 64 ]; do
    printf '%s ' "$pid"
    pid="$(_sim_stat_field "$pid" 2)"
    hops=$((hops + 1))
  done
}

# pids of real tt-sim server processes, one per line. $1, if given, is a
# substring the command line must also contain (used to select by run tag).
_sim_servers() {
  local want="${1:-}" dir pid cmd skip
  skip=" $(_sim_ancestry)"
  for dir in /proc/[0-9]*; do
    pid="${dir#/proc/}"
    case "$skip" in *" $pid "*) continue ;; esac
    # 2>/dev/null must precede the input redirection: a process that exits
    # mid-scan makes the *redirection* fail, and that message is the shell's.
    cmd="$(tr '\0' ' ' 2>/dev/null <"$dir/cmdline") " || continue
    # A server is a python running the server module, nothing else: a *shell*
    # whose argv merely quotes that module name is not a server (this scan runs
    # from scripts that are frequently launched with exactly that text in argv).
    # The trailing space anchors the module name, so
    # `-m driver.<arch>.server.<name>_replay_test` — an offline replay guard —
    # is not mistaken for a live server.
    case "${cmd%% *}" in */python* | python*) ;; *) continue ;; esac
    case "$cmd" in
      *"-m driver.wormhole.server "* | *"-m driver.blackhole.server "*) ;;
      *) continue ;;
    esac
    if [ -n "$want" ]; then
      case "$cmd" in *"$want"*) ;; *) continue ;; esac
    fi
    echo "$pid"
  done
}

# A tagged server is an orphan iff the run that started it is gone: its owner
# pid is either dead or has been reused by an unrelated process (start times
# differ). Anything that cannot be parsed is left alone — killing a stranger's
# server is far worse than leaving an orphan.
_sim_reap_orphans() {
  local pid cmd tag rest owner start n=0
  for pid in $(_sim_servers "run-tag ttsim-run."); do
    cmd="$(tr '\0' ' ' 2>/dev/null <"/proc/$pid/cmdline")" || continue
    tag="${cmd##*run-tag }"
    tag="${tag%% *}"
    rest="${tag#ttsim-run.}" # <label>.<pid>.<starttime>
    rest="${rest#*.}"        # <pid>.<starttime>
    owner="${rest%%.*}"
    start="${rest#*.}"
    case "$owner$start" in *[!0-9]* | "") continue ;; esac
    [ "$(_sim_starttime "$owner")" = "$start" ] && continue
    kill -9 "$pid" 2>/dev/null && n=$((n + 1))
  done
  [ "$n" -gt 0 ] && echo "[sim] reaped $n orphaned simulator server(s) from an earlier run" >&2
  return 0
}

_sim_kill_all() {
  local pid n=0
  for pid in $(_sim_servers); do
    kill -9 "$pid" 2>/dev/null && n=$((n + 1))
  done
  echo "[sim] TT_SIM_KILL_ALL_SERVERS: killed $n tt-sim server(s) on this machine" >&2
  return 0
}

# Call once, before running anything that can spawn a server.
sim_procs_init() {
  local label="${1:-run}"
  SIM_RUN_TAG="ttsim-run.${label}.$$.$(_sim_starttime $$)"
  export TT_SIM_RUN_TAG="$SIM_RUN_TAG"
  if _sim_truthy "${TT_SIM_KILL_ALL_SERVERS:-}"; then
    _sim_kill_all
    sleep 0.3
  else
    _sim_reap_orphans
  fi
  return 0
}

# Kill only the servers started under this invocation's tag.
sim_kill_own_servers() {
  local pid
  [ -n "$SIM_RUN_TAG" ] || return 0
  for pid in $(_sim_servers "run-tag $SIM_RUN_TAG"); do
    kill -9 "$pid" 2>/dev/null
  done
  return 0
}

# ---------------------------------------------------------------------------
# Command line
#
# Everything above is the library, and sourcing this file must define functions
# and do nothing else. The CLI below runs only when the file is executed.
# ---------------------------------------------------------------------------

# `ttsim-run.<label>.<owner>.<starttime>` -> "<owner> <starttime>", empty if the
# command line carries no parseable tag.
_sim_tag_owner() {
  local cmd="$1" tag rest owner start
  case "$cmd" in *"run-tag ttsim-run."*) ;; *) return 0 ;; esac
  tag="${cmd##*run-tag }"
  tag="${tag%% *}"
  rest="${tag#ttsim-run.}"
  rest="${rest#*.}"
  owner="${rest%%.*}"
  start="${rest#*.}"
  case "$owner$start" in *[!0-9]* | "") return 0 ;; esac
  printf '%s %s' "$owner" "$start"
}

# "orphan" (tagged, owner gone), "live" (tagged, owner running) or "untagged".
# An unparseable tag reads as "live": never propose killing what we cannot
# account for.
_sim_state() {
  local owner start pair
  pair="$(_sim_tag_owner "$1")"
  [ -n "$pair" ] || {
    case "$1" in *"run-tag "*) echo live ;; *) echo untagged ;; esac
    return 0
  }
  owner="${pair%% *}"
  start="${pair##* }"
  if [ "$(_sim_starttime "$owner")" = "$start" ]; then echo live; else echo orphan; fi
}

_sim_cmd_of() { tr '\0' ' ' 2>/dev/null <"/proc/$1/cmdline"; }

_sim_cli_list() {
  local pid cmd state arch age tag n=0
  printf '%-8s %-10s %-9s %-9s %s\n' PID ARCH AGE STATE TAG
  for pid in $(_sim_servers); do
    cmd="$(_sim_cmd_of "$pid")" || continue
    [ -n "$cmd" ] || continue
    case "$cmd" in *wormhole*) arch=wormhole ;; *blackhole*) arch=blackhole ;; *) arch='?' ;; esac
    state="$(_sim_state "$cmd")"
    age="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
    case "$cmd" in
      *"run-tag "*) tag="${cmd##*run-tag }"; tag="${tag%% *}" ;;
      *) tag='(none -- started by hand)' ;;
    esac
    printf '%-8s %-10s %-9s %-9s %s\n' "$pid" "$arch" "${age:-?}" "$state" "$tag"
    n=$((n + 1))
  done
  [ "$n" = 0 ] && echo "(no tt-sim servers running)"
  echo "" >&2
  echo "live    = its run is still going. Do not kill it; you will corrupt that run." >&2
  echo "orphan  = its run is gone. 'reap' clears these." >&2
  echo "untagged= started by hand, outside 'run'. Only 'kill <pid>' clears these." >&2
  return 0
}

_sim_cli_reap() {
  _sim_reap_orphans
  return 0
}

_sim_cli_kill() {
  local want pid found rc=0
  [ "$#" -gt 0 ] || { echo "usage: sim_procs.sh kill <pid>..." >&2; return 2; }
  for want in "$@"; do
    found=0
    for pid in $(_sim_servers); do
      [ "$pid" = "$want" ] && found=1 && break
    done
    if [ "$found" = 0 ]; then
      echo "[sim] refusing: pid $want is not a tt-sim server (or is gone)" >&2
      rc=1
      continue
    fi
    if [ "$(_sim_state "$(_sim_cmd_of "$want")")" = live ]; then
      echo "[sim] pid $want belongs to a RUN THAT IS STILL ALIVE -- killing it anyway, as asked" >&2
    fi
    kill -9 "$want" 2>/dev/null && echo "[sim] killed $want" >&2
  done
  return "$rc"
}

_sim_cli_run() {
  local label="${1:-}" rc
  shift 2>/dev/null || true
  [ -n "$label" ] || { echo "usage: sim_procs.sh run <label> -- <cmd>..." >&2; return 2; }
  [ "${1:-}" = "--" ] && shift
  [ "$#" -gt 0 ] || { echo "usage: sim_procs.sh run <label> -- <cmd>..." >&2; return 2; }
  case "$label" in *.*) echo "[sim] label must not contain dots (it delimits the tag)" >&2; return 2 ;; esac
  sim_procs_init "$label"
  # However the command ends -- success, failure, Ctrl-C, SIGTERM -- the servers
  # it caused to be started go with it. That is the whole point of the wrapper.
  trap 'sim_kill_own_servers' EXIT INT TERM
  echo "[sim] run tag $SIM_RUN_TAG" >&2
  "$@"
  rc=$?
  sim_kill_own_servers
  trap - EXIT INT TERM
  return "$rc"
}

_sim_cli_usage() {
  sed -n '/^# COMMAND LINE$/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  return 2
}

# Sourced: define and stop. Executed: dispatch.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  _sim_sub="${1:-}"
  shift 2>/dev/null || true
  case "$_sim_sub" in
    list) _sim_cli_list "$@" ;;
    reap | reap-orphans) _sim_cli_reap "$@" ;;
    run) _sim_cli_run "$@" ;;
    kill) _sim_cli_kill "$@" ;;
    "" | -h | --help | help) _sim_cli_usage ;;
    *) echo "unknown subcommand '$_sim_sub'" >&2; _sim_cli_usage ;;
  esac
  exit $?
fi
