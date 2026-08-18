#!/usr/bin/env bash
# Which tt-metal a card benchmark's build tree was made against -- recorded when
# it is built, and refused when it no longer matches.
#
# THE FAILURE THIS EXISTS FOR, and it is not hypothetical. A Wormhole write
# session on 2026-08-17 died with rc=127 AFTER the board reset:
#
#   ./build/dramratebench: symbol lookup error: undefined symbol:
#     tt::tt_metal::CreateKernel(Program&, const string&, ...)
#
# TT_METAL_HOME had been pointed at a different checkout between sessions.
# `src/build/CMakeCache.txt` still held the old one's absolute paths, cmake
# reused the cache rather than reconfiguring, and the binary was compiled
# against one tt-metal's headers and run against another's library. The
# pre-flight said `ok build/dramratebench is current`, because it checked that
# a build EXISTED -- not what it was built against. The card time was spent, the
# board was reset for nobody, and the failure named a C++ symbol rather than the
# environment variable that caused it.
#
# Every runner under perfbench/ has that shape, so the check lives here once.
#
#   . "$HERE/../build_provenance.sh"
#   bp_check_build "$SRC" || fail=1        # BEFORE cmake; refuses on mismatch
#   ...build...
#   bp_record_build "$SRC"                 # AFTER a successful build
#
# `bp_require_build` is the same check for the runners that have no pre-flight
# accumulator to add to: it exits 1 itself.
#
# Standard tools only -- these scripts run on a card box with nothing but
# perfbench/ rsynced onto it.
#
# SPDX-License-Identifier: Apache-2.0

# The stamp sits INSIDE build/, so discarding the tree discards the provenance
# with it and the two can never disagree.
BP_STAMP_NAME=".tt-metal-provenance"

# The tt-metal the next cmake run would actually use. The CMakeLists under
# perfbench/*/src/ prefer TT_METAL_RUNTIME_ROOT and fall back to TT_METAL_HOME,
# so this must resolve them in that order or it would check the wrong thing.
bp_tt_metal_root() {
  local root="${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME:-}}"
  [ -n "$root" ] || return 1
  # Canonicalised, because `/x/tt-metal` and `/x/tt-metal/` are the same
  # checkout and a trailing slash must not read as a different one.
  (cd "$root" 2>/dev/null && pwd -P) || printf '%s\n' "${root%/}"
}

bp_tt_metal_rev() {
  local rev
  rev="$(git -C "${1:-.}" rev-parse --short=10 HEAD 2>/dev/null)" || rev=""
  printf '%s\n' "${rev:-unknown}"
}

# The tt-metal a build tree was made against, or "" if it cannot be told.
bp_recorded_root() { # <build-dir>
  local bd="$1" v=""
  if [ -f "$bd/$BP_STAMP_NAME" ]; then
    v="$(sed -n 's/^root=//p' "$bd/$BP_STAMP_NAME" | head -1)"
  fi
  if [ -z "$v" ] && [ -f "$bd/CMakeCache.txt" ]; then
    # A tree built before the stamp existed is still self-identifying: cmake
    # caches the config-package directory of every tt-metal dependency
    # TT::Metalium pulls in (umd, fmt, spdlog, tt-logger, nlohmann_json), and
    # each sits under <tt-metal>/build[_Release]/{lib,share}/cmake/. So the
    # legacy trees on a card box are checked too, rather than being waved
    # through until somebody happens to rebuild them.
    v="$(sed -nE 's;^[A-Za-z0-9_-]+_DIR:PATH=(.*)/build(_Release)?/(lib|share)/cmake/.*;\1;p' \
         "$bd/CMakeCache.txt" | head -1)"
  fi
  [ -n "$v" ] && v="$( (cd "$v" 2>/dev/null && pwd -P) || printf '%s' "${v%/}")"
  printf '%s\n' "$v"
}

bp_recorded_rev() { # <build-dir>
  local bd="$1" v=""
  [ -f "$bd/$BP_STAMP_NAME" ] && v="$(sed -n 's/^rev=//p' "$bd/$BP_STAMP_NAME" | head -1)"
  printf '%s\n' "${v:-unknown}"
}

# Refuse a build tree that was made against a different tt-metal. Prints in the
# runners' pre-flight style; returns 1 to refuse.
#
# Pass `skip-build` as the second argument when the caller will NOT rebuild. A
# revision change inside the same checkout is only survivable because the
# rebuild picks it up; with the build skipped it is the same stale-binary
# failure as a changed path, one release apart.
bp_check_build() { # <src-dir> [build|skip-build]
  local src="$1" mode="${2:-build}"
  local bd="$src/build" now_root now_rev had_root had_rev
  if ! now_root="$(bp_tt_metal_root)"; then
    echo "   FAIL neither TT_METAL_RUNTIME_ROOT nor TT_METAL_HOME is set, so there"
    echo "        is no tt-metal to check the build tree against"
    return 1
  fi
  now_rev="$(bp_tt_metal_rev "$now_root")"

  if [ ! -d "$bd" ]; then
    echo "   ok   no build tree yet; it will be made against $now_root (rev $now_rev)"
    return 0
  fi

  had_root="$(bp_recorded_root "$bd")"
  had_rev="$(bp_recorded_rev "$bd")"

  if [ -z "$had_root" ]; then
    echo "   WARN $bd names no tt-metal, in a stamp or in its CMakeCache, so what"
    echo "        it was built against cannot be told. If this run fails with a"
    echo "        symbol lookup error, that is why:  rm -rf $bd"
    return 0
  fi

  if [ "$had_root" != "$now_root" ]; then
    echo "   FAIL $bd was configured against a DIFFERENT tt-metal:"
    echo "          built against : $had_root (rev $had_rev)"
    echo "          now configured: $now_root (rev $now_rev)"
    echo "        cmake reuses its cache, so building here compiles against one"
    echo "        checkout's headers and links against the other's library. That"
    echo "        aborts with a symbol lookup error AFTER the board reset, which"
    echo "        is somebody else's card time as well as yours."
    echo "        A build tree is derived and always safe to discard:"
    echo "          rm -rf $bd"
    return 1
  fi

  if [ "$had_rev" != unknown ] && [ "$now_rev" != unknown ] && [ "$had_rev" != "$now_rev" ]; then
    if [ "$mode" = skip-build ]; then
      echo "   FAIL $bd was built at rev $had_rev; $had_root is now at rev $now_rev,"
      echo "        and --skip-build would run the old binary against the new"
      echo "        library. Re-run without --skip-build."
      return 1
    fi
    echo "   WARN $bd was built at rev $had_rev; $had_root is now at rev $now_rev."
    echo "        Same checkout, so the rebuild below picks the change up."
    return 0
  fi

  echo "   ok   build tree was made against $now_root (rev $now_rev)"
  return 0
}

# The same check for a runner with no pre-flight to accumulate into.
bp_require_build() { # <src-dir> [build|skip-build]
  bp_check_build "$@" || exit 1
}

# Stamp a tree that has just built cleanly. Called after cmake, never before:
# the stamp must describe a tree that exists, not one that was attempted.
bp_record_build() { # <src-dir>
  local bd="$1/build" root rev
  [ -d "$bd" ] || return 0
  root="$(bp_tt_metal_root)" || return 0
  rev="$(bp_tt_metal_rev "$root")"
  {
    echo "# The tt-metal this build tree was configured and compiled against."
    echo "# Written by perfbench/build_provenance.sh. Discard the TREE, not this"
    echo "# file: a stamp without its build tree is a claim about nothing."
    echo "root=$root"
    echo "rev=$rev"
    echo "stamped=$(date -Iseconds 2>/dev/null || date)"
  } >"$bd/$BP_STAMP_NAME"
}
