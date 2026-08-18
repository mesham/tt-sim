#!/usr/bin/env bash
# Regenerate the Python flatbuffer bindings under server/_flatbuf/ from the
# tt-metal schema. flatc must be installed.
set -euo pipefail

# Defaults to the schema inside whichever tt-metal $TT_METAL_HOME points at;
# pass a path as $1 to override. Hard-coding a checkout here only ever worked
# on one machine.
SCHEMA="${1:-${TT_METAL_HOME:?set TT_METAL_HOME, or pass the .fbs path as the first argument}/tt_metal/third_party/umd/device/simulation/tt_simulation_device.fbs}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/_flatbuf"

if ! command -v flatc >/dev/null 2>&1; then
    echo "error: flatc not on PATH" >&2
    exit 1
fi
if [ ! -f "$SCHEMA" ]; then
    echo "error: schema not found: $SCHEMA" >&2
    exit 1
fi

mkdir -p "$OUT"
flatc --python -o "$OUT" "$SCHEMA"

# Patch DeviceRequestResponse.py so its inline import of tt_vcs_core works
# inside this package (the generator emits an absolute import).
sed -i 's|^\([[:space:]]*\)from tt_vcs_core import tt_vcs_core$|\1from .tt_vcs_core import tt_vcs_core|' \
    "$OUT/DeviceRequestResponse.py"

echo "regenerated bindings under $OUT"
