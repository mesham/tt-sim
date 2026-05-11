#!/usr/bin/env bash
# Regenerate the Python flatbuffer bindings under server/_flatbuf/ from the
# tt-metal schema. flatc must be installed.
set -euo pipefail

SCHEMA="${1:-/home/nick/projects/riscv/tt-metal/tt_metal/third_party/umd/device/simulation/tt_simulation_device.fbs}"
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
