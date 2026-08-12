"""tt-metal's compute grid, and the order it hands workers to a program.

``TT_SIM_TENSIX_CORES=N`` is only worth having if the N workers it
materialises are the N workers tt-metal will actually launch kernels on. That
order is tt-metal's, not ours. ``split_work_to_cores`` →
``num_cores_to_corerangeset`` (tt-metal ``tt_metal/common/work_split.cpp``)
fills the **compute-with-storage grid column-major**: logical ``(0,0)``,
``(0,1)`` … ``(0, H-1)``, then ``(1,0)`` … — where ``H`` is the height of *that*
grid, not of the SoC descriptor's worker grid and not of any block tt-sim
prefers. So the fill order moves with
``TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE``, and a count-based knob that
ignores the variable is wrong by construction as soon as ``N`` exceeds one
column: with the 4×5 override the 6th core is physical ``2-1``, with no
override it is ``1-7``.

Logical → physical is index-into-the-sorted-axis, not ``+1``: the SoC
descriptor's worker rows and columns have gaps (Wormhole skips physical row 6,
Blackhole skips columns 8–9), and UMD numbers logical coords by position in the
sorted list. Anything that adds one to each axis is right only inside the
gap-free corner and has already cost this project a wrong run.

The **default** grid — what tt-metal uses with no override — is a per-arch
constant supplied by the caller, because it comes from tt-metal's own
``tt_metal/core_descriptors/*_arch.yaml`` keyed by product name and dispatch
axis, which is not something the wire protocol carries. ``TT_SIM_COMPUTE_GRID``
overrides it for a tt-metal whose default has moved; the servers report the
grid they resolved at start-up so a mismatch is visible rather than silent.
"""

import os

#: Env var that pins the compute grid explicitly, as ``WxH`` (e.g. ``8x9``).
#: Escape hatch for a tt-metal release whose default grid differs from the
#: per-arch constant the driver ships.
COMPUTE_GRID_VAR = "TT_SIM_COMPUTE_GRID"

#: tt-metal's own grid override. Its value is the *inclusive maximum logical
#: x,y* of the compute-with-storage grid, so it names a grid one larger on each
#: axis (``3,4`` → 4×5).
METAL_GRID_OVERRIDE_VAR = "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"


def _parse_pair(raw, sep):
    x_s, found, y_s = raw.strip().strip("[]").partition(sep)
    if not found:
        raise ValueError(f"expected two values separated by {sep!r}, got {raw!r}")
    return int(x_s.strip()), int(y_s.strip())


def compute_grid(default_grid, env=None):
    """Return ``(width, height)`` of tt-metal's compute-with-storage grid.

    ``TT_SIM_COMPUTE_GRID=WxH`` wins outright; otherwise
    ``TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=x,y`` gives ``(x+1, y+1)``; with
    neither set the caller's ``default_grid`` stands.
    """
    if env is None:
        env = os.environ

    raw = env.get(COMPUTE_GRID_VAR)
    if raw and raw.strip():
        w, h = _parse_pair(raw, "x")
        if w < 1 or h < 1:
            raise ValueError(
                f"{COMPUTE_GRID_VAR}: grid must be at least 1x1, got {raw!r}"
            )
        return (w, h)

    raw = env.get(METAL_GRID_OVERRIDE_VAR)
    if raw and raw.strip():
        end_x, end_y = _parse_pair(raw, ",")
        if end_x < 0 or end_y < 0:
            raise ValueError(f"{METAL_GRID_OVERRIDE_VAR}: negative bound in {raw!r}")
        return (end_x + 1, end_y + 1)

    return tuple(default_grid)


def logical_to_physical(worker_coords):
    """Return ``(xs, ys)``: the sorted physical worker axes.

    Logical coord ``(i, j)`` is physical ``(xs[i], ys[j])`` — UMD numbers
    logical coords by position in these lists, which is why the mapping is an
    index rather than an offset.
    """
    xs = sorted({x for x, _ in worker_coords})
    ys = sorted({y for _, y in worker_coords})
    return xs, ys


def fill_order(worker_coords, default_grid, env=None):
    """Physical worker coords in the order tt-metal assigns program cores.

    The compute grid comes first, column-major (tt-metal's ``row_wise=false``
    default, which is what ``split_work_to_cores`` passes). Workers outside
    that grid follow, so a count larger than the grid still resolves to
    something rather than raising — they are simply cores no ordinary program
    will launch on.
    """
    xs, ys = logical_to_physical(worker_coords)
    width, height = compute_grid(default_grid, env)
    width = min(width, len(xs))
    height = min(height, len(ys))
    ordered = [(xs[i], ys[j]) for i in range(width) for j in range(height)]
    inside = set(ordered)
    ordered.extend(coord for coord in sorted(worker_coords) if coord not in inside)
    return ordered
