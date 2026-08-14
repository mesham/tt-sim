"""Detection of NoC 1 destination-directory *shadowing*.

``TT_Device._register_tile_internals`` keys NoC 1's directory in two coordinate
conventions at once — each tile's canonical (SoC-physical NoC 0) coord, and its
``(GRID_X-1-x, GRID_Y-1-y)`` mirror, which is what tt-metal's bank-to-noc table
emits on NoC 1. The two spaces overlap, and mirror registrations are
authoritative, so a mirror can take a cell that a live Tensix worker owns
canonically. Every NoC 1 request addressed to that worker is then delivered to
— and **ACKed by** — the wrong tile: ``noc_async_write_barrier`` returns
normally, the worker never sees the data, and nothing on either side says so.

On Wormhole with all 80 functional workers built, 56 canonical worker coords
resolve to a foreign tile (48 worker-shadowed-by-worker, 8 by DRAM). Blackhole
registers no Tensix mirror aliases at all — nothing there addresses a worker by
its mirror (``ArchProfile.noc1_tensix_mirror_aliases``) — so with all 140 built
only 6 do, every one of them a DRAM mirror landing on a worker column. Which of
those a given program hits depends on which workers it materialises and which
coords it addresses over NoC 1, so the failure is configuration-dependent and
silent.

This module does not fix that — the two conventions genuinely collide, and no
lookup order is right for both. It makes it *visible*, at the moment the
directory becomes ambiguous, naming the exact coordinate and both tiles.

Policy comes from ``TT_SIM_NOC1_SHADOW``:

``warn`` (default)
    One line per shadowed coordinate on stderr, at registration. Warn rather
    than error because a shadowed cell is only *wrong* once something
    addresses it, and configurations that are correct today do carry one:
    measured, ``programming_examples/vecadd_multi_core`` on the 4x5 grid
    (``TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4``) passes its own value
    check while shadowing ``(4, 2)``, ``(4, 3)`` and ``(4, 4)``, because it
    never writes to those workers over NoC 1. Erroring by default would break
    it. Erroring at *resolve* time instead is no safer: the same key is the
    mirror by which the DRAM bank table legitimately reaches DRAM.
``error``
    Raise :class:`NoC1ShadowError` instead. For a run that must not silently
    misdeliver — CI, or bisecting a hang.
``off``
    Say nothing.
"""

import os
import sys

#: The environment variable selecting the policy.
POLICY_ENV = "TT_SIM_NOC1_SHADOW"

_POLICIES = ("warn", "error", "off")


class NoC1ShadowError(Exception):
    """A canonical NoC 1 coordinate owned by a live Tensix worker is taken."""


def shadow_policy(env=None):
    """Read :data:`POLICY_ENV`; ``warn`` when unset."""
    env = os.environ if env is None else env
    value = (env.get(POLICY_ENV) or "warn").strip().lower()
    if value not in _POLICIES:
        raise ValueError(f"{POLICY_ENV}={value!r} is not one of {'/'.join(_POLICIES)}")
    return value


def describe(entry):
    """``ClassName(x, y)`` for a tile, or for whatever else holds a cell.

    Takes a directory entry as well as a tile, because the thing occupying a
    contested cell is not always one: a NoC 1 request to a coord with no tile
    caches a ``NullEndpoint`` there, which then outlives the miss and shadows
    any worker later built at that coord.
    """
    if entry is None:
        return "nothing"
    nui = getattr(entry, "get_noc_nui", None)
    if nui is not None:
        return f"{type(entry).__name__}{nui(0).get_id_pair()}"
    return f"{type(entry).__name__}{getattr(entry, 'coord', '')}"


class ShadowReporter:
    """Per-device record of NoC 1 shadowing, and the warn/error policy.

    Deduplicates by coordinate: a cell contested by three registrations is one
    finding, not three. :attr:`shadowed` is left on the device so a driver
    script (or a test) can assert on the set rather than scrape stderr.
    """

    #: Printed once per reporter, before the first finding. The per-coord lines
    #: are one line each precisely because a full grid produces dozens of them.
    PREAMBLE = (
        "NoC 1's destination directory is keyed in two coordinate conventions "
        "(each tile's canonical coord, plus the grid mirror tt-metal's "
        "bank-to-noc table emits), and mirrors win contested cells. Each line "
        "below names a live Tensix worker that is therefore unreachable on "
        "NoC 1: requests to it are delivered to the impostor and ACKed by it, "
        f"so neither end can tell. Set {POLICY_ENV}=error to make this fatal, "
        "=off to silence."
    )

    def __init__(self, env=None):
        self.env = env
        #: ``coord -> message``, in discovery order. Left on the device so a
        #: driver or a test can assert on the set rather than scrape stderr.
        self.shadowed = {}

    def report(self, coord, message):
        """Record ``message`` against ``coord``; warn or raise per policy."""
        if coord in self.shadowed:
            return
        policy = shadow_policy(self.env)
        if policy == "off":
            return
        first = not self.shadowed
        self.shadowed[coord] = message
        if policy == "error":
            raise NoC1ShadowError(f"[NoC1-SHADOW] {message}")
        if first:
            print(f"[NoC1-SHADOW] {self.PREAMBLE}", file=sys.stderr)
        print(f"[NoC1-SHADOW] {message}", file=sys.stderr)

    def report_displaced(self, coord, victim, owner):
        """A live Tensix owns ``coord`` canonically; ``owner`` holds the cell."""
        self.report(
            coord,
            f"{coord} is the canonical coord of {describe(victim)}, but NoC 1 "
            f"resolves it to {describe(owner)}, which claimed the cell as its "
            f"mirror.",
        )

    def report_materialised_mirror(self, coord, mirror):
        """A NoC 1 directory miss was answered by building ``coord``'s mirror.

        Distinct from the registration-time case and strictly worse: the tile
        that ends up owning ``coord`` is one the program never asked for, and
        the worker that really lives at ``coord`` is shadowed from the moment
        it is eventually built.
        """
        self.report(
            coord,
            f"{coord} is a functional worker, but a NoC 1 directory miss there "
            f"was answered by materialising its mirror {mirror}; this request "
            f"lands on worker {mirror} and worker {coord} is shadowed from now "
            f"on.",
        )
