"""Whether this run uses **NoC coordinate translation**, and how it found out.

Real Tenstorrent parts ship with NoC coordinate translation enabled: the NIU
recognises a second, *translated* coordinate range on top of the SoC-physical
one, and tt-metal addresses workers, ethernet and (on Blackhole) DRAM in that
range. tt-sim historically ran with it off, which is the configuration in which
a "physical" coordinate is NoC-dependent — and that is the root of the NoC 1
directory ambiguity documented in
``TT_Device._register_tile_internals``: on NoC 1 the same ``(x, y)`` names one
tile under the kernel's convention and a different one under the bank-to-noc
table's mirrored convention. With translation on, the two spaces are disjoint
and the ambiguity cannot arise: measured on the full grid, the 56 Wormhole and
102 Blackhole workers that resolve to a foreign tile on NoC 1 all resolve
correctly under translation.

**Who decides.** The decision is the *host's*, not ours: UMD reads a cluster
descriptor and the ``noc_translation`` flag in it settles which coordinates
tt-metal puts on the wire. tt-metal plumbs that descriptor for a Simulator
target through ``TT_METAL_MOCK_CLUSTER_DESC_PATH``
(``llrt/rtoptions.cpp``, ``llrt/tt_cluster.cpp``). So the safest possible
answer to "which convention are we in?" is to read *the same file the host
read*, and that is what :func:`translation_enabled` does by default.

That works because UMD spawns the simulator with ``uv_spawn`` and a NULL
``env`` (``umd/device/simulation/rtl_sim_communicator.cpp``), which libuv
documents as "the parent's environment is used" — so the server process
inherits ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` from the host program, exactly as
it already inherits ``NNG_SOCKET_ADDR``. Deriving the mode from the descriptor
rather than from a tt-sim-specific variable removes the whole class of failure
where the two ends disagree about the convention.

``TT_SIM_NOC_TRANSLATION`` overrides, for offline tests and for driving the
simulator without a host at all. It is an override, not the primary mechanism,
precisely so that nobody has to remember to keep two variables in step.

Inference is *not* attempted, deliberately: the first coordinate on the wire
does identify the convention, but only after the device is already built and
keyed. The server instead checks the first coordinates against the convention
it chose and fails loudly on a mismatch (see ``tt_sim.bridge.fabric``), so a
forgotten environment variable reads as an error and never as a plausible
result.
"""

import os

#: Explicit tt-sim override. Truthy values: ``1/true/yes/on``.
TRANSLATION_ENV = "TT_SIM_NOC_TRANSLATION"

#: The cluster descriptor tt-metal hands UMD; inherited from the host process.
CLUSTER_DESC_ENV = "TT_METAL_MOCK_CLUSTER_DESC_PATH"

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")


def _parse_bool(value):
    text = value.strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ValueError(
        f"{TRANSLATION_ENV}={value!r} is not a boolean "
        f"({'/'.join(_TRUTHY)} or {'/'.join(_FALSY)})"
    )


def descriptor_translation(path):
    """``noc_translation`` as declared by the cluster descriptor at ``path``.

    ``None`` when the file does not exist or declares no harvesting block —
    both of which mean "this descriptor tells us nothing", not "translation is
    off". Parsed with a targeted scan rather than a YAML load so that a
    descriptor UMD accepts but PyYAML chokes on (they are hand-written and
    quite loose) does not take the simulator down.
    """
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    found = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("noc_translation:"):
            continue
        value = stripped.split(":", 1)[1].strip().lower()
        # Every chip in a single-chip simulated cluster must agree; if a
        # descriptor ever disagreed, treat "any chip translated" as translated,
        # because tt-sim models one chip and would be addressed by that one.
        found = found or value in ("true", "1", "yes", "on")
    return found


def translation_source(env=None):
    """``(enabled, why)`` — the mode, and a phrase naming what decided it."""
    env = os.environ if env is None else env
    override = env.get(TRANSLATION_ENV)
    if override is not None and override.strip():
        return _parse_bool(override), f"{TRANSLATION_ENV}={override.strip()}"
    path = env.get(CLUSTER_DESC_ENV)
    if path:
        declared = descriptor_translation(path)
        if declared is None:
            return False, (
                f"{CLUSTER_DESC_ENV}={path} (unreadable or no noc_translation "
                f"key — assuming off)"
            )
        return declared, f"noc_translation: {str(declared).lower()} in {path}"
    return False, f"neither {TRANSLATION_ENV} nor {CLUSTER_DESC_ENV} is set"


def translation_enabled(env=None):
    """Whether this process should key its NoC directories by translated coords."""
    return translation_source(env)[0]
