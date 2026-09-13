"""The servers must import. Nothing else in the suite checks that.

UMD spawns ``driver/<arch>/server/__main__.py`` as a separate process. If that
module raises at import time the process dies *before* the socket exists, so
the tt-metal host blocks forever in ``recv`` with no error and no traceback —
indistinguishable from a simulator crash, and one of the two known causes of
"host hangs at startup" documented in the runbook's §3.1.

Every other test in this tree drives the library directly, so an entry point
that cannot be imported passes the whole suite and fails only in front of a
user. That is exactly what happened on 2026-08-24: a helper was added to
``tt_sim/bridge/device.py`` and imported by both servers from the package
façade, but ``tt_sim/bridge/__init__.py`` was not updated to re-export it. The
suite was green; both servers were dead. The compiler team found it while
running a real program, and reported it as a silent kill.

These tests are deliberately shallow — importing is the whole assertion.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "driver.wormhole.server.__main__",
        "driver.blackhole.server.__main__",
    ],
)
def test_the_server_entry_point_imports(module):
    """The check whose absence let a dead server ship.

    Importing is safe: ``main()`` sits behind an ``if __name__ == "__main__"``
    guard that does not fire under an ordinary import, so nothing binds a
    socket or builds a device here.
    """
    assert importlib.import_module(module) is not None


def test_the_bridge_facade_exports_everything_it_advertises():
    """``__all__`` and the re-export block must agree.

    The servers import from ``tt_sim.bridge`` rather than
    ``tt_sim.bridge.device``, so a name that reaches ``__all__`` without
    reaching the ``from ... import`` block above it is an ``ImportError`` in a
    subprocess nobody sees. Checking the façade against itself catches the
    half-edit whichever half was missed.
    """
    import tt_sim.bridge as bridge

    missing = [name for name in bridge.__all__ if not hasattr(bridge, name)]
    assert not missing, (
        f"tt_sim.bridge.__all__ advertises {missing}, which cannot be imported "
        "from it. Add them to the `from tt_sim.bridge.<module> import (...)` "
        "block; a server importing one of these dies before its socket exists "
        "and the tt-metal host hangs with no error."
    )
