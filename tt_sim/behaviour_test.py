"""The registry is held to the code from both directions.

A published capability marker is only worth what its maintenance is worth. The
failure it exists to prevent — a consumer's suite running green against a
simulator that silently does nothing — is exactly the failure a stale registry
reintroduces, one level up: assert on ``noc1-multicast-corner-order``, get a
pass, and have the name mean nothing because the guard behind it was deleted
two refactors ago.

So:

* **Forward** — every entry names a test, and that test has to exist. An entry
  whose guarantee is no longer checked anywhere cannot survive here.
* **Backward** — every exception class in ``tt_sim/`` has to be either
  registered or explicitly declined in ``_NOT_A_GUARANTEE``. This is the guard
  that stops the scheme decaying, because the omission people actually make is
  landing a new loudness guard and never telling anyone outside about it. The
  scan is over the source tree rather than a list, so it covers modules nobody
  thought about when this was written.

Runs standalone (``python3 -m tt_sim.behaviour_test``) or under pytest.
"""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from tt_sim.behaviour import (
    _NOT_A_GUARANTEE,
    BEHAVIOURS,
    UnsupportedBehaviour,
    main,
    require,
    supports,
)

_ROOT = Path(__file__).resolve().parent


def _exception_classes():
    """``(class name, path)`` for every exception defined under ``tt_sim/``.

    Parsed rather than imported: importing the whole tree to enumerate its
    exceptions would make a registry check the slowest test in the suite, and
    would drag in numpy, pyarrow and the device model to answer a question that
    is answerable from the text. A class counts as an exception if its own name
    or any base's ends in ``Error`` or ``Exception``, which is the naming this
    repo (and the standard library) keeps to.
    """
    out = []
    for path in sorted(_ROOT.rglob("*.py")):
        if path.name.endswith("_test.py") or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            names = [node.name]
            for base in node.bases:
                if isinstance(base, ast.Name):
                    names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    names.append(base.attr)
            if any(n.endswith(("Error", "Exception")) for n in names):
                out.append((node.name, path.relative_to(_ROOT.parent)))
    return out


# --- forward: an entry cannot outlive the test that checks it ---------------


def test_every_behaviour_names_a_test_that_exists():
    assert BEHAVIOURS, "the registry is empty"
    for name, behaviour in BEHAVIOURS.items():
        module_name, _, function = behaviour.pinned_by.partition(":")
        assert function, f"{name}: pinned_by must be 'module:function'"
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function, None)), (
            f"{name} claims to be pinned by {behaviour.pinned_by}, which does "
            f"not exist. Either the test was renamed (repoint it) or the "
            f"guarantee stopped being checked (drop the entry)."
        )


def test_every_behaviour_is_completely_described():
    for name, behaviour in BEHAVIOURS.items():
        assert name == behaviour.name
        assert name == name.lower(), f"{name}: keep names lowercase"
        assert " " not in name, (
            f"{name}: names are typed into other people's test files; keep "
            f"them hyphenated so they survive being pasted anywhere"
        )
        assert len(behaviour.guarantee) > 60, (
            f"{name}: the guarantee has to say what a run will observably do"
        )
        assert len(behaviour.since) == 10, f"{name}: since must be YYYY-MM-DD"
        assert behaviour.since.count("-") == 2, f"{name}: since must be YYYY-MM-DD"


# --- backward: a new guard cannot land unmentioned ---------------------------


def test_every_exception_is_registered_or_explicitly_declined():
    """The guard against decay. A new exception class in ``tt_sim/`` turns this
    red until somebody decides, in writing, whether an outside consumer would
    want to assert on it."""
    # A behaviour accounts for an exception only by *naming* it in the
    # guarantee text, so the registry cannot quietly claim a class without
    # telling a reader which one it means.
    guarantees = [behaviour.guarantee for behaviour in BEHAVIOURS.values()]
    unaccounted = [
        (name, path)
        for name, path in _exception_classes()
        if name not in _NOT_A_GUARANTEE
        and not any(name in guarantee for guarantee in guarantees)
    ]
    assert not unaccounted, (
        "these exception classes are neither named by a registered behaviour "
        "nor declined in tt_sim/behaviour.py's _NOT_A_GUARANTEE: "
        + ", ".join(f"{name} ({path})" for name, path in unaccounted)
        + ". If an external suite would want to assert that this build raises "
        "it, register a behaviour; if not, add it to _NOT_A_GUARANTEE with the "
        "reason."
    )


def test_nothing_is_declined_that_no_longer_exists():
    """The exemption list is checked in the same direction as the registry, so
    it cannot accumulate names for classes that were deleted years ago."""
    defined = {name for name, _ in _exception_classes()}
    stale = sorted(set(_NOT_A_GUARANTEE) - defined)
    assert not stale, f"_NOT_A_GUARANTEE names classes that do not exist: {stale}"


def test_every_reason_given_is_a_reason():
    for name, reason in _NOT_A_GUARANTEE.items():
        assert len(reason) > 20, f"{name}: 'why not' needs an actual answer"


# --- the consumer-facing surface --------------------------------------------


def test_the_snippet_a_consumer_writes():
    """Verbatim from the module docstring and the docs section."""
    require("noc1-multicast-corner-order")
    assert supports("noc1-multicast-corner-order")


def test_an_absent_behaviour_fails_loudly_and_usefully():
    with pytest.raises(UnsupportedBehaviour) as caught:
        require("noc1-multicast-corner-order", "no-such-guarantee")
    message = str(caught.value)
    assert "no-such-guarantee" in message
    # Not merely "unsupported": the reader is told what this build does have,
    # because a typo and an old checkout look identical otherwise.
    assert "noc1-multicast-corner-order" in message
    assert "docs/running-tt-metal-on-the-simulator.md" in message
    assert not supports("no-such-guarantee")
    assert not supports("noc1-multicast-corner-order", "no-such-guarantee")


def test_the_registry_cannot_be_talked_into_a_guarantee():
    with pytest.raises(TypeError):
        BEHAVIOURS["invented"] = None


def test_the_cli_lists_and_checks(capsys):
    assert main([]) == 0
    listed = capsys.readouterr().out.split()
    assert listed == sorted(BEHAVIOURS)

    assert main(["--require", *sorted(BEHAVIOURS)]) == 0
    assert main(["--require", "no-such-guarantee"]) == 1
    assert "no-such-guarantee" in capsys.readouterr().err


def test_the_cli_works_for_a_consumer_with_no_python_at_all():
    """A C++ harness or a Makefile shells out; that path must exit non-zero
    on a missing behaviour rather than printing something a script has to
    parse."""
    ok = subprocess.run(
        [sys.executable, "-m", "tt_sim.behaviour", "--require"] + sorted(BEHAVIOURS),
        capture_output=True,
        text=True,
        cwd=_ROOT.parent,
    )
    assert ok.returncode == 0, ok.stderr
    bad = subprocess.run(
        [sys.executable, "-m", "tt_sim.behaviour", "--require", "no-such-guarantee"],
        capture_output=True,
        text=True,
        cwd=_ROOT.parent,
    )
    assert bad.returncode == 1
    assert "no-such-guarantee" in bad.stderr


def main_standalone():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if fn.__code__.co_argcount:
                continue  # needs a pytest fixture
            fn()
    print("behaviour tests OK")


if __name__ == "__main__":
    main_standalone()
