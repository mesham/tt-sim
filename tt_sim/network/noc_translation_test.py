"""How the simulator decides which coordinate convention it is in."""

import pytest

from tt_sim.network.noc_translation import (
    CLUSTER_DESC_ENV,
    TRANSLATION_ENV,
    descriptor_translation,
    translation_enabled,
    translation_source,
)

TRANSLATED_DESCRIPTOR = """
arch:
  0: wormhole_b0
harvesting:
  0:
    noc_translation: true
    harvest_mask: 0
"""

PLAIN_DESCRIPTOR = TRANSLATED_DESCRIPTOR.replace("true", "false")


def _descriptor(tmp_path, text, name="cluster.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_off_when_nothing_is_set():
    enabled, why = translation_source({})
    assert enabled is False
    assert TRANSLATION_ENV in why
    assert CLUSTER_DESC_ENV in why


def test_reads_the_descriptor_the_host_read(tmp_path):
    path = _descriptor(tmp_path, TRANSLATED_DESCRIPTOR)
    enabled, why = translation_source({CLUSTER_DESC_ENV: path})
    assert enabled is True
    # The reason has to name the file, because that is what the user has to go
    # and look at when the guard fires.
    assert path in why


def test_descriptor_declaring_translation_off_is_honoured(tmp_path):
    path = _descriptor(tmp_path, PLAIN_DESCRIPTOR)
    assert translation_enabled({CLUSTER_DESC_ENV: path}) is False


def test_missing_descriptor_is_off_and_says_so(tmp_path):
    path = str(tmp_path / "nope.yaml")
    enabled, why = translation_source({CLUSTER_DESC_ENV: path})
    assert enabled is False
    assert "unreadable" in why


def test_explicit_override_beats_the_descriptor(tmp_path):
    path = _descriptor(tmp_path, TRANSLATED_DESCRIPTOR)
    env = {CLUSTER_DESC_ENV: path, TRANSLATION_ENV: "0"}
    enabled, why = translation_source(env)
    assert enabled is False
    assert why == f"{TRANSLATION_ENV}=0"
    env[TRANSLATION_ENV] = "yes"
    assert translation_enabled(env) is True


def test_a_nonsense_override_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="not a boolean"):
        translation_enabled({TRANSLATION_ENV: "maybe"})


def test_blank_override_falls_through_to_the_descriptor(tmp_path):
    path = _descriptor(tmp_path, TRANSLATED_DESCRIPTOR)
    assert translation_enabled({CLUSTER_DESC_ENV: path, TRANSLATION_ENV: ""}) is True


def test_the_checked_in_descriptors_declare_translation():
    # These are the files the docs tell users to export, so if one ever stopped
    # declaring translation the whole feature would silently stop working.
    from driver.blackhole.server.coords import (
        CLUSTER_DESCRIPTOR_PATH as bh_descriptor,
    )
    from driver.wormhole.server.coords import CLUSTER_DESCRIPTOR_PATH as wh_descriptor

    for path in (wh_descriptor, bh_descriptor):
        assert path.exists(), path
        assert descriptor_translation(path) is True, path
