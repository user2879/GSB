from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from probelease import DependencyError, GrantError, SessionValidationError
from probelease.declarations import declaration_from_dicts, load_declarations

from conftest import SAMPLES


def _documents():
    return (
        json.loads((SAMPLES / "sessions.json").read_text(encoding="utf-8")),
        json.loads((SAMPLES / "probes.json").read_text(encoding="utf-8")),
        json.loads((SAMPLES / "ack_policy.json").read_text(encoding="utf-8")),
    )


def _load_modified(tmp_path: Path, mutator):
    sessions, probes, policy = _documents()
    mutator(sessions, probes, policy)
    return declaration_from_dicts(sessions, probes, policy, tmp_path)


@pytest.mark.parametrize(
    ("name", "code", "field_path"),
    [
        ("sessions_corrupt.json", "E_SESSION", "sessions"),
        ("sessions_ack_version_mismatch.json", "E_GRANT", "sessions.ack_policy_version"),
        (
            "sessions_dependency_cycle.json",
            "E_DEPEND",
            "sessions.sessions[4].depends_on[1]",
        ),
    ],
)
def test_broken_sample_codes(name, code, field_path):
    with pytest.raises((SessionValidationError, GrantError, DependencyError)) as caught:
        load_declarations(
            SAMPLES.parent / "broken" / name,
            SAMPLES / "probes.json",
            SAMPLES / "ack_policy.json",
        )
    assert caught.value.code == code
    assert caught.value.field_path == field_path


def test_valid_declaration_digest_is_stable(declarations):
    assert declarations.content_digest == (
        "c44ec48f538f0203e9cb7d5c5b46618926e8edfd76afe6ef3348261928a35786"
    )
    assert [session.session_id for session in declarations.sessions] == [
        "alpha",
        "lossy",
        "late",
        "mismatch",
        "child",
    ]


@pytest.mark.parametrize(
    ("mutator", "field_path"),
    [
        (
            lambda s, p, a: s["sessions"][1].__setitem__(
                "session_id", s["sessions"][0]["session_id"]
            ),
            "sessions.sessions[1].session_id",
        ),
        (
            lambda s, p, a: s["sessions"][1].__setitem__("port", 18101),
            "sessions.sessions[1].port",
        ),
        (
            lambda s, p, a: s["sessions"][0].__setitem__("datagram_timeout_ms", 220),
            "sessions.sessions[0].datagram_timeout_ms",
        ),
        (
            lambda s, p, a: s["sessions"][0].__setitem__(
                "ack_confirmation_floor_bp", 10001
            ),
            "sessions.sessions[0].ack_confirmation_floor_bp",
        ),
        (
            lambda s, p, a: s["sessions"][0].__setitem__("poll_period_ms", True),
            "sessions.sessions[0].poll_period_ms",
        ),
        (
            lambda s, p, a: s["sessions"][0].__setitem__("depends_on", ["missing"]),
            "sessions.sessions[0].depends_on[0]",
        ),
        (
            lambda s, p, a: p["probes"]["alpha_drop"].__setitem__("behavior", "random"),
            "probes.probes[alpha_drop].behavior",
        ),
    ],
)
def test_first_invalid_field_path(tmp_path, mutator, field_path):
    with pytest.raises(SessionValidationError) as caught:
        _load_modified(tmp_path, mutator)
    assert caught.value.code == "E_SESSION"
    assert caught.value.field_path == field_path


def test_ack_policy_version_mismatch_is_grant_error(tmp_path):
    def mutate(sessions, probes, policy):
        sessions["ack_policy_version"] = 8

    with pytest.raises(GrantError) as caught:
        _load_modified(tmp_path, mutate)
    assert caught.value.code == "E_GRANT"
    assert caught.value.field_path == "sessions.ack_policy_version"


def test_no_partial_load_on_invalid_dependency(tmp_path):
    def mutate(sessions, probes, policy):
        sessions["sessions"][0]["depends_on"] = ["does_not_exist"]

    with pytest.raises(SessionValidationError) as caught:
        _load_modified(tmp_path, mutate)
    assert caught.value.code == "E_SESSION"
    assert caught.value.field_path == "sessions.sessions[0].depends_on[0]"
    # No declaration object is returned; the helper never writes a ledger.
    assert list(tmp_path.glob("*.sqlite3")) == []
