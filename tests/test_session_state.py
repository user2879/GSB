import pytest

from cfgdist.errors import BizError, ErrorCode
from cfgdist.session import (
    ALLOWED_TRANSITIONS,
    SessionStore,
    STATE_COMMITTED,
    STATE_DISTRIBUTING,
    STATE_FAILED,
    STATE_PREPARED,
)


def _plan():
    return {"config_version": 5, "slices": [], "derivations": [], "plan_id": "p"}


def test_state_machine_records_every_transition(tmp_path):
    with SessionStore(tmp_path / "db.sqlite") as store:
        store.create_session("s1", 5, _plan())
        store.transition("s1", STATE_DISTRIBUTING, ["node-a"])
        store.transition("s1", STATE_COMMITTED, ["node-a", "node-b"])
        session = store.get_session("s1")
        assert session["state"] == STATE_COMMITTED
        transitions = store.list_transitions("s1")
        assert [t["to_state"] for t in transitions] == [
            STATE_PREPARED,
            STATE_DISTRIBUTING,
            STATE_COMMITTED,
        ]
        assert transitions[1]["confirmed_nodes"] == ["node-a"]
        for transition in transitions:
            assert transition["config_version"] == 5
            assert "ts" in transition


def test_illegal_transition_rejected(tmp_path):
    with SessionStore(tmp_path / "db.sqlite") as store:
        store.create_session("s1", 5, _plan())
        # prepared -> committed is not a legal direct transition.
        with pytest.raises(BizError) as exc_info:
            store.transition("s1", STATE_COMMITTED, [])
        assert exc_info.value.code == ErrorCode.E_STATE


def test_failed_is_terminal(tmp_path):
    with SessionStore(tmp_path / "db.sqlite") as store:
        store.create_session("s1", 5, _plan())
        store.transition("s1", STATE_DISTRIBUTING, [])
        store.transition("s1", STATE_FAILED, [])
        assert ALLOWED_TRANSITIONS[STATE_FAILED] == set()
        with pytest.raises(BizError):
            store.transition("s1", STATE_COMMITTED, [])


def test_commit_is_idempotent_per_config_version(tmp_path):
    with SessionStore(tmp_path / "db.sqlite") as store:
        store.create_session("s1", 5, _plan())
        store.transition("s1", STATE_DISTRIBUTING, [])
        first = store.commit("s1")
        assert first["idempotent"] is False
        second = store.commit("s1")
        assert second["idempotent"] is True
        assert second["committed"]["session_id"] == "s1"
        rows = store.conn.execute("SELECT COUNT(*) AS n FROM committed_versions").fetchone()
        assert rows["n"] == 1
