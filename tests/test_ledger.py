from __future__ import annotations

import json
from pathlib import Path

import pytest

from qosfdiag.engine import CascadeEngine
from qosfdiag.errors import LedgerError
from qosfdiag.generator import DeterministicUdpGenerator
from qosfdiag.ledger import DecisionLedger
from qosfdiag.policy import load_policy
from qosfdiag.samples import three_stage_generator, three_stage_policy
from qosfdiag.transport import InMemoryTransport


class SimulatedKill(Exception):
    pass


def _policy():
    return load_policy(three_stage_policy())


def _generator():
    return DeterministicUdpGenerator(three_stage_generator())


def _engine(policy, generator, crash_hook=None):
    return CascadeEngine(
        policy,
        generator,
        scale=3,
        transport=InMemoryTransport([shaper.shaper_id for shaper in policy.shapers]),
        crash_hook=crash_hook,
    )


def _uninterrupted(tmp_path: Path):
    path = tmp_path / "clean.sqlite"
    policy, generator = _policy(), _generator()
    with DecisionLedger(path) as ledger:
        with _engine(policy, generator) as engine:
            result = engine.run(ledger=ledger)
        return result, ledger.progress().content_digest


def test_crash_marks_only_inflight_batch_unknown_and_resume_matches(tmp_path: Path) -> None:
    clean_result, clean_digest = _uninterrupted(tmp_path)
    path = tmp_path / "crash.sqlite"
    policy, generator = _policy(), _generator()

    def crash(batch_no: int, phase: str) -> None:
        if batch_no == 2 and phase == "after_marker":
            raise SimulatedKill()

    with DecisionLedger(path) as ledger:
        with _engine(policy, generator, crash) as engine:
            with pytest.raises(SimulatedKill):
                engine.run(ledger=ledger)

    with DecisionLedger(path) as ledger:
        ledger.initialize(policy=policy, scale=3, generator=generator)
        recovery = ledger.recover()
        assert recovery.requires_rerun is True
        assert recovery.cursor_seq == 8
        assert recovery.unknown_packet_ids == (
            "p00000008",
            "p00000009",
            "p00000010",
            "p00000011",
        )
        assert len(ledger.decisions()) > 0
        assert all(item["resolved"] == 0 for item in ledger.unknown_packets())

    with DecisionLedger(path) as ledger:
        ledger.initialize(policy=policy, scale=3, generator=generator)
        with _engine(policy, generator) as engine:
            resumed = engine.run(ledger=ledger)
        assert resumed.recovery is not None
        assert resumed.recovery.requires_rerun is True
        assert ledger.unknown_packets(unresolved_only=True) == ()
        assert ledger.progress().cursor_seq == 40
        assert ledger.progress().content_digest == clean_digest
        assert resumed.summary == clean_result.summary
        assert [event.as_dict() for event in resumed.events] == [
            event.as_dict() for event in clean_result.events
        ]


def test_content_digest_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "tamper.sqlite"
    policy, generator = _policy(), _generator()
    with DecisionLedger(path) as ledger:
        with _engine(policy, generator) as engine:
            engine.run(ledger=ledger)
        ledger.connection.execute(
            "UPDATE decisions SET decision='TAMPERED' WHERE event_seq=1"
        )
        ledger.connection.commit()
        with pytest.raises(LedgerError, match="digest mismatch"):
            ledger.verify()


def test_ledger_rejects_different_session(tmp_path: Path) -> None:
    path = tmp_path / "session.sqlite"
    policy, generator = _policy(), _generator()
    with DecisionLedger(path) as ledger:
        with _engine(policy, generator) as engine:
            engine.run(ledger=ledger)
    changed = json.loads(json.dumps(three_stage_policy()))
    changed["policy_version"] = "other-version"
    with DecisionLedger(path) as ledger:
        with pytest.raises(LedgerError, match="signature"):
            ledger.initialize(
                policy=load_policy(changed), scale=3, generator=generator
            )
