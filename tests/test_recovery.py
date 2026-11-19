from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tcp_workbench.errors import E_STATE, BusinessError
from tcp_workbench.models import Event
from tcp_workbench.service import SessionService
from tests.conftest import ROOT, SAMPLES


def test_forced_process_termination_marks_pending_unknown_and_retry(
    service_factory, tmp_path
):
    db_path = tmp_path / "crash.sqlite3"
    script = textwrap.dedent(
        f"""
        import os
        from tcp_workbench.config import load_session_config
        from tcp_workbench.models import Event
        from tcp_workbench.service import SessionService

        config, script = load_session_config({str(SAMPLES / "session.json")!r})

        def crash(_event):
            os._exit(23)

        service = SessionService.open({str(db_path)!r}, config, script, crash_after_begin=crash)
        service.ingest_event(Event(event_type='SYN', recorded_mono_ms=0, seq=1000))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 23, completed.stderr

    recovered = SessionService.open_existing(db_path)
    try:
        report = recovered.recovery_report
        assert report is not None
        assert report.unknown_event_seqs == (1,)
        assert report.rerun_required is True
        assert report.exit_report_available is False
        assert report.complete_event_seqs == ()
        assert "不能生成退出报告" in report.message
        with pytest.raises(BusinessError) as shown:
            recovered.ingest_event(
                Event(event_type="SYN", recorded_mono_ms=1, seq=1000)
            )
        assert shown.value.code == E_STATE
        retried = recovered.retry_unknown_events()
        assert [result.event.event_seq for result in retried] == [1]
        assert recovered.recovery_report.exit_report_available is True
        assert recovered.items["connection_state"].value == "SYN_SENT"
        recovered.verify_incremental_equivalence()
    finally:
        recovered.close()


def test_normal_recovery_uses_only_complete_events(service_factory, sample_events):
    service = service_factory("normal.sqlite3")
    for raw in sample_events:
        service.ingest_event(Event.from_dict({**raw, "event_seq": 0}))
    db_path = service.ledger.path
    service.close()
    recovered = SessionService.open_existing(db_path)
    try:
        assert recovered.recovery_report.complete_event_seqs == tuple(range(1, 21))
        assert recovered.recovery_report.unknown_event_seqs == ()
        assert recovered.recovery_report.exit_report_available is True
        assert recovered.items["connection_state"].value == "CLOSED"
        recovered.verify_incremental_equivalence()
    finally:
        recovered.close()
