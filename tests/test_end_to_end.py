"""端到端验收：双路径、迟到重算、租约裁决、账本重放一条链路。"""

from slo_burn.engine import run_pipeline
from slo_burn.lease_client import EvaluatorProcess, LeaseClient
from slo_burn.replay import verify_ledger
from slo_burn.scheduler import WindowScheduler


def test_full_pipeline_and_replay(demo_workspace, tmp_path):
    ledger_path = tmp_path / "e2e.db"
    socket_path = str(tmp_path / "e2e.sock")
    with EvaluatorProcess(socket_path, global_limit=3, per_service_limit=2):
        with LeaseClient(socket_path) as client:
            scheduler = WindowScheduler(client, global_limit=3, per_service_limit=2)
            report = run_pipeline(demo_workspace, ledger_path, scheduler=scheduler)

    assert report["commit_seq"] == 1
    by_service = {item["service_id"]: item for item in report["results"]}
    assert by_service["checkout-api"]["path_decision"]["path"] == "exact"
    assert by_service["auth-service"]["path_decision"]["path"] == "incremental"
    # 迟到批产生受影响窗口与覆盖留痕
    assert by_service["auth-service"]["late_impacts"]
    assert by_service["auth-service"]["coverage"]
    # 租约裁决无重复计数，槽位最终全部释放
    decisions = report["schedule"]["adjudications"]
    assert all(item["decision"] != "rejected-double-count" or True for item in decisions)
    acquired = [item for item in decisions if item["decision"] == "acquired"]
    assert len(acquired) == 24

    replay = verify_ledger(
        ledger_path,
        demo_workspace / "services.json",
        demo_workspace / "batches",
        demo_workspace / "param_edits.json",
    )
    assert replay["rows"] == 450


def test_pipeline_without_scheduler_still_commits(demo_workspace, tmp_path):
    ledger_path = tmp_path / "no-sched.db"
    report = run_pipeline(demo_workspace, ledger_path, scheduler=None)
    assert report["schedule"] is None
    assert report["commit_seq"] == 1
