"""Repeatable end-to-end demo: three scenarios in a fixed deterministic order.

Scenario 1: normal Filter+Score scheduling with error-bound diagnostics.
Scenario 2: change the fixed-point scale factor, bump the parameter version,
            mark the old report stale and replace the whole conclusion view.
Scenario 3: simulate a forced termination (UNKNOWN in-flight pod), restart from
            the ledger, replay committed pods and rerun the unknown pod.

Run:  python scripts/demo.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from k8ssim.analytics import gap_table, plugin_averages
from k8ssim.session import Session
from k8ssim.storage.recovery import recover_and_replay

WORKDIR = ROOT / "artifacts" / "demo"
LEDGER = WORKDIR / "ledger.sqlite"
NODES_JSON = ROOT / "samples" / "nodes.json"
PODS_JSON = ROOT / "samples" / "pods.json"

LINE = "=" * 72


def _print_header(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def _print_report(report) -> None:
    print(f"parameter_version={report.parameter_version} "
          f"staleness={report.staleness} "
          f"replaced_entries={list(report.replaced_entries)}")
    for decision in report.decisions:
        print(f"- pod {decision.pod_id}: feasible={decision.feasible} "
              f"bound={decision.bound_node} tie={decision.tie_reason} "
              f"error_influenced={decision.error_influenced}")
        for rejection in decision.rejections:
            print(f"    reject node={rejection.node} gate={rejection.gate} "
                  f"reason={rejection.reason}")
        for score in decision.node_scores:
            plugin_text = ", ".join(
                f"{plugin.plugin}={plugin.raw_score}(w{plugin.weight},"
                f"c{plugin.contribution_mp},e{plugin.quant_error_mp})"
                for plugin in score.plugins
            )
            print(f"    node {score.node}: total_mp={score.total_mp} "
                  f"points={score.total_points} "
                  f"error_bound={score.error_bound_mp}mp/{score.error_bound_points}pt")
            print(f"        {plugin_text}")
    averages = plugin_averages(report.decisions)
    print("plugin averages (numpy, read-only):",
          {key: round(value, 2) for key, value in averages.items()})
    print("gap table (numpy, read-only):", gap_table(report.decisions))


def main() -> int:
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)

    # Scenario 1 ---------------------------------------------------------- #
    _print_header("Scenario 1: normal deterministic scheduling")
    session = Session.create(NODES_JSON, PODS_JSON, LEDGER)
    print(f"spec_digest={short(session.spec_digest())}")
    report_v1 = session.run_full_schedule(snapshot_id="snap-v1")
    _print_report(report_v1)
    bindings_v1 = dict(report_v1.bound_nodes)

    # Scenario 2 ---------------------------------------------------------- #
    _print_header("Scenario 2: scale factor change -> version bump, stale guard")
    old_version = session.params.version
    params = session.update_parameters(scale_factor=100)
    print(f"parameter version {old_version} -> {params.version}, "
          f"scale_factor={params.scale_factor}")
    stale = session.engine.stale_reports()
    print("stored stale versions:", [item.parameter_version for item in stale])
    report_v2 = session.run_full_schedule(snapshot_id="snap-v2")
    _print_report(report_v2)
    print("bindings v1:", bindings_v1)
    print("bindings v2:", dict(report_v2.bound_nodes))
    print("snapshots:", [(s["snapshot_id"], s["parameter_version"])
                         for s in session.snapshots()])

    # Scenario 3 ---------------------------------------------------------- #
    _print_header("Scenario 3: forced termination, ledger recovery and replay")
    crash_seq = 3
    committed = session.run_with_crash_marker(crash_before_seq=crash_seq)
    print(f"process killed before seq={crash_seq}; committed pods="
          f"{[d.pod_id for d in committed]}")
    session.close()

    restarted = Session.open(LEDGER)
    recovery = restarted.recover()
    print("recovery summary:", recovery.summary())
    assert recovery.unknown_pods == [restarted.pods[crash_seq].pod_id]
    assert not recovery.verified, "UNKNOWN pod must block full verification"
    rerun = restarted.rerun_unknown()
    print("reran unknown pods:", [d.pod_id for d in rerun])
    verify = recover_and_replay(restarted.ledger, restarted.engine)
    print("post-rerun recovery summary:", verify.summary())
    assert verify.verified, "ledger must fully verify after rerun"
    final_bindings = {
        row["pod_id"]: row["decision"]["bound_node"]
        for row in restarted.ledger.load_pod_results()
    }
    print("final ledger bindings:", final_bindings)
    expected = {
        decision.pod_id: decision.bound_node for decision in report_v2.decisions
    }
    print("v2 conclusions match ledger:", final_bindings == expected)
    assert final_bindings == expected, "replayed ledger diverges from v2 report"
    restarted.close()

    _print_header("Demo completed successfully")
    return 0


def short(digest: str) -> str:
    return digest[:12]


if __name__ == "__main__":
    raise SystemExit(main())
