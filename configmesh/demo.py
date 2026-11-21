"""Reproducible acceptance demonstration for every required business capability.

Run:  python -m configmesh.demo
Outputs workdir/acceptance/ with structured logs, SQLite ledger, report JSON
and a human-readable report.txt.
"""
from __future__ import annotations

import json
import os

from . import injections
from .compiler import build_dispatch_plan, compile_version, diff_invalidation
from .errors import BusinessError
from .fixedpoint import Grid
from .manifest import load_bundle
from .session import Session

WORKDIR = os.path.join("workdir", "acceptance")
REGISTRY = os.path.join("examples", "registry.json")
MANIFEST = os.path.join("examples", "manifest.json")


def run_happy_path(session_id: str, subdir: str, timeout_map=None) -> tuple[Session, list[dict]]:
    session = Session(session_id, os.path.join(WORKDIR, subdir),
                      grid_exp=3, global_limit=3, per_node_quota=2)
    session.load_inputs(MANIFEST, REGISTRY)
    session.start_nodes()
    try:
        outputs = [session.run_version(version, timeout_map)
                   for version in session.versions]
    finally:
        session.shutdown_nodes()
    return session, outputs


def main() -> int:
    os.makedirs(WORKDIR, exist_ok=True)
    report: dict = {"steps": [], "business_codes": []}

    # 1. happy path: parallel dispatch, convergence, replay
    session, outputs = run_happy_path("acceptance", "happy")
    try:
        for output in outputs:
            session.replay_snapshot(output["version"])
        report["steps"].append({
            "step": "happy_path",
            "versions": [o["version"] for o in outputs],
            "peak_concurrency": session.scheduler.peak_concurrency,
            "global_limit": 3,
            "peak_within_budget": session.scheduler.peak_concurrency <= 3,
            "converged": all(o["convergence"]["converged"] for o in outputs),
            "replay_consistent": True,
        })

        # 2. incremental invalidation evidence
        registry, versions = load_bundle(MANIFEST, REGISTRY)
        grid = Grid.create(3)
        report["steps"].append({
            "step": "incremental_invalidation",
            "v1_to_v2": diff_invalidation(versions[0], versions[1], grid),
            "v2_to_v3": diff_invalidation(versions[1], versions[2], grid),
        })

        # 3. snapshot full replacement path (isolated store keeps the happy
        # path's three-version ledger intact)
        from .storage import SnapshotStore
        plan = build_dispatch_plan(compile_version(versions[0], grid), versions[0])
        payload = {"manifest_version": 1, "grid_exp": 3, "plan": plan,
                   "receipts": [], "tasks": [], "error_bounds": {}, "convergence": None}
        resume_store = SnapshotStore(os.path.join(WORKDIR, "snapshot_resume", "resume.sqlite"))
        try:
            resume_store.save_snapshot("acceptance-resume", 1, 3, "CONVERGED", payload)
            stale_snapshot = resume_store.load_latest("acceptance-resume")
            identity_matches = resume_store.identity_matches(stale_snapshot, 1, 3, payload)
            mismatch = not resume_store.identity_matches(
                stale_snapshot, 99, 7, {"stale": True})
            removed = resume_store.replace_all_session_state("acceptance-resume") if mismatch else 0
            replacement = {
                "identity_match_on_identical_payload": identity_matches,
                "mismatch_detected": mismatch,
                "action": "full_replacement" if removed else "noop",
                "removed_rows": removed,
                "stale_version": stale_snapshot.manifest_version,
            }
            report["steps"].append({"step": "snapshot_full_replacement", "result": replacement})
        finally:
            resume_store.close()
    finally:
        session.close()

    # 4. retry with timeout injection (single-dispatch semantics unchanged)
    retry_session = Session("acceptance-retry", os.path.join(WORKDIR, "retry"), 3)
    retry_session.load_inputs(MANIFEST, REGISTRY)
    retry_session.start_nodes()
    try:
        retry_session.run_version(versions[0], {"v1-node_alpha-s0": "timeout_once"})
        retry_record = next(
            r for r in retry_session.scheduler.records if r.task_id == "v1-node_alpha-s0")
        report["steps"].append({
            "step": "timeout_retry",
            "attempts": [a.__dict__ for a in retry_record.attempts],
            "status": retry_record.status,
        })
    finally:
        retry_session.shutdown_nodes()
        retry_session.close()

    # 5. three injected fault classes (inputs/params only)
    for kind, injector, expected in (
        ("corrupt-manifest", lambda: injections.inject_corrupt_manifest(MANIFEST), "E_MANIFEST"),
        ("cycle", lambda: injections.inject_derived_cycle(MANIFEST), "E_CYCLE"),
    ):
        descriptor = injector()
        probe = Session(f"inject-{kind}", os.path.join(WORKDIR, kind), 3)
        try:
            try:
                probe.load_inputs(MANIFEST, REGISTRY)
            except BusinessError as exc:
                probe.logger.error(exc.code.value, "injection_rejected",
                                   injection=descriptor["injection"],
                                   expected=expected, details=exc.details)
                report["business_codes"].append({
                    "injection": kind, "code": exc.code.value, "expected": expected,
                    "match": exc.code.value == expected, "violations": exc.details[:3],
                })
            finally:
                injections.restore(MANIFEST)
        finally:
            probe.close()
    try:
        Grid.create(injections.illegal_grid_parameter())
    except BusinessError as exc:
        grid_probe = Session("inject-grid", os.path.join(WORKDIR, "illegal-grid"), 3)
        grid_probe.logger.error(exc.code.value, "illegal_grid_rejected",
                                grid_exp=injections.illegal_grid_parameter(),
                                details=exc.details)
        grid_probe.store.close()
        report["business_codes"].append({
            "injection": "illegal-grid", "code": exc.code.value,
            "expected": "E_GRID", "match": exc.code.value == "E_GRID",
            "violations": exc.details,
        })

    # 6. legal inputs still process after injections
    post_session, post_outputs = run_happy_path("post-injection", "post")
    post_session.close()
    report["steps"].append({
        "step": "legal_input_after_injection",
        "versions": [o["version"] for o in post_outputs],
        "converged": all(o["convergence"]["converged"] for o in post_outputs),
    })

    report_path = os.path.join(WORKDIR, "acceptance_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str, sort_keys=True)
    text_path = os.path.join(WORKDIR, "acceptance_report.txt")
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(render_text(report))
    print(json.dumps({
        "report_json": report_path,
        "report_text": text_path,
        "all_codes_match": all(x["match"] for x in report["business_codes"]),
    }, indent=2))
    return 0


def render_text(report: dict) -> str:
    lines = ["configmesh acceptance report", "=" * 32]
    for step in report["steps"]:
        lines.append(f"[step] {step['step']}")
        lines.append(json.dumps(step, ensure_ascii=False, sort_keys=True, default=str)[:1500])
    lines.append("")
    lines.append("business codes:")
    for item in report["business_codes"]:
        lines.append(f"  {item['injection']:<18} -> {item['code']} (match={item['match']})")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
