"""End-to-end orchestration: collect, compare, attribute, persist, recover."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .attribution import attribute_findings
from .canonical import canonical_digest
from .compare import NO_RECEIPT, REGRESSION, compare_results
from .differ import diff_manifests
from .errorbound import analyze_error_bounds, quantize_version
from .ledger import (
    LEDGER_FORMAT_VERSION,
    PRECISION_SCHEME_VERSION,
    Ledger,
)
from .model import Registry, SessionConfig, Manifest
from .scheduler import Scheduler, build_jobs, now_ms
from .transport import NodeProcess


@dataclass
class SessionArtifacts:
    diff: dict
    error_bounds: dict
    findings: list[dict]
    attributions: list[dict]
    collection: list[dict]
    unknown_items: list[dict] = field(default_factory=list)
    snapshot_report: dict = field(default_factory=dict)
    replay_report: dict = field(default_factory=dict)
    verdict: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "diff": self.diff,
            "error_bounds": self.error_bounds,
            "findings": self.findings,
            "attributions": self.attributions,
            "collection": self.collection,
            "unknown_items": self.unknown_items,
            "snapshot_report": self.snapshot_report,
            "replay_report": self.replay_report,
        }


def start_nodes(registry: Registry) -> dict[str, NodeProcess]:
    processes: dict[str, NodeProcess] = {}
    try:
        for node_id in sorted(registry.nodes):
            proc = NodeProcess(registry.nodes[node_id])
            proc.start()
            processes[node_id] = proc
    except Exception:
        for proc in processes.values():
            proc.stop()
        raise
    return processes


def stop_nodes(processes: dict[str, NodeProcess]) -> None:
    for proc in processes.values():
        proc.stop()


def _scheduler(session: SessionConfig) -> Scheduler:
    return Scheduler(
        global_concurrency=session.global_concurrency,
        node_concurrency=session.node_concurrency,
        socket_timeout_ms=session.socket_timeout_ms,
        max_attempts=session.max_attempts,
        retry_backoff_ms=session.retry_backoff_ms)


def _collection_records(results: list) -> list[dict]:
    records = []
    for result in results:
        records.append({
            "job_id": result.job_id,
            "node_id": result.node_id,
            "manifest_version": result.manifest_version,
            "status": result.status,
            "queue_wait_ms": result.queue_wait_ms,
            "wait_started_monotonic_ms": result.wait_started_monotonic_ms,
            "wait_ended_monotonic_ms": result.wait_ended_monotonic_ms,
            "attempts": [
                {"attempt": rec.attempt,
                 "started_monotonic_ms": rec.started_monotonic_ms,
                 "ended_monotonic_ms": rec.ended_monotonic_ms,
                 "outcome": rec.outcome,
                 "error": rec.error}
                for rec in result.attempts],
            "error": result.error,
        })
    return sorted(records, key=lambda rec: (rec["node_id"],
                                            rec["manifest_version"]))


def _snapshot_payload(baseline: Manifest, candidate: Manifest,
                      session: SessionConfig) -> dict:
    return {
        "baseline_manifest_id": baseline.manifest_id,
        "candidate_manifest_id": candidate.manifest_id,
        "baseline_version": baseline.version,
        "candidate_version": candidate.version,
        "scale_exponent": session.scale_exponent,
        "precision_scheme_version": PRECISION_SCHEME_VERSION,
        "ledger_format_version": LEDGER_FORMAT_VERSION,
        "base_seed": session.base_seed,
    }


def run_session(baseline: Manifest, candidate: Manifest,
                registry: Registry, session: SessionConfig,
                db_path: str) -> SessionArtifacts:
    diff_obj = diff_manifests(baseline, candidate)
    error_bounds = analyze_error_bounds(baseline, candidate, diff_obj,
                                        session.scale_exponent)
    q_base = quantize_version(baseline, session.scale_exponent)
    q_cand = quantize_version(candidate, session.scale_exponent)

    ledger = Ledger(db_path)
    snapshot_report = ledger.verify_or_replace_snapshot(
        _snapshot_payload(baseline, candidate, session),
        LEDGER_FORMAT_VERSION, PRECISION_SCHEME_VERSION)

    base_jobs = build_jobs(baseline, registry, q_base.values, session)
    cand_jobs = build_jobs(candidate, registry, q_cand.values, session)
    processes = start_nodes(registry)
    # Rewrite job endpoints to the ports the subprocesses actually bound.
    for jobs in (base_jobs, cand_jobs):
        for job in jobs:
            job.host = processes[job.node_id].host
            job.port = processes[job.node_id].port
    try:
        scheduler = _scheduler(session)
        base_results = scheduler.run_all(base_jobs)
        cand_results = scheduler.run_all(cand_jobs)
        findings_obj = compare_results(baseline, candidate, session,
                                       base_results, cand_results)
        attributions_obj = attribute_findings(findings_obj, diff_obj,
                                              q_base, q_cand, session)
        _commit_batches(ledger, baseline, candidate, diff_obj, error_bounds,
                        findings_obj, attributions_obj)
    finally:
        stop_nodes(processes)

    findings = [finding.as_dict() for finding in
                sorted(findings_obj, key=lambda f: (f.node_id, f.key))]
    attributions = [attr.as_dict() for attr in attributions_obj]
    stored_findings = stored_payload_findings(ledger)
    replay_report = _safe_replay(ledger, stored_findings, diff_obj,
                                 error_bounds, q_base, q_cand, session)
    ledger.close()
    artifacts = SessionArtifacts(
        diff=diff_obj.as_dict(),
        error_bounds=error_bounds,
        findings=findings,
        attributions=attributions,
        collection=_collection_records(base_results + cand_results),
        snapshot_report=snapshot_report,
        replay_report=replay_report,
    )
    artifacts.verdict = _verdict(findings)
    return artifacts


def _commit_batches(ledger: Ledger, baseline: Manifest, candidate: Manifest,
                    diff_obj, error_bounds: dict, findings_obj,
                    attributions_obj) -> None:
    findings_by_node: dict[str, list] = {}
    for finding in findings_obj:
        findings_by_node.setdefault(finding.node_id, []).append(finding)
    attrs_by_node: dict[tuple[str, str], dict] = {
        (attr.node_id, attr.key): attr.as_dict()
        for attr in attributions_obj}
    all_nodes = sorted(findings_by_node)
    for node_id in all_nodes:
        batch_id = f"batch-{baseline.version}-{candidate.version}-{node_id}"
        ledger.acquire_lease(batch_id, node_id, now_ms(),
                             {"state": "collecting"})
        node_findings = [finding.as_dict()
                         for finding in sorted(findings_by_node[node_id],
                                               key=lambda f: f.key)]
        for finding in node_findings:
            finding["attribution"] = attrs_by_node.get(
                (node_id, finding["key"]))
        batch_payload = {
            "batch_id": batch_id,
            "node_id": node_id,
            "baseline_version": baseline.version,
            "candidate_version": candidate.version,
            "diff_conclusion": diff_obj.as_dict(),
            "error_bounds": error_bounds,
            "findings": node_findings,
            "progress_cursor": {"node_id": node_id, "done": True},
        }
        ledger.commit_batch(batch_id, node_id, baseline.version,
                            candidate.version, now_ms(), batch_payload,
                            node_findings)


def _verdict(findings: list[dict]) -> str:
    if any(f["status"] == REGRESSION for f in findings):
        return REGRESSION
    if any(f["status"] == NO_RECEIPT for f in findings):
        return NO_RECEIPT
    return "ok"


# --------------------------------------------------------------------------- #
# Recovery after forced termination
# --------------------------------------------------------------------------- #
def recover_session(baseline: Manifest, candidate: Manifest,
                    registry: Registry, session: SessionConfig,
                    db_path: str) -> SessionArtifacts:
    """Resume using only fully committed batches; rerun everything else."""

    diff_obj = diff_manifests(baseline, candidate)
    error_bounds = analyze_error_bounds(baseline, candidate, diff_obj,
                                        session.scale_exponent)
    q_base = quantize_version(baseline, session.scale_exponent)
    q_cand = quantize_version(candidate, session.scale_exponent)

    ledger = Ledger(db_path)
    snapshot_report = ledger.verify_or_replace_snapshot(
        _snapshot_payload(baseline, candidate, session),
        LEDGER_FORMAT_VERSION, PRECISION_SCHEME_VERSION)
    open_leases = ledger.open_leases()
    committed_nodes = ledger.committed_nodes()
    unknown_items = [
        {"lease_id": row["lease_id"], "node_id": row["node_id"],
         "state": "unknown_rerun_required",
         "reason": "lease was never committed (forced termination)"}
        for row in open_leases]
    pending_nodes = sorted(set(registry.nodes) - committed_nodes)

    base_jobs = build_jobs(baseline, registry, q_base.values, session)
    cand_jobs = build_jobs(candidate, registry, q_cand.values, session)
    base_jobs = [job for job in base_jobs if job.node_id in pending_nodes]
    cand_jobs = [job for job in cand_jobs if job.node_id in pending_nodes]

    collection_records: list[dict] = []
    findings_obj = []
    if pending_nodes:
        processes = start_nodes(registry)
        for jobs in (base_jobs, cand_jobs):
            for job in jobs:
                job.host = processes[job.node_id].host
                job.port = processes[job.node_id].port
        try:
            scheduler = _scheduler(session)
            base_results = scheduler.run_all(base_jobs)
            cand_results = scheduler.run_all(cand_jobs)
            new_findings = compare_results(baseline, candidate, session,
                                           base_results, cand_results)
            new_attributions = attribute_findings(new_findings, diff_obj,
                                                   q_base, q_cand, session)
            _commit_batches(ledger, baseline, candidate, diff_obj,
                            error_bounds, new_findings, new_attributions)
            collection_records = _collection_records(
                base_results + cand_results)
            findings_obj = list(new_findings)
        finally:
            stop_nodes(processes)

    all_findings = stored_payload_findings(ledger)
    # Replay verification: recompute diff, impact, bounds and judgments from
    # the versions and scale recorded in the ledger.
    replay_report = _safe_replay(ledger, all_findings, diff_obj,
                                 error_bounds, q_base, q_cand, session)
    stored_attributions = _stored_attributions(ledger=ledger)
    ledger.close()
    artifacts = SessionArtifacts(
        diff=diff_obj.as_dict(),
        error_bounds=error_bounds,
        findings=all_findings,
        attributions=stored_attributions,
        collection=collection_records,
        unknown_items=unknown_items,
        snapshot_report=snapshot_report,
        replay_report=replay_report,
    )
    artifacts.verdict = _verdict(all_findings)
    return artifacts


def stored_payload_findings(ledger: Ledger) -> list[dict]:
    """Authoritative ledgered findings, read from committed batch payloads."""

    import json
    rows: list[dict] = []
    for row in ledger.committed_batches():
        payload = json.loads(row["payload_json"])
        rows.extend(payload["findings"])
    return sorted(rows, key=lambda f: (f["node_id"], f["key"]))


def recompute_judgments(session: SessionConfig,
                        ledger: Ledger):
    """Re-run the collection judgment logic from committed raw summaries.

    No node is contacted: the ledger stores both versions' summaries and the
    session stores the tolerances, so status/categories/first-diff-key can be
    recomputed independently and compared against the ledgered conclusions.
    """

    from .compare import Finding, _compare_summaries, REGRESSION, NO_RECEIPT
    findings: list[Finding] = []
    for stored in stored_payload_findings(ledger):
            finding = Finding(node_id=stored["node_id"],
                              key=stored["key"], status="ok")
            finding.baseline_received = stored.get("baseline_received", False)
            finding.candidate_received = stored.get("candidate_received", False)
            if stored["status"] == NO_RECEIPT:
                finding.status = NO_RECEIPT
                finding.categories = list(stored.get("categories", []))
                finding.detail = stored.get("detail")
                findings.append(finding)
                continue
            if finding.baseline_received != finding.candidate_received:
                finding.status = REGRESSION
                finding.categories.append("distribution_target_changed")
                finding.detail = stored.get("detail")
            if finding.baseline_received and finding.candidate_received:
                finding.baseline_summary = stored["baseline_summary"]
                finding.candidate_summary = stored["candidate_summary"]
                finding.baseline_elapsed_ms = stored["baseline_elapsed_ms"]
                finding.candidate_elapsed_ms = stored["candidate_elapsed_ms"]
                finding.elapsed_delta_ms = abs(
                    finding.candidate_elapsed_ms
                    - finding.baseline_elapsed_ms)
                finding.tolerance = {
                    "value_abs": session.value_tolerance,
                    "elapsed_ms": session.elapsed_tolerance_ms}
                first_group, deltas, value_regress = _compare_summaries(
                    stored["baseline_summary"], stored["candidate_summary"],
                    session.value_tolerance)
                finding.first_diff_group = first_group
                finding.summary_deltas = deltas
                elapsed_regress = (finding.elapsed_delta_ms
                                   > session.elapsed_tolerance_ms)
                if value_regress:
                    finding.status = REGRESSION
                    finding.categories.append("summary_out_of_tolerance")
                if elapsed_regress:
                    finding.status = REGRESSION
                    finding.categories.append("elapsed_out_of_tolerance")
            findings.append(finding)
    return sorted(findings, key=lambda f: (f.node_id, f.key))


def _safe_replay(ledger: Ledger, stored_findings: list[dict],
                 diff_obj, error_bounds: dict, q_base, q_cand,
                 session) -> dict:
    import json
    # The embedded attribution is verified separately; exclude it from the
    # field-by-field finding comparison.
    stripped_stored = [{k: v for k, v in row.items() if k != "attribution"}
                       for row in stored_findings]
    replayed_obj = recompute_judgments(session, ledger)
    replayed_findings = [finding.as_dict() for finding in replayed_obj]
    replayed_attrs = [attr.as_dict() for attr in attribute_findings(
        replayed_obj, diff_obj, q_base, q_cand, session)]
    stored_attrs = _stored_attributions(ledger=ledger)

    structural_violations: list[tuple[str, str]] = []
    stored_diff_digest = stored_bound_digest = None
    for row in ledger.committed_batches():
        payload = json.loads(row["payload_json"])
        stored_diff_digest = canonical_digest(payload["diff_conclusion"])
        stored_bound_digest = canonical_digest(payload["error_bounds"])
        if stored_diff_digest != canonical_digest(diff_obj.as_dict()):
            structural_violations.append(
                (f"{row['batch_id']}.diff_conclusion",
                 "recomputed diff digest differs from ledger"))
        if stored_bound_digest != canonical_digest(error_bounds):
            structural_violations.append(
                (f"{row['batch_id']}.error_bounds",
                 "recomputed error-bound digest differs from ledger"))
    try:
        if structural_violations:
            from .errors import ReplayError
            raise ReplayError("structural replay mismatch",
                              structural_violations)
        Ledger.replay_check(stripped_stored, replayed_findings,
                            stored_attrs, replayed_attrs)
    except Exception as exc:  # surfaced as E_REPLAY detail, never silent
        return {"status": "E_REPLAY",
                "code": getattr(exc, "code", "E_REPLAY"),
                "violations": getattr(exc, "details", lambda: [])()}
    return {"status": "ok", "checked_findings": len(stripped_stored),
            "checked_attributions": len(stored_attrs),
            "stored_finding_digest": canonical_digest(stripped_stored),
            "replay_finding_digest": canonical_digest(replayed_findings),
            "diff_digest": stored_diff_digest,
            "error_bound_digest": stored_bound_digest}


def _stored_attributions(ledger: Ledger | None = None,
                         ledger_path: str | None = None) -> list[dict]:
    own_ledger = ledger is None
    if own_ledger:
        ledger = Ledger(ledger_path)  # type: ignore[arg-type]
    attrs: list[dict] = []
    for row in ledger.committed_batches():
        import json
        payload = json.loads(row["payload_json"])
        for finding in payload["findings"]:
            if finding.get("attribution"):
                attrs.append(finding["attribution"])
    if own_ledger:
        ledger.close()  # type: ignore[union-attr]
    return sorted(attrs, key=lambda a: (a["node_id"], a["key"]))
