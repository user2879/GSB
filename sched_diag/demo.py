"""Deterministic three-scenario acceptance demonstration."""
from __future__ import annotations

import shutil
import stat
from pathlib import Path
from typing import Callable

from .manifest import load_manifests
from .persistence import Persistence
from .reporting import invalidation_lines, pod_lines, print_lines, session_lines
from .restore import replay_payload, restore_payload
from .scheduler import WorkbenchSession


def _sample_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parent.parent
    return root / "samples" / "nodes.json", root / "samples" / "pods.json", root / "samples" / "plugins.json"


def _reset_demo_workdir(workdir: Path) -> None:
    resolved = workdir.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ValueError(f"refusing to reset unsafe demo workdir: {resolved}")
    if resolved.exists():
        def make_writable_and_retry(function: Callable[..., None], path: str, _exc: object) -> None:
            try:
                Path(path).chmod(stat.S_IWRITE | stat.S_IREAD)
                function(path)
            except OSError:
                raise

        shutil.rmtree(resolved, onerror=make_writable_and_retry)
    resolved.mkdir(parents=True, exist_ok=True)


def run_demo(
    workdir: str | Path = "artifacts/demo-state",
    *,
    printer: Callable[[str], None] = print,
    reset: bool = True,
) -> dict[str, object]:
    workdir_path = Path(workdir)
    if reset:
        _reset_demo_workdir(workdir_path)
    persistence = Persistence(workdir_path)
    nodes_path, pods_path, plugins_path = _sample_paths()
    project_root = Path(__file__).resolve().parent.parent
    nodes, pods, plugins = load_manifests(nodes_path, pods_path, plugins_path)

    print_lines(
        [
            "SCENARIO 1: deterministic full Filter/Score schedule",
            (
                "  manifests "
                f"nodes={nodes_path.relative_to(project_root)} "
                f"pods={pods_path.relative_to(project_root)} "
                f"plugins={plugins_path.relative_to(project_root)}"
            ),
        ],
        printer,
    )
    session = WorkbenchSession(nodes, pods, plugins)
    session.full_schedule()
    print_lines(session_lines(session, title="FULL_SCHEDULE", include_all_scores=True), printer)
    verify = session.verify_full_consistency()
    persistence.save_current(session)
    persistence.log(
        "INFO",
        "demo_full_schedule",
        digest=session.content_digest(),
        pods=len(session.pods),
        stats=len(session.stats),
    )
    print_lines(
        [
            f"  FULL_VERIFY ok={verify['ok']} pods={verify['podCount']} stats={verify['statCount']}",
        ],
        printer,
    )

    print_lines(
        [
            "SCENARIO 2: incremental node-label change node-west-a disk: ssd -> hdd",
        ],
        printer,
    )
    before_versions = {pod.pod_id: session.results[pod.pod_id]["version"] for pod in session.pods}
    report = session.edit_node_label("node-west-a", "disk", "hdd")
    print_lines(invalidation_lines(report), printer)
    print_lines(
        [
            f"  recomputed_pods={','.join(session.last_recomputed_pods)}",
            f"  unaffected={session._unaffected_check}",
            f"  versions_before={before_versions}",
            f"  versions_after={ {pod.pod_id: session.results[pod.pod_id]['version'] for pod in session.pods} }",
        ],
        printer,
    )
    for pod_id in session.last_recomputed_pods:
        print_lines(pod_lines(session, pod_id, include_all_scores=True), printer)
    incremental_verify = session.verify_full_consistency()
    incremental_unaffected = session._unaffected_check
    persistence.save_current(session)
    persistence.log(
        "INFO",
        "demo_incremental_schedule",
        digest=session.content_digest(),
        affectedPods=report["pods"],
        affectedStats=len(report["stats"]),
    )
    print_lines(
        [
            (
                f"  INCREMENTAL_VERIFY ok={incremental_verify['ok']} "
                f"pods={incremental_verify['podCount']} stats={incremental_verify['statCount']} "
                f"unaffected_byte_identical={incremental_verify['unaffectedCheck']['byte_identical']}"
            )
        ],
        printer,
    )

    snapshot = persistence.save_snapshot(session, "after-increment")
    print_lines(
        [
            "SCENARIO 3: snapshot save, divergent mutation, whole replacement and replay",
            (
                f"  snapshot name={snapshot['name']} revision={snapshot['revision']} "
                f"digest={snapshot['digest']} backup={snapshot['backupPath']}"
            ),
        ],
        printer,
    )
    session.edit_node_label("node-east-a", "disk", "nvme")
    session.edit_pod_resources("probe-west", 2, 512)
    session.verify_full_consistency()
    persistence.save_current(session)
    divergent_versions = session.versions()
    print_lines(
        [
            (
                f"  divergent_current versions={divergent_versions} "
                f"digest={session.content_digest()}"
            )
        ],
        printer,
    )

    payload = persistence.load_snapshot("after-increment")
    restored, restore_report = restore_payload(payload, current_session=session)
    print_lines(
        [
            (
                f"  RESTORE replaceRequired={restore_report['replaceRequired']} "
                f"versionsMatch={restore_report['versionsMatch']} digestMatch={restore_report['digestMatch']}"
            ),
            f"  RESTORE changedNodes={restore_report['changedNodes']}",
            f"  RESTORE changedPods={restore_report['changedPods']}",
        ],
        printer,
    )
    replay = replay_payload(payload)
    print_lines(
        [
            (
                f"  REPLAY ok={replay['ok']} invalidationPods={replay['invalidation']['pods']} "
                f"steps={replay['invalidation']['steps']} fullRecomputeMatched={replay['fullRecomputeMatched']}"
            ),
            f"  REPLAY comparedPods={replay['comparedPods']}",
        ],
        printer,
    )
    persistence.save_current(restored)
    persistence.log(
        "INFO",
        "demo_snapshot_replay",
        snapshot="after-increment",
        revision=snapshot["revision"],
        digest=payload["contentDigest"],
        replayedAffectedPods=replay["replayedAffectedPods"],
    )
    snapshots = persistence.list_snapshots()
    print_lines(
        [
            "SNAPSHOT_LIST " + "; ".join(
                f"{item['name']}@r{item['revision']}:{item['digest']}" for item in snapshots
            ),
            "DEMO_OK full_schedule=true incremental=true snapshot_restore=true replay=true",
        ],
        printer,
    )
    return {
        "ok": True,
        "workdir": str(workdir_path),
        "snapshot": snapshot,
        "restore": restore_report,
        "replay": replay,
        "incremental": {
            "affectedPods": report["pods"],
            "affectedStats": len(report["stats"]),
            "unaffected": incremental_unaffected,
        },
    }
