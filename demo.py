"""Headless end-to-end demonstration of the subtitle workbench.

Run twice and diff the output directories: every artefact is byte-identical.

    QT_QPA_PLATFORM=offscreen .venv/bin/python demo.py --out run_a
    QT_QPA_PLATFORM=offscreen .venv/bin/python demo.py --out run_b
    diff -r run_a run_b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from subtitle_workbench.ass_export import export_ass, verify_export
from subtitle_workbench.controller import WorkbenchController
from subtitle_workbench.layout import solve_layout
from subtitle_workbench.sample import DEFAULT_SEED, generate_sample, write_sample
from subtitle_workbench.snapshot import SnapshotStore


def main(out_dir: str, seed: int = DEFAULT_SEED) -> dict:
    root = Path(out_dir)
    sample_dir = root / "samples"
    snapshot_dir = root / "snapshots"
    export_dir = root / "exports"
    for directory in (sample_dir, snapshot_dir, export_dir):
        directory.mkdir(parents=True, exist_ok=True)

    log: dict = {"steps": []}

    # 1. deterministic sample ------------------------------------------------
    project = generate_sample(seed)
    sample_path = sample_dir / "sample_project.json"
    write_sample(project, sample_path)
    log["steps"].append({
        "step": "sample",
        "items": len(project.items),
        "shots": len(project.timeline.shots),
        "four_way_clusters": sum(1 for m in project.manifest
                                 if len(m["overlaps_with"]) >= 3),
        "cross_cut_cues": sum(1 for m in project.manifest if m["crosses_cut"]),
        "ad_cues": sum(1 for item in project.items if item.is_ad),
    })

    # 2. initial layout + snapshot ------------------------------------------
    layout = solve_layout(project.items)
    store = SnapshotStore(snapshot_dir)
    v1 = store.commit(project.items, layout)
    log["steps"].append({
        "step": "initial_layout",
        "version": v1.version,
        "layout_hash": v1.layout_hash,
        "arbitrations": [a.to_dict() for a in layout.arbitrations],
    })

    controller = WorkbenchController(store)

    # 3. legal drag -> commit v2 --------------------------------------------
    target = project.items[0]
    controller.begin_drag(target.item_id)
    preview = controller.preview_drag(target.item_id, 5,
                                      target.start_frame + 10,
                                      target.end_frame + 10)
    assert preview.accepted, preview.error
    _, v2 = controller.commit_drag()
    log["steps"].append({
        "step": "legal_drag_commit",
        "item_id": target.item_id,
        "version": v2.version,
        "layout_hash": v2.layout_hash,
    })

    # 4. rejected drags: all three business codes ----------------------------
    rejections = []
    for lane, start, end, expected in (
        (store.load_current().layout.config.total_lanes + 5, 100, 140,
         "E_DRAG_BOUNDS"),
        (2, 500, 500, "E_TIME_INVALID"),
    ):
        controller.begin_drag(target.item_id)
        pv = controller.preview_drag(target.item_id, lane, start, end)
        rejections.append(pv.error.to_dict())
        controller.cancel_drag()

    # E_TRACK_CONFLICT: low-priority cue forced onto a priority-1 cue's lane
    current = store.load_current()
    lane_by_id = {cue.item.item_id: cue.lane_index
                  for cue in current.layout.cues}
    high = next(item for item in current.items if item.priority == 1)
    victim = next(item for item in current.items
                  if item.item_id != high.item_id
                  and item.priority > high.priority)
    controller.begin_drag(victim.item_id)
    pv = controller.preview_drag(victim.item_id, lane_by_id[high.item_id],
                                 high.start_frame + 1, high.end_frame - 1)
    rejections.append(pv.error.to_dict())
    controller.cancel_drag()
    log["steps"].append({"step": "rejected_drags", "records": rejections})
    assert [row["code"] for row in rejections] == [
        "E_DRAG_BOUNDS", "E_TIME_INVALID", "E_TRACK_CONFLICT"]

    # committed state unchanged by the rejected drags
    assert store.load_current().version == v2.version
    log["steps"].append({"step": "state_unchanged_after_rejections",
                         "version": v2.version})

    # 5. cancel restores pre-drag state --------------------------------------
    before_cancel = [item.to_dict() for item in controller.committed_items]
    controller.begin_drag(target.item_id)
    controller.preview_drag(target.item_id, 3, 900, 980)
    restored = controller.cancel_drag()
    after_cancel = [item.to_dict() for item in controller.committed_items]
    log["steps"].append({
        "step": "cancel_restores_state",
        "restored_item_id": restored.item_id,
        "identical": before_cancel == after_cancel,
    })

    # 6. undo back to v1 -----------------------------------------------------
    controller.undo(v1.version)
    log["steps"].append({"step": "undo", "version":
                         store.load_current().version})

    # 7. export ASS + independent verification -------------------------------
    current = store.load_current()
    ass_path = export_dir / "layout_v1.ass"
    export_ass(current, ass_path)
    report = verify_export(ass_path, store._snapshot_path(current.version))
    (export_dir / "verification_report.json").write_bytes(
        json.dumps(report.to_dict(), sort_keys=True, ensure_ascii=False,
                   indent=2).encode("utf-8"))
    log["steps"].append({"step": "export_verify", **report.to_dict()})
    assert report.ok

    # 8. restart recovery: fresh store instance reads only the pointer ------
    recovered = SnapshotStore(snapshot_dir).load_current()
    log["steps"].append({
        "step": "restart_recovery",
        "recovered_version": recovered.version,
        "hash_match": recovered.layout_hash == current.layout_hash,
    })
    assert recovered.version == current.version

    (root / "run_log.json").write_bytes(
        json.dumps(log, sort_keys=True, ensure_ascii=False, indent=2)
        .encode("utf-8"))
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/demo_run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    summary = main(args.out, args.seed)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2))
