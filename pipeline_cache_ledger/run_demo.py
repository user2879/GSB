"""Repeatable end-to-end demonstration of the cache ledger workbench.

Scenarios, in fixed order:
  1. cold-start registration and dual-budget hit judging
  2. artifact version update -> reference-driven invalidation + snapshot
  3. legacy v1/v2 cache library open upgrade (with simulated kill + resume)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pcl.digest import digest_bytes, digest_file  # noqa: E402
from pcl.engine import (  # noqa: E402
    HIT,
    MISS,
    REGISTERED,
    Engine,
    EngineConfig,
)
from pcl.entries import (  # noqa: E402
    CURRENT_ENTRY_VERSION,
    Entry,
    decode_entry,
    encode_entry,
)
from pcl.report import render_diagnostic_report, render_upgrade_report  # noqa: E402
from pcl.samples import SAMPLE_CONTENT, write_sample_workspace  # noqa: E402
from pcl.store import UpgradeInterrupted  # noqa: E402
from pcl.workbench import TK_AVAILABLE, launch_workbench  # noqa: E402

LEDGER_ROOT = Path(__file__).resolve().parent
WORKSPACE_A = LEDGER_ROOT / "workspace_main"
WORKSPACE_B = LEDGER_ROOT / "workspace_legacy"


def banner(title):
    print("\n" + "#" * 72)
    print("# " + title)
    print("#" * 72)


def fresh(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def scenario_one():
    banner("Scenario 1: cold-start registration + dual-budget scheduling")
    fresh(WORKSPACE_A)
    manifest_path, _ = write_sample_workspace(WORKSPACE_A)

    engine = Engine(
        WORKSPACE_A,
        EngineConfig(global_limit=2, default_stage_quota=1),
    )
    plan = engine.load(manifest_path)
    print(f"compiled plan: {plan['task_count']} tasks across "
          f"{len(engine.manifest.stages)} stages")

    records, results = engine.run_plan()
    print("\nschedule (logical ticks, stable order):")
    for rec in records:
        print(
            f"  {rec.key:<14} stage={rec.stage_id:<17} "
            f"q=t{rec.queued_tick} start=t{rec.start_tick} "
            f"end=t{rec.end_tick} slot=G{rec.slot} "
            f"wait={rec.wait_ticks} -> {rec.outcome}/{rec.reason}"
        )

    outcomes = {k: (v.outcome, v.reason) for k, v in results.items()}
    print("\nfirst-run outcome set:")
    for key in sorted(outcomes):
        print(f"  {key:<14} {outcomes[key][0]:<10} {outcomes[key][1]}")
    assert all(v[0] == MISS for v in outcomes.values()), "cold start must miss"

    # Re-run: identical inputs must now hit, and a second scheduler config
    # with different interleaving must produce the same hit conclusion set.
    engine2 = Engine(
        WORKSPACE_A,
        EngineConfig(global_limit=3, default_stage_quota=2, jitter_ms=1),
    )
    engine2.load(manifest_path)
    _, results2 = engine2.run_plan()
    outcomes2 = {k: (v.outcome, v.reason) for k, v in results2.items()}
    print("\nsecond-run outcome set (different interleaving):")
    for key in sorted(outcomes2):
        print(f"  {key:<14} {outcomes2[key][0]:<10} {outcomes2[key][1]}")
    assert all(v[0] == HIT for v in outcomes2.values()), "rerun must hit"

    # Authoritative single-threaded recomputation must agree with the UI data.
    authoritative = engine2.authoritative_recompute()
    mismatch = [k for k in authoritative if authoritative[k] != outcomes2[k]]
    print(f"\nauthoritative recompute agreement: {not mismatch} {mismatch or ''}")
    assert not mismatch

    engine.save_snapshot("snap-cold-start")
    print("saved snapshot: snap-cold-start")
    return engine, manifest_path


def scenario_two(engine: Engine, manifest_path: Path):
    banner("Scenario 2: artifact version update -> invalidation propagation")
    changed_path = WORKSPACE_A / "artifacts" / "base-lib.txt"
    new_bytes = b"base-lib\ncontent-version=2\nline-a\nline-b\nline-c\n"
    changed_path.write_bytes(new_bytes)
    new_digest = digest_file(changed_path)
    print(f"base-lib content changed -> digest {new_digest}")

    propagation = engine.note_change("base-lib")
    print("\nstable invalidation order (independent of manifest traversal):")
    for step in propagation["trail"]:
        print(f"  {step['position']}. {step['key']}")
    expected = [
        "base-lib", "codegen", "core-bundle",
        "api-image", "release-notes", "web-image",
    ]
    assert propagation["invalidation_order"] == expected

    # Bump declared version and refresh the manifest, then re-run the plan.
    manifest_path2, _ = write_sample_workspace(
        WORKSPACE_A, pipeline_version=2,
        content_overrides={"base-lib": new_bytes},
        version_overrides={"base-lib": 2},
    )
    engine.load(manifest_path2)
    records, results = engine.run_plan()
    print("\npost-change settlements:")
    for rec in records:
        print(f"  {rec.key:<14} {rec.outcome:<10} {rec.reason}")
    summary = {}
    for key, result in sorted(results.items()):
        summary[key] = (result.outcome, result.reason)
    assert summary["base-lib"][0] == MISS
    assert summary["config"][0] == HIT, "unaffected artifact must still hit"

    # Snapshot version mismatch -> full replacement restore (no merge).
    diff = engine.snapshots.compare("snap-cold-start", engine.session_doc())
    print("\nsnapshot vs current disk state:")
    print(json.dumps(diff, indent=2, sort_keys=True))
    assert diff["versions_equal"] is False
    restore = engine.restore_snapshot("snap-cold-start")
    print(
        "restored snapshot (full replacement); readonly="
        f"{restore['diff']['name']} -> old snapshot kept read-only"
    )
    names = [s["name"] for s in engine.snapshots.list_snapshots()]
    assert "snap-cold-start" in names
    return engine


def scenario_three():
    banner("Scenario 3: legacy v1/v2 cache open upgrade + crash resume")
    fresh(WORKSPACE_B)
    manifest_path, paths = write_sample_workspace(WORKSPACE_B)

    # Hand-craft a mixed-version cache library on disk.
    from pcl.store import CacheStore

    store = CacheStore(WORKSPACE_B / "cache")
    v1_key, v2_key, v3_key = "base-lib", "config", "codegen"
    store.cache_dir.mkdir(parents=True, exist_ok=True)

    def write_legacy(key, fmt):
        data = paths[key].read_bytes()
        decl_digest = digest_bytes(data)
        entry = Entry(
            key=key,
            digest=decl_digest,
            version=1,
            size=len(data),
            stage_id="stage-foundation" if key == v2_key else "stage-build",
        )
        blob = encode_entry(entry, fmt)
        store._entry_path(key).write_bytes(blob)

    write_legacy(v1_key, 1)
    write_legacy(v2_key, 2)
    write_legacy(v3_key, CURRENT_ENTRY_VERSION)
    on_disk_before = {
        p.name[: -len(".pcl")]: decode_entry(p.read_bytes(), p.stem).format_version
        for p in store.scan_entry_files()
    }
    print(f"on-disk formats before open: {on_disk_before}")

    # First open: force a kill after the first upgrade; checkpoint is durable.
    try:
        store.open_and_upgrade(simulate_kill_after=1)
    except UpgradeInterrupted as exc:
        print(f"simulated process kill: {exc}")
    cp1 = json.loads((store.cache_dir / "upgrade_checkpoint.json").read_text())
    print(f"checkpoint after kill: processed={cp1['processed']}")
    assert cp1["processed"] == ["base-lib"]

    # Reopen: resume strictly from the checkpoint, no entry processed twice.
    report = store.open_and_upgrade()
    print(render_upgrade_report(report))
    cp2 = json.loads((store.cache_dir / "upgrade_checkpoint.json").read_text())
    print(f"checkpoint after resume: processed={sorted(cp2['processed'])}")
    assert sorted(cp2["processed"]) == sorted(on_disk_before)
    assert len(report["upgraded"]) == 2, "v3 entry must not be rewritten"

    verification = store.upgrade_verification()
    print(f"\nverification: {json.dumps(verification, sort_keys=True)}")
    assert verification["all_current"]
    assert verification["processed_covers_disk"]

    # Cross-version hit consistency:
    #  (a) every normalized digest equals the authoritative file digest;
    #  (b) the SAME key written independently as v1, v2 and v3 normalizes to
    #      the identical digest and identical hit conclusion.
    engine = Engine(
        WORKSPACE_B,
        EngineConfig(global_limit=2, default_stage_quota=1),
    )
    engine.load(manifest_path)
    normalized = {
        key: store.read_entry(
            key,
            size=paths[key].stat().st_size,
            stage_id=engine.manifest.artifacts[key].stage_id,
        )
        for key in (v1_key, v2_key, v3_key)
    }
    backfill_map = {u["key"]: u["backfilled"] for u in report["upgraded"]}
    print("\ncross-version normalized digests vs authoritative files:")
    for key, entry in normalized.items():
        file_digest = digest_file(paths[key])
        print(
            f"  {key}: source=v{entry.source_format} "
            f"backfilled={backfill_map.get(key, [])} match="
            f"{entry.digest == file_digest}"
        )
        assert entry.digest == file_digest

    probe_key = "core-bundle"
    probe_data = paths[probe_key].read_bytes()
    probe_digest = digest_bytes(probe_data)
    same_key_sources = {}
    for fmt in (1, 2, 3):
        legacy = Entry(
            key=probe_key, digest=probe_digest, version=1,
            size=len(probe_data), stage_id="stage-build",
        )
        same_key_sources[fmt] = decode_entry(
            encode_entry(legacy, fmt), probe_key
        ).digest
    print(f"\nsame-key '{probe_key}' digests across v1/v2/v3: "
          f"{same_key_sources}")
    assert len(set(same_key_sources.values())) == 1

    # Build two identical legacy cache libraries and run them under
    # different scheduler interleavings: conclusion sets must match.
    alt_root = WORKSPACE_B.parent / "workspace_legacy_alt"
    fresh(alt_root)
    write_sample_workspace(alt_root)
    alt_store = CacheStore(alt_root / "cache")
    for key, fmt in ((v1_key, 1), (v2_key, 2), (v3_key, 3)):
        data = (alt_root / "artifacts" / f"{key}.txt").read_bytes()
        legacy = Entry(
            key=key, digest=digest_bytes(data), version=1,
            size=len(data), stage_id="stage-foundation",
        )
        alt_store._entry_path(key).write_bytes(encode_entry(legacy, fmt))

    def conclusions_for(root, limit, quota):
        eng = Engine(root, EngineConfig(global_limit=limit,
                                        default_stage_quota=quota))
        eng.load(root / "pipeline_manifest.json")
        eng.store.open_and_upgrade()
        _, res = eng.run_plan()
        return {k: v.outcome for k, v in res.items()}

    conclusions = conclusions_for(WORKSPACE_B, 2, 1)
    alt_conclusions = conclusions_for(alt_root, 3, 2)
    for key in sorted(conclusions):
        print(f"  hit judgment {key:<14} {conclusions[key]}")
    print(f"interleaving-stable conclusions: {alt_conclusions == conclusions}")
    assert alt_conclusions == conclusions
    return engine


def main():
    print("tkinter available in this environment:", TK_AVAILABLE)
    engine_a, manifest_path = scenario_one()
    engine_a = scenario_two(engine_a, manifest_path)
    engine_b = scenario_three()

    banner("Final diagnostic view (main workspace, stable ordering)")
    view = engine_a.diagnostic_view()
    output = launch_workbench(view)
    if isinstance(output, str):
        print(output)
    print(render_diagnostic_report(engine_b.diagnostic_view()))

    banner("Summary")
    print("scenario 1: cold misses -> rerun hits, scheduler budgets respected")
    print("scenario 2: base-lib change invalidated 6 downstream keys in "
          "topological order; snapshot restored via full replacement")
    print("scenario 3: v1/v2 entries atomically upgraded to v3; kill resumed "
          "from checkpoint; cross-version digests and hits identical")
    print("\nworkspaces:")
    print(f"  {WORKSPACE_A}")
    print(f"  {WORKSPACE_B}")


if __name__ == "__main__":
    main()
