"""Top-level orchestration: compile plan, register, judge hits, invalidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .digest import digest_file
from .entries import Entry
from .graph import RefGraph
from .manifest import Manifest, load_manifest
from .scheduler import BudgetScheduler, TaskResult, TaskSpec
from .snapshot import SnapshotLibrary
from .state import StateStore
from .store import CacheStore

HIT = "HIT"
MISS = "MISS"
REGISTERED = "REGISTERED"
INVALIDATED = "INVALIDATED"

REASON_NEW = "no_existing_entry"
REASON_MATCH = "digest_refs_version_match"
REASON_DIGEST = "digest_mismatch"
REASON_REFS = "refs_mismatch"
REASON_VERSION = "version_mismatch"
REASON_LEGACY_REFS = "legacy_entry_refs_normalized_empty"
REASON_INVALID = "entry_previously_invalidated"


@dataclass
class EngineConfig:
    global_limit: int = 2
    default_stage_quota: int = 1
    stage_quotas: dict = None
    jitter_ms: int = 0


class Engine:
    def __init__(self, root, config: EngineConfig | None = None):
        self.root = Path(root)
        self.config = config or EngineConfig()
        self.cache_dir = self.root / "cache"
        self.state_dir = self.root / "state"
        self.snapshot_dir = self.root / "snapshots"
        self.store = CacheStore(self.cache_dir)
        self.state = StateStore(self.state_dir)
        self.snapshots = SnapshotLibrary(self.snapshot_dir)
        self.manifest: Manifest | None = None
        self.graph = RefGraph()
        self.last_records = []
        self._hit_seq = 0

    # ------------------------------------------------------------------ plan
    def load(self, manifest_path):
        self.manifest = load_manifest(manifest_path, self.root)
        self.state.set_meta(
            "pipeline_version", self.manifest.pipeline_version
        )
        self.graph = self._build_graph(self.manifest)
        self.graph.require_acyclic()
        return self.compile_plan()

    def _build_graph(self, manifest: Manifest) -> RefGraph:
        graph = RefGraph()
        for decl in manifest.artifacts.values():
            graph.add_node(decl.key)
        for decl in manifest.artifacts.values():
            for ref in decl.refs:
                graph.add_edge(ref.key, decl.key)
        return graph

    def compile_plan(self):
        tasks = []
        for stage in self.manifest.stages:
            for idx, decl in enumerate(stage.artifacts):
                tasks.append(
                    {
                        "key": decl.key,
                        "stage_id": decl.stage_id,
                        "order": len(tasks),
                        "version": decl.version,
                        "file": decl.rel_path,
                        "refs": [r.to_dict() for r in decl.refs],
                    }
                )
        return {"task_count": len(tasks), "tasks": tasks}

    # ------------------------------------------------------------- judgments
    def _evaluate(self, decl):
        """Pure judgment: no state mutation. Returns (outcome, reason, data)."""
        disk_digest = digest_file(decl.abs_path)
        existing = self.state.get_artifact(decl.key)
        try:
            entry = self.store.read_entry(
                decl.key,
                size=decl.abs_path.stat().st_size,
                stage_id=decl.stage_id,
            )
            has_entry = True
        except KeyError:
            entry = None
            has_entry = False

        if not has_entry or existing is None:
            outcome, reason = MISS, REASON_NEW
        elif existing["status"] == INVALIDATED:
            outcome, reason = MISS, REASON_INVALID
        else:
            outcome, reason = self._compare(decl, entry, existing, disk_digest)
        return outcome, reason, disk_digest, entry, has_entry

    def _judge(self, decl) -> TaskResult:
        outcome, reason, disk_digest, entry, has_entry = self._evaluate(decl)
        existing = self.state.get_artifact(decl.key)
        if outcome == HIT:
            self.state.upsert_artifact(
                decl.key, decl.stage_id, decl.version, disk_digest,
                [r.to_dict() for r in decl.refs], HIT,
                source_format=entry.source_format,
                backfilled=entry.backfilled,
            )
        else:
            self._register(decl, disk_digest)
        self._hit_seq += 1
        self.state.record_hit(
            decl.key, outcome, reason, disk_digest, self._hit_seq
        )
        return TaskResult(
            key=decl.key,
            stage_id=decl.stage_id,
            outcome=outcome,
            reason=reason,
            detail={"digest": disk_digest},
        )

    def _compare(self, decl, entry: Entry, existing, disk_digest):
        if entry.digest != disk_digest or existing["digest"] != disk_digest:
            return MISS, REASON_DIGEST
        if existing["version"] != decl.version:
            return MISS, REASON_VERSION
        canonical = tuple(sorted(r.as_tuple() for r in decl.refs))
        if entry.source_format < 3 and canonical:
            return MISS, REASON_LEGACY_REFS
        if entry.normalized_refs() != canonical:
            return MISS, REASON_REFS
        return HIT, REASON_MATCH

    def _register(self, decl, disk_digest):
        entry = Entry(
            key=decl.key,
            digest=disk_digest,
            version=decl.version,
            size=decl.abs_path.stat().st_size,
            stage_id=decl.stage_id,
            refs=tuple(r.as_tuple() for r in decl.refs),
            compressed=False,
            source_format=3,
        )
        self.store.write_entry(entry)
        self.state.upsert_artifact(
            decl.key, decl.stage_id, decl.version, disk_digest,
            [r.to_dict() for r in decl.refs], REGISTERED,
            source_format=3,
        )

    # --------------------------------------------------------------- execute
    def run_plan(self):
        assert self.manifest is not None, "call load() first"
        quotas = dict(self.config.stage_quotas or {})
        for stage in self.manifest.stages:
            quotas.setdefault(stage.stage_id, self.config.default_stage_quota)
        scheduler = BudgetScheduler(
            self.config.global_limit,
            quotas,
            jitter_ms=self.config.jitter_ms,
        )
        specs = []
        for stage in self.manifest.stages:
            for decl in stage.artifacts:
                specs.append(
                    TaskSpec(
                        key=decl.key,
                        stage_id=decl.stage_id,
                        order=len(specs),
                        payload=decl,
                    )
                )

        def worker(spec, slot, start_tick):
            return self._judge(spec.payload)

        records, results = scheduler.run(specs, worker)
        self.last_records = records
        self.state.save_schedule(records)
        return records, results

    # --------------------------------------------------------- change/invalid
    def note_change(self, changed_key):
        """Recompute one artifact from disk and propagate invalidation."""
        decl = self.manifest.artifacts[changed_key]
        order = self.graph.invalidation_order([changed_key])
        new_digest = digest_file(decl.abs_path)
        trail = []
        for position, key in enumerate(order):
            self.state.mark_status(key, INVALIDATED)
            trail.append({"position": position, "key": key})
        self.state.record_invalidation(changed_key, order, order)
        return {
            "root": changed_key,
            "new_digest": new_digest,
            "invalidation_order": order,
            "trail": trail,
        }

    def delete_artifact(self, key):
        self.graph.require_removable(key)
        self.store.delete_entry(key)

    # ------------------------------------------------------ authoritative run
    def authoritative_recompute(self):
        """Single-threaded, side-effect-free ground truth for UI verification."""
        outcomes = {}
        for decl in sorted(self.manifest.artifacts.values(), key=lambda d: d.key):
            outcome, reason, _, _, _ = self._evaluate(decl)
            outcomes[decl.key] = (outcome, reason)
        return outcomes

    def conclusions_match_authoritative(self, results):
        authoritative = self.authoritative_recompute()
        for key, result in results.items():
            if (result.outcome, result.reason) != authoritative[key]:
                return False, key
        return True, None

    # -------------------------------------------------------------- snapshots
    def session_doc(self):
        rows = self.state.all_artifacts()
        digest_set = sorted(a["digest"] for a in rows)
        return {
            "pipeline_version": int(
                self.state.get_meta("pipeline_version", 1)
            ),
            "entry_digest_set": digest_set,
            "graph_version": self.graph.graph_version,
            "invalidations": self.state.invalidation_trails(),
            "hits": self.state.hit_history(),
            "artifacts": rows,
            "schedule": self.state.schedule_rows(),
        }

    def save_snapshot(self, name):
        return self.snapshots.create(name, self.session_doc())

    def restore_snapshot(self, name):
        manifest, session_doc = self.snapshots.open(name)
        diff = self.snapshots.compare(name, self.session_doc())
        # Full replacement, never incremental merge.
        self.state.replace_with(
            {
                "artifacts": session_doc["artifacts"],
                "hits": session_doc["hits"],
                "invalidations": session_doc["invalidations"],
                "schedule": session_doc["schedule"],
                "meta": {"pipeline_version": session_doc["pipeline_version"]},
            }
        )
        # Rebuild the cache view entirely from snapshot digest set.
        for path in self.store.scan_entry_files():
            path.unlink()
        for art in session_doc["artifacts"]:
            restored_size = art.get("size")
            if restored_size is None and self.manifest and art["key"] in \
                    self.manifest.artifacts:
                restored_size = self.manifest.artifacts[art["key"]]\
                    .abs_path.stat().st_size
            self.store.write_entry(
                Entry(
                    key=art["key"],
                    digest=art["digest"],
                    version=art["version"],
                    size=restored_size,
                    stage_id=art["stage_id"],
                    refs=tuple(
                        (r["key"], r["digest"], r["version"])
                        for r in _loads(art["refs_json"])
                    ),
                    compressed=False,
                )
            )
            source = art.get("source_format") or 3
            self.store.source_versions[art["key"]] = source
            backfills = _loads(art.get("backfilled_json", "[]"))
            if backfills:
                self.store.backfilled[art["key"]] = tuple(backfills)
        self.snapshots.mark_readonly(name)
        return {"restored": name, "diff": diff, "manifest": manifest}

    # --------------------------------------------------------------- views
    def diagnostic_view(self):
        rows = self.state.all_artifacts()
        last = self.state.last_outcomes()
        for row in rows:
            outcome = last.get(row["key"])
            row["hit_outcome"] = outcome["outcome"] if outcome else None
            row["hit_reason"] = outcome["reason"] if outcome else None
            row["refs"] = _loads(row["refs_json"])
            row["backfilled"] = _loads(row["backfilled_json"])
        return {
            "stages": [
                {
                    "stage_id": st.stage_id,
                    "order": st.order,
                    "artifacts": [a.key for a in st.artifacts],
                }
                for st in (self.manifest.stages if self.manifest else [])
            ],
            "artifacts": rows,
            "trails": self.state.invalidation_trails(),
            "slot_timeline": self._slot_timeline(),
            "snapshots": self.snapshots.list_snapshots(),
            "source_versions": dict(sorted(self.store.source_versions.items())),
            "backfilled_fields": dict(sorted(self.store.backfilled.items())),
        }

    def _slot_timeline(self):
        # Reconstruct the per-slot view from persisted schedule rows.
        timeline: dict[int, list] = {}
        for row in self.state.schedule_rows():
            timeline.setdefault(row["slot"], []).append(
                {
                    "key": row["key"],
                    "stage_id": row["stage_id"],
                    "start_tick": row["start_tick"],
                    "end_tick": row["end_tick"],
                }
            )
        return [
            {"slot": slot, "occupancy": sorted(
                timeline[slot], key=lambda o: o["start_tick"]
            )}
            for slot in sorted(timeline)
        ]

    def close(self):
        self.state.close()


def _loads(value):
    import json

    return json.loads(value) if isinstance(value, str) else value
