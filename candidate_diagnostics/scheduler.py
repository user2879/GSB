from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import Config
from .engine import EngineRun, IncrementalEngine
from .errors import E_POINT_BUDGET_PENDING, E_REQUEST_QUOTA_EXCEEDED, Failure
from .models import Request


def atomic_write_json_via_temp_dir(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".tmp-{path.stem}-", dir=path.parent) as tmp_name:
        tmp_path = Path(tmp_name) / path.name
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)


class SnapshotStore:
    def __init__(self, root: str | Path, config: Config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.manifest_path = self.root / "manifest.json"

    def begin_run(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            same_config = previous.get("config_hash") == self.config.config_hash
            same_weights = (
                previous.get("channel_weights_hash")
                == self.config.channel_weights_fingerprint
            )
            if same_config and same_weights:
                manifest = dict(previous)
                manifest["startup_decision"] = "RESUME_SAME_VERSION"
                manifest["resume_count"] = int(previous.get("resume_count", 0)) + 1
            else:
                manifest = {
                    "global_version": int(previous.get("global_version", 1)) + 1,
                    "previous_global_version": previous.get("global_version", 1),
                    "config_hash": self.config.config_hash,
                    "channel_weights_hash": self.config.channel_weights_fingerprint,
                    "sampling_hash": self.config.sampling_fingerprint,
                    "completed_cursor": None,
                    "completed_requests": [],
                    "pending_queue": [],
                    "startup_decision": "RESTART_VERSION_CHANGED",
                    "version_change_reason": (
                        "config_hash_changed" if not same_config else "channel_weights_hash_changed"
                    ),
                }
        else:
            manifest = {
                "global_version": 1,
                "config_hash": self.config.config_hash,
                "channel_weights_hash": self.config.channel_weights_fingerprint,
                "sampling_hash": self.config.sampling_fingerprint,
                "completed_cursor": None,
                "completed_requests": [],
                "pending_queue": [],
                "startup_decision": "FRESH_START",
            }
        self.commit_json("manifest", manifest)
        return manifest

    def commit_json(self, stage: str, payload: Any) -> Path:
        path = self.root / f"{stage}.json"
        atomic_write_json_via_temp_dir(path, payload)
        return path

    def commit_progress(
        self,
        *,
        completed: list[str],
        pending: list[str],
        cursor: str | None,
        extra_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "global_version": 1,
            "config_hash": self.config.config_hash,
            "channel_weights_hash": self.config.channel_weights_fingerprint,
            "sampling_hash": self.config.sampling_fingerprint,
            "completed_cursor": cursor,
            "completed_requests": completed,
            "pending_queue": pending,
            "startup_decision": "PROGRESS_COMMITTED",
        }
        if self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest["global_version"] = int(previous.get("global_version", 1))
            manifest["startup_decision"] = previous.get("startup_decision", "FRESH_START")
            manifest["resume_count"] = previous.get("resume_count", 0)
        if extra_manifest:
            manifest.update(extra_manifest)
        self.commit_json("pending_queue", {"pending_queue": pending, "cursor": cursor})
        self.commit_json("manifest", manifest)
        return manifest


def schedule_requests(
    requests: dict[str, Request],
    config: Config,
    *,
    blocked_request_ids: set[str] | None = None,
) -> dict[str, Any]:
    blocked = blocked_request_ids or set()
    remaining = config.batch_budget
    processed: list[str] = []
    pending: list[str] = []
    quota_exceeded: list[str] = []
    blocked_list: list[str] = []
    failures: list[Failure] = []
    cursor: str | None = None
    stable_ids = sorted(requests)
    budget_open = True

    for request_id in stable_ids:
        if request_id in blocked:
            blocked_list.append(request_id)
            continue
        request = requests[request_id]
        cost = len(request.candidates)
        if cost > config.request_quota:
            quota_exceeded.append(request_id)
            failure = Failure(
                E_REQUEST_QUOTA_EXCEEDED,
                f"request point cost {cost} exceeds quota {config.request_quota}",
                source="scheduler",
                request_id=request_id,
                business_code=E_REQUEST_QUOTA_EXCEEDED,
            )
            failures.append(failure)
            continue
        if not budget_open or cost > remaining:
            budget_open = False
            pending.append(request_id)
            failures.append(
                Failure(
                    E_POINT_BUDGET_PENDING,
                    f"request cost {cost} waits; remaining batch budget {remaining}",
                    source="scheduler",
                    request_id=request_id,
                    business_code=E_POINT_BUDGET_PENDING,
                )
            )
            continue
        processed.append(request_id)
        remaining -= cost
        cursor = request_id

    return {
        "batch_point_budget": config.batch_budget,
        "request_point_quota": config.request_quota,
        "remaining_budget": remaining,
        "processed": processed,
        "pending": pending,
        "quota_exceeded": quota_exceeded,
        "blocked": blocked_list,
        "completed_cursor": cursor,
        "failures": [failure.to_dict() for failure in failures],
        "stable_order": stable_ids,
    }


def run_scheduled_snapshot(
    engine: IncrementalEngine,
    requests: dict[str, Request],
    config: Config,
    store: SnapshotStore,
    *,
    blocked_request_ids: set[str] | None = None,
) -> tuple[EngineRun | None, dict[str, Any], dict[str, Any]]:
    plan = schedule_requests(requests, config, blocked_request_ids=blocked_request_ids)
    manifest = store.begin_run()
    scheduled_requests = {rid: requests[rid] for rid in plan["processed"]}
    run = engine.run(scheduled_requests) if scheduled_requests else None
    if run is not None:
        store.commit_json("normalization_truncation", {
            rid: run.results[rid].to_dict() for rid in run.results
        })
        store.commit_json("sampling", {
            rid: {
                "sampled_ids": run.results[rid].sampled_ids,
                "unsampled_topk_ids": run.results[rid].unsampled_topk_ids,
                "shuffle_order": run.results[rid].shuffle_order,
                "layers": run.results[rid].layers,
            }
            for rid in run.results
        })
        store.commit_json("serialization", {
            "records": run.records,
            "v1_lines": run.v1_lines,
        })
        store.commit_json("reuse_ledger", run.ledger)
    progress_manifest = store.commit_progress(
        completed=plan["processed"],
        pending=plan["pending"],
        cursor=plan["completed_cursor"],
        extra_manifest={"startup_decision": manifest["startup_decision"]},
    )
    return run, plan, progress_manifest
