from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import codes
from .analysis import build_error_trace, build_layer_stats
from .attribution import analyze_tier_flips
from .config import config_fingerprint, quant_fingerprint, scheme_fingerprint
from .hashing import fingerprint, task_sort_hash, task_sort_key
from .pipeline import fixed_scores, reference_scores
from .samples import business_code_counts
from .schemes import make_operator

STAGE_FILES = {
    "load": "load.json",
    "position": "position.json",
    "score": "score.json",
    "stats": "stats.json",
    "error": "error.json",
    "task_index": "task_index.json",
    "scheduler": "scheduler.json",
    "ledger": "ledger.json",
    "config": "config.json",
}


def length_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint("lengths:v1", {
        "train_length": config["train_length"],
        "target_length": config["target_length"],
    })


def tolerance_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint("tolerances:v1", config["tolerances"])


def build_task_metadata(sample: dict[str, Any], scheme: dict[str, Any],
                        config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sample_fp = sample["content_fingerprint"]
    scheme_fp = scheme_fingerprint(scheme)
    quant_fp = quant_fingerprint(config)
    lengths_fp = length_fingerprint(config)
    tolerance_fp = tolerance_fingerprint(config)

    position_key = fingerprint("cache:position:v1", {
        "sample": sample_fp,
        "scheme": scheme_fp,
        "quant": quant_fp,
    })
    score_key = fingerprint("cache:score:v1", {
        "position_key": position_key,
        "kernel": "ascending-int-v1",
    })
    stats_key = fingerprint("cache:stats:v1", {
        "score_key": score_key,
        "lengths": lengths_fp,
        "kernel": "layer-stats-v1",
    })
    error_key = fingerprint("cache:error:v1", {
        "position_key": position_key,
        "score_key": score_key,
        "lengths": lengths_fp,
        "tolerances": tolerance_fp,
        "flip_frac_bits": config["frac_bits"],
        "flip_tiers": ("Q8", "Q16", "Q32"),
        "kernel": "trace-flip-v1",
    })
    keys = {
        "position": position_key,
        "score": score_key,
        "stats": stats_key,
        "error": error_key,
    }
    params = {
        "sample_fingerprint": sample_fp,
        "scheme": scheme,
        "quant": {
            "frac_bits": config["frac_bits"],
            "quant_tier": config["quant_tier"],
        },
        "lengths": {
            "train_length": config["train_length"],
            "target_length": config["target_length"],
            "sample_train_length": sample["train_length"],
            "sample_target_length": sample["target_length"],
        },
        "tolerances": config["tolerances"],
    }
    result = {}
    for stage, cache_key in keys.items():
        sort_key = task_sort_key(sample["id"], scheme["name"], stage)
        result[sort_key] = {
            "sort_key": sort_key,
            "sort_hash": task_sort_hash(sort_key),
            "sample_id": sample["id"],
            "scheme": scheme["name"],
            "stage": stage,
            "cache_key": cache_key,
            "inputs": {
                "sample_fingerprint": sample_fp,
                "scheme_fingerprint": scheme_fp,
                "quant_fingerprint": quant_fp,
                "length_fingerprint": lengths_fp,
                "tolerance_fingerprint": tolerance_fp,
                "params": params,
            },
        }
    return result


def compute_stage(stage: str, sample: dict[str, Any], scheme: dict[str, Any],
                  config: dict[str, Any], cached: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if stage == "position":
        operator = make_operator(scheme)
        position = operator.fixed_encoding(
            sample, int(config["frac_bits"]), str(config["quant_tier"]),
        )
        digest = fingerprint("position-digest:v1", {
            "query_pe_int": position["query"]["pe_int"],
            "query_encoded_int": position["query"]["encoded_int"],
            "keys": [
                {
                    "position": key["position"],
                    "pe_int": key["pe_int"],
                    "encoded_int": key["encoded_int"],
                }
                for key in position["keys"]
            ],
        })
        return {"position": position, "position_digest": digest}

    if stage == "score":
        position_payload = cached["position"]
        operator = make_operator(scheme)
        float_encoding = operator.float_encoding(sample)
        score = fixed_scores(position_payload["position"])
        reference = reference_scores(float_encoding)
        return {
            "score": score,
            "reference": reference,
            "float_encoding": float_encoding,
        }

    if stage == "stats":
        position_payload = cached["position"]
        score_payload = cached["score"]
        bundle = {
            "position": position_payload["position"],
            "score": score_payload["score"],
            "reference": score_payload["reference"],
        }
        return {"layers": build_layer_stats(sample, bundle)}

    if stage == "error":
        position_payload = cached["position"]
        score_payload = cached["score"]
        bundle = {
            "position": position_payload["position"],
            "score": score_payload["score"],
            "reference": score_payload["reference"],
            "float_encoding": score_payload["float_encoding"],
        }
        trace = build_error_trace(sample, scheme, config, bundle)
        flips = analyze_tier_flips(sample, scheme, int(config["frac_bits"]))
        return {"trace": trace, "flips": flips}

    raise KeyError(stage)


def compute_sample_scheme(sample: dict[str, Any], scheme: dict[str, Any],
                          config: dict[str, Any], needed: list[str],
                          reused: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cached = dict(reused)
    results: dict[str, dict[str, Any]] = {}
    for stage in ("position", "score", "stats", "error"):
        if stage in needed:
            payload = compute_stage(stage, sample, scheme, config, cached)
            results[stage] = payload
            cached[stage] = payload
    return results


def diff_input_basis(old_inputs: dict[str, Any], new_inputs: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for field in (
        "sample_fingerprint", "scheme_fingerprint", "quant_fingerprint",
        "length_fingerprint", "tolerance_fingerprint",
    ):
        if old_inputs.get(field) != new_inputs.get(field):
            changed.append(field)

    def walk(old: Any, new: Any, prefix: str) -> None:
        if type(old) != type(new):
            changed.append(prefix)
            return
        if isinstance(old, dict):
            for key in sorted(set(old) | set(new)):
                path = f"{prefix}.{key}"
                if key not in old or key not in new:
                    changed.append(path)
                else:
                    walk(old[key], new[key], path)
        elif old != new:
            changed.append(prefix)

    walk(old_inputs.get("params", {}), new_inputs.get("params", {}), "params")
    return sorted(set(changed))


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.snapshots_dir = self.root / "snapshots"
        self.current_file = self.root / "CURRENT.json"
        self.stores: dict[str, dict[str, dict[str, Any]]] = {
            "position": {}, "score": {}, "stats": {}, "error": {},
        }
        self.task_index: dict[str, dict[str, Any]] = {}
        self.ledger_events: list[dict[str, Any]] = []
        self.invalidation_records: list[dict[str, Any]] = []
        self.reuse_records: list[dict[str, Any]] = []
        self.scheduler: dict[str, Any] = {
            "batch_id": 0,
            "pending": [],
            "pending_entries": [],
            "last_batch": None,
        }
        self.global_version = 0
        self.loaded_version = 0
        self.recovery = "fresh_workspace"
        self.last_manifest: dict[str, Any] | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._load_committed()

    def _load_committed(self) -> None:
        if not self.current_file.exists():
            return
        try:
            current = json.loads(self.current_file.read_text(encoding="utf-8"))
            snapshot_dir = Path(current["snapshot_dir"])
            if not snapshot_dir.is_absolute():
                snapshot_dir = self.root / snapshot_dir
            manifest_file = snapshot_dir / "manifest.json"
            if not manifest_file.exists():
                self.recovery = "current_pointer_without_manifest_fresh_rebuild"
                return
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            for stage in self.stores:
                path = snapshot_dir / STAGE_FILES[stage]
                if path.exists():
                    self.stores[stage] = json.loads(path.read_text(encoding="utf-8"))
            index_path = snapshot_dir / STAGE_FILES["task_index"]
            if index_path.exists():
                self.task_index = json.loads(index_path.read_text(encoding="utf-8"))
            ledger_path = snapshot_dir / STAGE_FILES["ledger"]
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                self.ledger_events = ledger.get("events", [])
                self.invalidation_records = ledger.get("invalidations", [])
                self.reuse_records = ledger.get("reuses", [])
            scheduler_path = snapshot_dir / STAGE_FILES["scheduler"]
            if scheduler_path.exists():
                self.scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
            self.global_version = int(manifest["global_version"])
            self.loaded_version = self.global_version
            self.last_manifest = manifest
            self.recovery = "resumed_committed_snapshot"
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            self.recovery = f"corrupt_committed_pointer_fresh_rebuild:{type(exc).__name__}"

    def commit(self, config: dict[str, Any], load_payload: dict[str, Any]) -> dict[str, Any]:
        current_version = 0
        if self.current_file.exists():
            current = json.loads(self.current_file.read_text(encoding="utf-8"))
            current_version = int(current.get("global_version", 0))
        if current_version != self.loaded_version:
            raise RuntimeError(
                f"snapshot version conflict: loaded {self.loaded_version}, current {current_version}"
            )

        new_version = self.global_version + 1
        config_hash = config_fingerprint(config)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"tx-{new_version}-", dir=self.root))
        try:
            _write_json(temp_dir / STAGE_FILES["config"], config)
            _write_json(temp_dir / STAGE_FILES["load"], load_payload)
            for stage, store in self.stores.items():
                _write_json(temp_dir / STAGE_FILES[stage], store)
            _write_json(temp_dir / STAGE_FILES["task_index"], self.task_index)
            _write_json(temp_dir / STAGE_FILES["scheduler"], self.scheduler)
            ledger_payload = {
                "events": self.ledger_events,
                "invalidations": self.invalidation_records,
                "reuses": self.reuse_records,
            }
            _write_json(temp_dir / STAGE_FILES["ledger"], ledger_payload)
            manifest = {
                "global_version": new_version,
                "config_hash": config_hash,
                "stage_files": STAGE_FILES,
                "created_from_recovery": self.recovery,
            }
            _write_json(temp_dir / "manifest.json", manifest)
            final_dir = self.snapshots_dir / f"snapshot-{new_version:06d}"
            os.replace(temp_dir, final_dir)
            pointer = {
                "global_version": new_version,
                "config_hash": config_hash,
                "snapshot_dir": str(Path("snapshots") / final_dir.name),
            }
            pointer_tmp = self.root / "CURRENT.json.tmp"
            _write_json(pointer_tmp, pointer)
            os.replace(pointer_tmp, self.current_file)
        finally:
            if temp_dir.exists():
                # A killed transaction may leave this directory; commit never points at it.
                for child in temp_dir.iterdir():
                    child.unlink(missing_ok=True)
                temp_dir.rmdir()

        self.global_version = new_version
        self.loaded_version = new_version
        self.last_manifest = manifest
        return manifest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def active_schemes(config: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [scheme for scheme in config["schemes"] if scheme.get("enabled", True)],
        key=lambda item: item["name"],
    )
