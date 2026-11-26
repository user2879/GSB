from __future__ import annotations

from pathlib import Path
from typing import Any

from .codes import E_BUDGET_PENDING, E_INPUT_INVALID, OK
from .config import config_fingerprint, load_config
from .depth import build_depth_map, depth_cache_key, depth_dependencies
from .features import extract_features, feature_cache_key
from .ledger import ContentLedger, SnapshotManager
from .matching import match_cache_key, match_pair
from .scheduler import allocate_point_budget, run_ordered
from .triangulation import point_cache_key, triangulate_match
from .utils import fingerprint, read_json
from .validation import validate_dataset
from .voxels import build_spatial_refs, run_voxel_stage


class Pipeline:
    def __init__(
        self,
        root: str | Path,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        dataset: dict[str, Any] | None = None,
        parallel: bool = True,
        use_snapshot: bool = True,
        command_errors: list[dict[str, str]] | None = None,
    ):
        self.root = Path(root)
        self.config = config if config is not None else load_config(config_path or self.root / "configs" / "reconstruction.json")
        self.dataset = dataset
        self.parallel = parallel
        self.use_snapshot = use_snapshot
        self.command_errors = command_errors or []
        self.ledger = ContentLedger(self.root)
        self.snapshots = SnapshotManager(self.ledger)
        self.workers = int(self.config.get("budgets", {}).get("max_workers", 1))

    def run(self) -> dict[str, Any]:
        product_config = _product_config(self.config)
        config_hash = config_fingerprint(product_config)
        recovery = self.snapshots.recover() if self.use_snapshot else {"decision": "fresh", "code": OK}
        previous_state = recovery.get("state", {}) if recovery.get("decision") == "resume_with_snapshot" else {}

        raw_dataset = self.dataset
        dataset_path = self.root / "samples" / "dataset.json"
        if raw_dataset is None:
            raw_dataset = read_json(dataset_path)
        validated = validate_dataset(raw_dataset, self.root / "samples", self.config)

        input_errors = list(validated.errors) + list(self.command_errors)
        counts = {
            OK: 1,
            E_INPUT_INVALID: len(input_errors),
            E_BUDGET_PENDING: 0,
        }

        state: dict[str, Any] = {
            "schema_version": 1,
            "config": self.config,
            "product_config_hash": config_hash,
            "recovery": {k: v for k, v in recovery.items() if k != "state"},
            "input_errors": input_errors,
            "business_code_counts": counts,
            "views_summary": [],
            "features": {},
            "matches": {},
            "triangulation": {"points": [], "rejected": []},
            "budget": {},
            "depth_maps": {},
            "voxels": {},
            "pending": [],
            "lineage": [],
            "cache_events": [],
        }

        if not validated.views:
            state["fatal"] = "no valid views"
            if self.use_snapshot:
                self.snapshots.commit(state, config_hash)
            return state

        state["views_summary"] = [
            {
                "view_id": view["view_id"],
                "sequence": view["sequence"],
                "image": view.get("image"),
                "image_sha256": validated.image_sha256[view["view_id"]],
            }
            for view in validated.views
        ]

        prev_features = previous_state.get("features", {})
        feature_items: list[tuple[dict[str, Any], dict[str, Any]]] = []

        def feature_job(view: dict[str, Any]):
            view_id = view["view_id"]
            image_sha = validated.image_sha256[view_id]
            key = feature_cache_key(view_id, image_sha, self.config)
            cached = self.ledger.load_artifact(key)
            if cached is not None:
                return cached, _event("hit", "feature", view_id, key, ["content_ledger"])
            artifact = extract_features(view, image_sha, self.config)
            return artifact, _event("rebuilt", "feature", view_id, key, ["cache_key_absent"])

        feature_results = run_ordered(feature_job, validated.views, self.workers, self.parallel)
        for view, (artifact, event) in zip(validated.views, feature_results):
            view_id = view["view_id"]
            key = artifact["cache_key"]
            self.ledger.store_artifact(artifact)
            old = prev_features.get(view_id)
            if old and old.get("cache_key") != key:
                event["invalidation"] = _feature_change_reasons(old, artifact)
                state["cache_events"].append(_event("invalidated", "feature", view_id, old["cache_key"], event["invalidation"]))
            elif not old:
                event["basis"].append("first_snapshot")
            state["cache_events"].append(event)
            self.ledger.append_event(event)
            state["features"][view_id] = artifact
            feature_items.append((artifact, event))

        for old_id in sorted(set(prev_features) - set(state["features"])):
            old_key = prev_features[old_id]["cache_key"]
            event = _event("invalidated", "feature", old_id, old_key, ["view_removed_or_invalid"])
            state["cache_events"].append(event)
            self.ledger.append_event(event)

        match_keys = [pair for pair in validated.pairs]
        prev_matches = previous_state.get("matches", {})

        def match_job(pair: tuple[str, str]):
            left = state["features"][pair[0]]
            right = state["features"][pair[1]]
            key = match_cache_key(pair, left, right, self.config)
            cached = self.ledger.load_artifact(key)
            if cached is not None:
                return cached, _event("hit", "match", f"{pair[0]}|{pair[1]}", key, ["content_ledger"])
            artifact = match_pair(pair, state["features"], self.config)
            return artifact, _event("rebuilt", "match", f"{pair[0]}|{pair[1]}", key, ["cache_key_absent"])

        match_results = run_ordered(match_job, list(match_keys), self.workers, self.parallel)
        matches_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for pair, (artifact, event) in zip(match_keys, match_results):
            unit_id = f"{pair[0]}|{pair[1]}"
            key = artifact["cache_key"]
            self.ledger.store_artifact(artifact)
            old = prev_matches.get(unit_id)
            if old and old.get("cache_key") != key:
                event["invalidation"] = ["upstream_feature_key_or_matching_params_changed"]
                state["cache_events"].append(_event("invalidated", "match", unit_id, old["cache_key"], event["invalidation"]))
            state["cache_events"].append(event)
            self.ledger.append_event(event)
            state["matches"][unit_id] = artifact
            matches_by_pair[pair] = artifact

        match_contexts: list[dict[str, Any]] = []
        for pair in validated.pairs:
            bundle = matches_by_pair[pair]
            for match in sorted(
                bundle["matches"],
                key=lambda m: (m["left_xy"][0], m["left_xy"][1], m["right_xy"][0], m["right_xy"][1]),
            ):
                match_contexts.append(match)

        def point_job(match: dict[str, Any]):
            key = point_cache_key(match, validated.cameras, self.config)
            cached = self.ledger.load_artifact(key)
            if cached is not None:
                return cached, _event("hit", "triangulation_point", match["match_id"], key, ["content_ledger"])
            record = triangulate_match(match, validated.cameras, self.config)
            return record, _event("rebuilt", "triangulation_point", match["match_id"], key, ["cache_key_absent"])

        point_results = run_ordered(point_job, match_contexts, self.workers, self.parallel)
        all_records: list[dict[str, Any]] = []
        for match, (record, event) in zip(match_contexts, point_results):
            self.ledger.store_artifact(record)
            state["cache_events"].append(event)
            self.ledger.append_event(event)
            all_records.append(record)

        all_records.sort(key=lambda r: (tuple(r["pair"]), r["point_id"]))
        accepted_all = [record for record in all_records if record["status"] == "accepted"]
        rejected = [record for record in all_records if record["status"] == "rejected"]
        tri_key = fingerprint(
            {
                "stage": "triangulation",
                "point_keys": [p["cache_key"] for p in accepted_all],
                "rejected_keys": [p["cache_key"] for p in rejected],
                "config": {
                    "min_parallax": self.config["min_parallax"],
                    "frac_bits": self.config["frac_bits"],
                    "quant_level": self.config["quant_level"],
                },
            }
        )
        state["triangulation"] = {
            "stage": "triangulation",
            "cache_key": tri_key,
            "point_count": len(accepted_all),
            "rejected_count": len(rejected),
            "points": accepted_all,
            "rejected": rejected,
        }

        sequence_by_view = {view["view_id"]: int(view["sequence"]) for view in validated.views}
        previous_pending_points = [
            item["unit_id"] for item in previous_state.get("pending", []) if item.get("unit_type") == "triangulation_point"
        ]
        budget = allocate_point_budget(
            accepted_all,
            sequence_by_view,
            self.config,
            previous_pending_ids=previous_pending_points,
        )
        state["budget"] = budget
        allowed = set(budget["allowed_ids"])
        active_points = [point for point in accepted_all if point["point_id"] in allowed]
        pending = list(budget["pending"])

        prev_depths = previous_state.get("depth_maps", {})

        def depth_job(view: dict[str, Any]):
            view_id = view["view_id"]
            camera = validated.cameras[view_id]
            dependencies = depth_dependencies(view_id, active_points, camera, self.config)
            key = depth_cache_key(view_id, camera, dependencies, self.config)
            cached = self.ledger.load_artifact(key)
            if cached is not None:
                return cached, _event("hit", "depth_map", view_id, key, ["content_ledger"])
            artifact = build_depth_map(view_id, view, active_points, validated.cameras, self.config)
            return artifact, _event("rebuilt", "depth_map", view_id, artifact["cache_key"], ["cache_key_absent"])

        depth_results = run_ordered(depth_job, validated.views, self.workers, self.parallel)
        depth_maps: dict[str, dict[str, Any]] = {}
        for view, (artifact, event) in zip(validated.views, depth_results):
            view_id = view["view_id"]
            self.ledger.store_artifact(artifact)
            old = prev_depths.get(view_id)
            if old and old.get("cache_key") != artifact["cache_key"]:
                state["cache_events"].append(
                    _event("invalidated", "depth_map", view_id, old["cache_key"], ["upstream_point_set_or_depth_params_changed"])
                )
            state["cache_events"].append(event)
            self.ledger.append_event(event)
            depth_maps[view_id] = artifact
            state["depth_maps"][view_id] = artifact

        spatial = build_spatial_refs(
            state["triangulation"],
            depth_maps,
            validated.views,
            validated.cameras,
            self.config,
        )
        previous_pending_voxels = [
            item["unit_id"] for item in previous_state.get("pending", []) if item.get("unit_type") == "voxel_subdivision"
        ]
        voxel_stage = run_voxel_stage(
            spatial,
            self.config,
            max_workers=self.workers,
            parallel=self.parallel,
            priority_pending=previous_pending_voxels,
        )

        prev_voxels = previous_state.get("voxels", {}).get("nodes", {})
        for node in voxel_stage["node_list"]:
            key = node["cache_key"]
            cached = self.ledger.load_artifact(key)
            if cached is not None and cached.get("status") != "pending":
                event = _event("hit", "voxel", node["voxel_id"], key, ["content_ledger"])
                voxel_stage["nodes"][node["voxel_id"]] = cached
            else:
                event = _event("rebuilt", "voxel", node["voxel_id"], key, ["cache_key_absent"])
                if node.get("status") != "pending":
                    self.ledger.store_artifact(node)
                elif cached is not None:
                    event["basis"].append("stale_pending_artifact_ignored")
            old = prev_voxels.get(node["voxel_id"])
            if old and old.get("cache_key") != key:
                state["cache_events"].append(
                    _event("invalidated", "voxel", node["voxel_id"], old["cache_key"], ["upstream_refs_parent_or_params_changed"])
                )
            state["cache_events"].append(event)
            self.ledger.append_event(event)

        voxel_stage["node_order"] = [node["voxel_id"] for node in voxel_stage["node_list"]]
        voxel_stage.pop("node_list", None)
        state["voxels"] = voxel_stage
        pending.extend(voxel_stage.get("pending", []))
        pending.sort(key=lambda item: (item.get("unit_type", ""), item.get("unit_id", "")))
        state["pending"] = pending
        counts[E_BUDGET_PENDING] = len(pending)

        state["lineage"] = build_lineage(state)
        state["canonical_ledger"] = [
            {
                "stage": item["stage"],
                "unit_id": item["unit_id"],
                "cache_key": item["cache_key"],
                "upstream_cache_keys": item.get("upstream_cache_keys", []),
            }
            for item in state["lineage"]
        ]

        if self.use_snapshot:
            manifest = self.snapshots.commit(state, config_hash)
            state["committed_manifest"] = manifest
        return state


def _event(decision: str, stage: str, unit_id: str, cache_key: str, basis: list[str]) -> dict[str, Any]:
    return {
        "decision": decision,
        "stage": stage,
        "unit_id": unit_id,
        "cache_key": cache_key,
        "basis": list(basis),
    }


def _feature_change_reasons(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if old.get("image_sha256") != new.get("image_sha256"):
        reasons.append("image_content_fingerprint_changed")
    if old.get("param_fingerprint") != new.get("param_fingerprint"):
        reasons.append("feature_param_or_quant_fingerprint_changed")
    return reasons or ["feature_cache_key_changed"]


def _product_config(config: dict[str, Any]) -> dict[str, Any]:
    copied = dict(config)
    budgets = dict(copied.get("budgets", {}))
    budgets.pop("max_workers", None)
    copied["budgets"] = budgets
    return copied


def build_lineage(state: dict[str, Any]) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    for view_id in sorted(state["features"]):
        artifact = state["features"][view_id]
        lineage.append(
            {
                "stage": "feature",
                "unit_id": view_id,
                "cache_key": artifact["cache_key"],
                "upstream_cache_keys": [f"image:{artifact['image_sha256']}", f"params:{artifact['param_fingerprint']}"],
            }
        )
    for unit_id in sorted(state["matches"]):
        artifact = state["matches"][unit_id]
        lineage.append(
            {
                "stage": "match",
                "unit_id": unit_id,
                "cache_key": artifact["cache_key"],
                "upstream_cache_keys": artifact["upstream_cache_keys"],
            }
        )
    for point in state["triangulation"]["points"] + state["triangulation"]["rejected"]:
        lineage.append(
            {
                "stage": "triangulation_point",
                "unit_id": point["point_id"],
                "cache_key": point["cache_key"],
                "upstream_cache_keys": [point["match_cache_key"]],
            }
        )
    for view_id in sorted(state["depth_maps"]):
        artifact = state["depth_maps"][view_id]
        upstream = [item["cache_key"] for item in artifact["upstream_point_keys"]]
        lineage.append(
            {
                "stage": "depth_map",
                "unit_id": view_id,
                "cache_key": artifact["cache_key"],
                "upstream_cache_keys": upstream,
            }
        )
    voxel_nodes = state["voxels"].get("nodes", {})
    for voxel_id in sorted(voxel_nodes):
        node = voxel_nodes[voxel_id]
        upstream = [item["cache_key"] for item in node["tri_point_refs"]]
        upstream.extend(item["depth_cache_key"] for item in node.get("depth_pixel_links", []))
        if node.get("parent_cache_key"):
            upstream.append(node["parent_cache_key"])
        lineage.append(
            {
                "stage": "voxel",
                "unit_id": voxel_id,
                "cache_key": node["cache_key"],
                "upstream_cache_keys": sorted(set(upstream)),
                "parent_id": node.get("parent_id"),
            }
        )
    lineage.sort(key=lambda item: (item["stage"], item["unit_id"]))
    return lineage


def canonical_pipeline_state(state: dict[str, Any]) -> dict[str, Any]:
    config = dict(state["config"])
    budgets = dict(config.get("budgets", {}))
    budgets.pop("max_workers", None)
    config["budgets"] = budgets
    keys = [
        "schema_version",
        "views_summary",
        "input_errors",
        "business_code_counts",
        "features",
        "matches",
        "triangulation",
        "budget",
        "depth_maps",
        "voxels",
        "pending",
        "canonical_ledger",
    ]
    result = {key: state[key] for key in keys if key in state}
    result["config"] = config
    result["product_config_hash"] = state.get("product_config_hash") or config_fingerprint(_product_config(state["config"]))
    return result


def canonical_fingerprint(state: dict[str, Any]) -> str:
    return fingerprint(canonical_pipeline_state(state))
