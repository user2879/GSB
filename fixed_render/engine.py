from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import PngImagePlugin

from .config import RenderConfig
from .errors import BusinessError, CODES, StageFault
from .fixedpoint import QuantSpec, validate_quant_spec
from .hashing import array_fingerprint, cache_key, fingerprint_json
from .inputs import ConditionBundle
from .kernel import (
    NamedVector,
    condition_map,
    decode_tile,
    denoise_step,
    encode_image,
    encode_motion,
    encode_text,
    fuse_vectors,
    initialize_latent,
    kernel_parameter_fingerprint,
    latent_shape,
    tile_latent_range,
)
from .storage import ArtifactStore, LedgerStore, SnapshotStore, atomic_write_json


class RenderEngine:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.workspace / "cache")
        self.snapshots = SnapshotStore(self.workspace)
        self.ledgers = LedgerStore(self.workspace)
        self.frames_dir = self.workspace / "frames"
        self.rows_dir = self.workspace / "row_png"
        self.frames_dir.mkdir(exist_ok=True)
        self.rows_dir.mkdir(exist_ok=True)
        self.content_index_path = self.store.root / "content_index.json"

    def startup_decision(self, config: RenderConfig, bundle: ConditionBundle) -> dict[str, Any]:
        current = self.snapshots.current()
        if current is None:
            return {"decision": "fresh_build", "version": 0}
        same_config = current.get("config_fingerprint") == config.config_fingerprint()
        same_conditions = current.get("condition_fingerprint") == bundle.condition_fingerprint()
        if same_config and same_conditions:
            return {
                "decision": "resume",
                "version": int(current.get("version", 0)),
                "pending": current.get("pending", []),
            }
        return {
            "decision": "replan_with_content_cache",
            "version": int(current.get("version", 0)),
            "same_config": same_config,
                "same_conditions": same_conditions,
            }

    def render(
        self,
        config: RenderConfig,
        bundle: ConditionBundle,
        lease_client: Any | None = None,
        lease_info: dict[str, Any] | None = None,
        injection: dict[str, Any] | None = None,
        ledger_name: str = "latest",
    ) -> dict[str, Any]:
        import fcntl

        lock_path = self.workspace / ".render.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if lease_client is not None and lease_info:
                record = lease_client.renew(
                    lease_info["lease_id"], int(lease_info["generation"])
                )
                lease_info["expires_ms"] = record["expires_ms"]
            return self._render_locked(
                config,
                bundle,
                lease_client=lease_client,
                lease_info=lease_info,
                injection=injection,
                ledger_name=ledger_name,
            )

    def _render_locked(
        self,
        config: RenderConfig,
        bundle: ConditionBundle,
        lease_client: Any | None = None,
        lease_info: dict[str, Any] | None = None,
        injection: dict[str, Any] | None = None,
        ledger_name: str = "latest",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        spec = validate_quant_spec(config.fixed_frac_bits, config.quant_mode)
        plan = self.build_plan(config, bundle)
        current = self.snapshots.current() or {}
        previous_nodes = current.get("nodes", {}) if isinstance(current.get("nodes"), dict) else {}
        nodes: dict[str, dict[str, Any]] = {}
        ledger_entries: list[dict[str, Any]] = []
        lease_trajectory: list[dict[str, Any]] = []
        if lease_info:
            lease_trajectory.append(
                {
                    "at_ms": int(time.time_ns() // 1_000_000),
                    "action": "lease_bound_to_render",
                    "lease_id": lease_info.get("lease_id"),
                    "generation": lease_info.get("generation"),
                    "expires_ms": lease_info.get("expires_ms"),
                }
            )

        for index, node in enumerate(plan):
            node_id = node["id"]
            entry: dict[str, Any] = {
                "order_index": index,
                "node_id": node_id,
                "stage": node["stage"],
                "upstream": node.get("upstream", []),
                "status": "pending",
            }
            started_node = time.perf_counter()
            try:
                self._authorize_lease(lease_client, lease_info, node_id, lease_trajectory)
                if self._injection_hits(injection, node_id, "lease_stale"):
                    self._inject_stale_lease(lease_client, lease_info, injection, lease_trajectory)

                blocked_by = self._blocked_upstream(node, nodes)
                if blocked_by:
                    entry.update(
                        status="blocked",
                        reason=f"upstream failed: {','.join(blocked_by)}",
                        injection_id=(injection or {}).get("id"),
                    )
                    ledger_entries.append(entry)
                    nodes[node_id] = {"id": node_id, "status": "blocked", "upstream": node.get("upstream", [])}
                    continue

                self._force_injection_fault(injection, node_id, before_cache=True)
                identity = self._identity(
                    node, config, bundle, nodes, spec, int((lease_info or {}).get("generation", 0))
                )
                entry.update(
                    content_key=identity["content_key"],
                    cache_key=identity["cache_key"],
                    params_fingerprint=identity["params_fingerprint"],
                    upstream_refs=identity["upstream_refs"],
                )
                previous = previous_nodes.get(node_id)
                basis = self._invalidation_basis(node_id, identity, previous, current.get("version"))

                artifact_key, hit_kind = self._find_artifact(identity["content_key"], identity["cache_key"])
                if artifact_key:
                    metadata = self.store.load_metadata(artifact_key)
                    metadata["status"] = "reused"
                    metadata["artifact_cache_key"] = artifact_key
                    nodes[node_id] = metadata
                    entry.update(
                        status="reused",
                        hit_source=str(self.store.artifact_dir(artifact_key)),
                        hit_kind=hit_kind,
                        artifact_cache_key=artifact_key,
                        content_fingerprint=metadata.get("content_fingerprint"),
                        invalidation_basis="content_key_match" if hit_kind == "exact" else "content_fingerprint_match_after_lease_rebind",
                        duration_ms=(time.perf_counter() - started_node) * 1000,
                    )
                    ledger_entries.append(entry)
                    continue

                arrays, extra = self._execute_node(
                    node, config, bundle, spec, nodes, injection
                )
                content_fp = self._output_fingerprint(arrays, extra)
                metadata = {
                    "id": node_id,
                    "node_id": node_id,
                    "stage": node["stage"],
                    "status": "recomputed",
                    "content_key": identity["content_key"],
                    "cache_key": identity["cache_key"],
                    "lease_generation": int((lease_info or {}).get("generation", 0)),
                    "params_fingerprint": identity["params_fingerprint"],
                    "quant": spec.fingerprint(),
                    "upstream_refs": identity["upstream_refs"],
                    "content_fingerprint": content_fp,
                    "trace": extra,
                    "created_at_ms": int(time.time_ns() // 1_000_000),
                }
                self.store.write(identity["cache_key"], arrays, metadata)
                self._register_content(identity["content_key"], identity["cache_key"], content_fp)
                metadata["artifact_cache_key"] = identity["cache_key"]
                nodes[node_id] = metadata
                entry.update(
                    status="recomputed",
                    content_fingerprint=content_fp,
                    invalidation_basis=basis,
                    artifact_cache_key=identity["cache_key"],
                    trace=extra,
                    duration_ms=(time.perf_counter() - started_node) * 1000,
                )
            except BusinessError as exc:
                fault_injection_id = None
                if self._injection_hits(injection, node_id, exc_code=exc.code):
                    fault_injection_id = (injection or {}).get("id")
                entry.update(
                    status="failed",
                    business_code=exc.code,
                    error=exc.message,
                    error_fields=getattr(exc, "fields", {}),
                    injection_id=fault_injection_id or (injection or {}).get("id")
                    if self._injection_targets(injection, node_id)
                    else None,
                    duration_ms=(time.perf_counter() - started_node) * 1000,
                )
                ledger_entries.append(entry)
                nodes[node_id] = {
                    "id": node_id,
                    "node_id": node_id,
                    "status": "failed",
                    "business_code": exc.code,
                    "error": exc.message,
                    "upstream": node.get("upstream", []),
                }
                continue
            except Exception as exc:
                fault = StageFault(CODES["STAGE_FAILED"], f"unexpected stage failure: {exc}")
                entry.update(
                    status="failed",
                    business_code=fault.code,
                    error=str(exc),
                    injection_id=(injection or {}).get("id")
                    if self._injection_targets(injection, node_id)
                    else None,
                    duration_ms=(time.perf_counter() - started_node) * 1000,
                )
                ledger_entries.append(entry)
                nodes[node_id] = {
                    "id": node_id,
                    "node_id": node_id,
                    "status": "failed",
                    "business_code": fault.code,
                    "error": str(exc),
                    "upstream": node.get("upstream", []),
                }
                continue

            ledger_entries.append(entry)

        frame_results = self._frame_results(plan, nodes)
        stage_order = [node["id"] for node in plan]
        recomputed = [entry["node_id"] for entry in ledger_entries if entry["status"] == "recomputed"]
        reused = [entry["node_id"] for entry in ledger_entries if entry["status"] == "reused"]
        failed = [entry["node_id"] for entry in ledger_entries if entry["status"] == "failed"]
        blocked = [entry["node_id"] for entry in ledger_entries if entry["status"] == "blocked"]
        pending = failed + blocked
        affected_frames = self._affected_frames(current, frame_results)
        completed_frames = sum(1 for frame in frame_results if frame.get("pixel_fingerprint"))
        tile_completion = self._tile_completion(config, nodes)
        elapsed_ms = (time.perf_counter() - started) * 1000
        code_counts: dict[str, int] = {}
        for entry in ledger_entries:
            code = entry.get("business_code")
            if code:
                code_counts[code] = code_counts.get(code, 0) + 1

        version = int(current.get("version", 0)) + 1
        manifest = {
            "ok": not failed,
            "version": version,
            "workspace": str(self.workspace),
            "config_fingerprint": config.config_fingerprint(),
            "parameter_fingerprint": config.parameter_fingerprint(),
            "condition_fingerprint": bundle.condition_fingerprint(),
            "stage_order": stage_order,
            "recompute_order": recomputed,
            "reuse_order": reused,
            "failed_order": failed,
            "blocked_order": blocked,
            "nodes": nodes,
            "frames": frame_results,
            "frame_fingerprints": {
                str(frame["frame_index"]): frame.get("pixel_fingerprint")
                for frame in frame_results
                if frame.get("pixel_fingerprint")
            },
            "affected_frame_range": affected_frames,
            "completed_frames": completed_frames,
            "tile_completion": tile_completion,
            "lease": {
                **(lease_info or {}),
                "trajectory": lease_trajectory,
            },
            "budget": {
                "elapsed_render_ms": elapsed_ms,
                "preview_budget_ms": config.preview_budget_ms,
                "consumed_ms": elapsed_ms,
            },
            "cursor": {
                "last_completed_node": self._last_completed(ledger_entries),
                "completed_count": len(recomputed) + len(reused),
                "total_nodes": len(plan),
            },
            "pending": pending,
            "business_code_counts": code_counts,
            "ledger_name": ledger_name,
        }
        ledger = {
            "ledger_version": 1,
            "name": ledger_name,
            "workspace": str(self.workspace),
            "manifest_version": version,
            "config_fingerprint": manifest["config_fingerprint"],
            "condition_fingerprint": manifest["condition_fingerprint"],
            "stage_order": stage_order,
            "recompute_order": recomputed,
            "reuse_order": reused,
            "failed_order": failed,
            "blocked_order": blocked,
            "entries": ledger_entries,
            "lease_trajectory": lease_trajectory,
            "counts": {
                "recomputed": len(recomputed),
                "reused": len(reused),
                "failed": len(failed),
                "blocked": len(blocked),
            },
            "business_code_counts": code_counts,
            "timings_ms": {
                entry["node_id"]: entry.get("duration_ms")
                for entry in ledger_entries
                if entry.get("duration_ms") is not None
            },
        }
        ledger_path = self.ledgers.save(ledger_name, ledger)
        manifest["ledger_path"] = str(ledger_path)
        self.snapshots.commit(version, manifest)
        atomic_write_json(self.workspace / "latest_manifest.json", manifest)
        return {"manifest": manifest, "ledger": ledger}

    def build_plan(self, config: RenderConfig, bundle: ConditionBundle) -> list[dict[str, Any]]:
        final_step = max(1, int(config.denoise_steps))
        plan: list[dict[str, Any]] = [
            {
                "id": "encode:text",
                "stage": "condition_encoding",
                "kind": "encode_text",
                "upstream": [],
                "input_fingerprints": {
                    "vocab": bundle.vocab.fingerprint,
                    "prompt": fingerprint_json(list(bundle.prompt_tokens)),
                },
            },
            {
                "id": "encode:image",
                "stage": "condition_encoding",
                "kind": "encode_image",
                "upstream": [],
                "input_fingerprints": {"image": bundle.image_fingerprint or "absent"},
            },
            {
                "id": "encode:motion",
                "stage": "condition_encoding",
                "kind": "encode_motion",
                "upstream": [],
                "input_fingerprints": {"motion": bundle.motion_fingerprint},
            },
            {
                "id": "fuse",
                "stage": "condition_fusion",
                "kind": "fuse",
                "upstream": ["encode:text", "encode:image", "encode:motion"],
            },
            {
                "id": "latent:init",
                "stage": "latent_initialization",
                "kind": "latent_init",
                "upstream": ["fuse"],
            },
        ]
        previous = "latent:init"
        for step in range(1, config.denoise_steps + 1):
            node_id = f"denoise:step:{step:04d}"
            plan.append(
                {
                    "id": node_id,
                    "stage": "denoise_step",
                    "kind": "denoise_step",
                    "step": step,
                    "upstream": [previous, "fuse"],
                }
            )
            previous = node_id
        final_denoise = previous

        tile_count = config.frame_height // config.tile_height
        for frame in range(config.num_frames):
            decode_id = f"decode:frame:{frame:02d}"
            plan.append(
                {
                    "id": decode_id,
                    "stage": "frame_decoding",
                    "kind": "decode_frame",
                    "frame": frame,
                    "upstream": [final_denoise],
                }
            )
            tile_ids: list[str] = []
            for tile in range(tile_count):
                tile_id = f"tile:frame:{frame:02d}:row:{tile:02d}"
                tile_ids.append(tile_id)
                plan.append(
                    {
                        "id": tile_id,
                        "stage": "tile_row_decoding",
                        "kind": "decode_tile",
                        "frame": frame,
                        "tile": tile,
                        "upstream": [decode_id, final_denoise],
                    }
                )
            plan.append(
                {
                    "id": f"output:frame:{frame:02d}",
                    "stage": "frame_output",
                    "kind": "output_frame",
                    "frame": frame,
                    "upstream": tile_ids,
                }
            )
        return plan

    def _identity(
        self,
        node: dict[str, Any],
        config: RenderConfig,
        bundle: ConditionBundle,
        nodes: dict[str, dict[str, Any]],
        spec: QuantSpec,
        lease_generation: int,
    ) -> dict[str, Any]:
        upstream_refs: dict[str, Any] = {}
        upstream_content_refs: dict[str, Any] = {}
        for upstream_id in node.get("upstream", []):
            metadata = nodes[upstream_id]
            upstream_refs[upstream_id] = {
                "content_fingerprint": metadata.get("content_fingerprint"),
                "cache_key": metadata.get("cache_key"),
            }
            upstream_content_refs[upstream_id] = {
                "content_fingerprint": metadata.get("content_fingerprint"),
            }
        params = self._stage_params(node, config, bundle)
        params_fp = fingerprint_json(params)
        external = node.get("input_fingerprints", {})
        content_key = cache_key(
            {
                "node": node["id"],
                "external_inputs": external,
                "upstream": upstream_content_refs,
                "params_fp": params_fp,
                "quant": spec.fingerprint() if node["kind"] not in {"decode_frame", "output_frame"} else None,
                "content_key_version": 1,
            }
        )
        cache_key_value = cache_key(
            {
                "content_key": content_key,
                "lease_generation": lease_generation,
            }
        )
        return {
            "upstream_refs": upstream_refs,
            "params": params,
            "params_fingerprint": params_fp,
            "content_key": content_key,
            "cache_key": cache_key_value,
        }

    def _stage_params(self, node: dict[str, Any], config: RenderConfig, bundle: ConditionBundle) -> dict[str, Any]:
        kind = node["kind"]
        if kind == "encode_text":
            return {"text_dim": config.text_dim}
        if kind == "encode_image":
            return {
                "image_dim": config.image_dim,
                "image_grid": config.image_grid,
                "image_width": config.image_width,
                "image_height": config.image_height,
            }
        if kind == "encode_motion":
            return {
                "motion_dim": config.motion_dim,
                "motion_block_h": config.motion_block_h,
                "motion_block_w": config.motion_block_w,
            }
        if kind == "fuse":
            return {"fusion_dim": config.fusion_dim, "fusion_weights": config.fusion_weights}
        if kind == "latent_init":
            return {"kernel": kernel_parameter_fingerprint(config), "quant": None}
        if kind == "denoise_step":
            return {"step": node["step"], "kernel": kernel_parameter_fingerprint(config)}
        if kind == "decode_frame":
            return {
                "frame": node["frame"],
                "seed": config.seed,
                "frame_width": config.frame_width,
                "frame_height": config.frame_height,
                "latent_scale": config.latent_scale,
                "latent_channels": config.latent_channels,
            }
        if kind == "decode_tile":
            return {
                "frame": node["frame"],
                "tile": node["tile"],
                "tile_height": config.tile_height,
                "frame_width": config.frame_width,
                "latent_scale": config.latent_scale,
                "latent_channels": config.latent_channels,
            }
        if kind == "output_frame":
            return {
                "frame": node["frame"],
                "frame_width": config.frame_width,
                "frame_height": config.frame_height,
                "png": {"format": "PNG", "compress_level": 6, "mode": "RGB"},
            }
        return {}

    def _blocked_upstream(
        self, node: dict[str, Any], nodes: dict[str, dict[str, Any]]
    ) -> list[str]:
        blocked: list[str] = []
        for upstream_id in node.get("upstream", []):
            status = nodes.get(upstream_id, {}).get("status")
            if status in {"failed", "blocked"}:
                blocked.append(upstream_id)
        return blocked

    def _authorize_lease(
        self,
        client: Any | None,
        lease_info: dict[str, Any] | None,
        node_id: str,
        trajectory: list[dict[str, Any]],
    ) -> None:
        if not lease_info:
            return
        if client is None:
            return
        result = client.authorize(lease_info["lease_id"], int(lease_info["generation"]))
        trajectory.append(
            {
                "at_ms": int(time.time_ns() // 1_000_000),
                "action": "authorize",
                "node_id": node_id,
                "lease_id": lease_info["lease_id"],
                "generation": lease_info["generation"],
                "expires_ms": result.get("expires_ms") if isinstance(result, dict) else None,
            }
        )

    def _inject_stale_lease(
        self,
        client: Any | None,
        lease_info: dict[str, Any] | None,
        injection: dict[str, Any] | None,
        trajectory: list[dict[str, Any]],
    ) -> None:
        if client is None or lease_info is None:
            raise StageFault(CODES["LEASE_STALE"], "stale lease injection requires loopback client")
        bad_generation = int(lease_info["generation"]) + int((injection or {}).get("generation_offset", 1))
        try:
            client.authorize(lease_info["lease_id"], bad_generation)
        except BusinessError as exc:
            trajectory.append(
                {
                    "at_ms": int(time.time_ns() // 1_000_000),
                    "action": "injected_stale_authorize",
                    "presented_generation": bad_generation,
                    "business_code": exc.code,
                    "injection_id": (injection or {}).get("id"),
                }
            )
            raise

    def _injection_hits(
        self,
        injection: dict[str, Any] | None,
        node_id: str,
        fault: str | None = None,
        exc_code: str | None = None,
    ) -> bool:
        if not injection:
            return False
        if injection.get("target") != node_id:
            return False
        if fault is not None and injection.get("fault") != fault:
            return False
        if exc_code is not None and injection.get("expected_code") != exc_code:
            return False
        return True

    def _injection_targets(self, injection: dict[str, Any] | None, node_id: str) -> bool:
        return bool(injection and injection.get("target") == node_id)

    def _force_injection_fault(
        self, injection: dict[str, Any] | None, node_id: str, before_cache: bool = False
    ) -> bool:
        if not self._injection_targets(injection, node_id):
            return False
        fault = injection.get("fault")
        if before_cache and fault in {"dim_mismatch", "nonfinite_latent", "quant_invalid", "tile_out_of_range"}:
            if fault == "dim_mismatch":
                bad = np.zeros((int(injection.get("actual_dim", 999)),), dtype=np.int64)
                expected = int(injection.get("expected_dim", 24))
                if bad.shape[0] != expected:
                    raise StageFault(
                        CODES["DIM_MISMATCH"],
                        "injected condition vector dimension mismatch",
                        field_name=injection.get("field_name", "condition_vector"),
                        details={
                            "injection_id": injection.get("id"),
                            "actual": int(bad.shape[0]),
                            "configured": expected,
                        },
                    )
            if fault == "quant_invalid":
                validate_quant_spec(int(injection.get("frac_bits", 99)), str(injection.get("mode", "bad_mode")))
            if fault == "tile_out_of_range":
                # Config is supplied by caller through injection details in diagnostic runner;
                # engine-level injection uses an obviously invalid row and generic bound.
                tile_index = int(injection.get("tile_index", 999))
                tile_count = int(injection.get("tile_count", 8))
                if not 0 <= tile_index < tile_count:
                    raise StageFault(
                        CODES["TILE_OUT_OF_RANGE"],
                        f"injected tile row {tile_index} outside 0..{tile_count - 1}",
                        details={"injection_id": injection.get("id")},
                    )
            if fault == "nonfinite_latent":
                shape = tuple(injection.get("shape", [2, 16, 16, 4]))
                poisoned = np.zeros(shape, dtype=np.float64)
                poisoned.flat[0] = np.nan
                from .fixedpoint import assert_finite_numeric

                assert_finite_numeric(poisoned, node_id)
        return False

    def _execute_node(
        self,
        node: dict[str, Any],
        config: RenderConfig,
        bundle: ConditionBundle,
        spec: QuantSpec,
        nodes: dict[str, dict[str, Any]],
        injection: dict[str, Any] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        kind = node["kind"]
        if kind == "encode_text":
            values, trace = encode_text(bundle, config, spec)
            return {"values": values}, trace
        if kind == "encode_image":
            values, trace = encode_image(bundle, config, spec)
            return {"values": values}, trace
        if kind == "encode_motion":
            values, trace = encode_motion(bundle, config, spec)
            return {"values": values}, trace
        if kind == "fuse":
            vectors = [
                NamedVector("text", config.text_dim, self._upstream_array(nodes, "encode:text", "values"), True),
                NamedVector("image", config.image_dim, self._upstream_array(nodes, "encode:image", "values"), bundle.image is not None),
                NamedVector("motion", config.motion_dim, self._upstream_array(nodes, "encode:motion", "values"), True),
            ]
            fused, trace = fuse_vectors(vectors, config, spec)
            mapped = condition_map(fused, config, spec)
            return {"fused": fused, "condition_map": mapped}, trace
        if kind == "latent_init":
            fusion_meta = nodes["fuse"]
            latent, seed = initialize_latent(str(fusion_meta["content_fingerprint"]), config, spec)
            return {"latent": latent}, {"stage": "latent_initialization", "derived_seed": seed}
        if kind == "denoise_step":
            step = int(node["step"])
            previous_id = node["upstream"][0]
            previous = self._upstream_array(nodes, previous_id, "latent")
            if self._injection_targets(injection, node["id"]) and injection.get("fault") == "nonfinite_latent":
                previous = previous.astype(np.float64)
                previous.flat[injection.get("flat_index", 0)] = np.nan
            mapped = self._upstream_array(nodes, "fuse", "condition_map")
            latent, trace = denoise_step(previous, mapped, step, config, spec)
            return {"latent": latent}, trace
        if kind == "decode_frame":
            frame = int(node["frame"])
            seed = derive_seed_safe("frame-decoder", config.seed, frame, config.frame_width, config.frame_height)
            return {"decoder_seed": np.asarray(seed, dtype=np.int64)}, {
                "stage": "frame_decoding",
                "frame": frame,
                "decoder_fingerprint": fingerprint_json({"seed": seed, "frame": frame}),
            }
        if kind == "decode_tile":
            return self._execute_tile(node, config, spec, nodes)
        if kind == "output_frame":
            return self._execute_output(node, config, bundle, nodes)
        raise StageFault(CODES["STAGE_FAILED"], f"unknown node kind {kind}")

    def _execute_tile(
        self,
        node: dict[str, Any],
        config: RenderConfig,
        spec: QuantSpec,
        nodes: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        frame = int(node["frame"])
        tile = int(node["tile"])
        decode_id = f"decode:frame:{frame:02d}"
        decoder_fp = str(nodes[decode_id].get("trace", {}).get("decoder_fingerprint", nodes[decode_id]["content_fingerprint"]))
        final_id = node["upstream"][1]
        full_latent = self._upstream_array(nodes, final_id, "latent")
        _, _, latent_y0, latent_y1 = tile_latent_range(tile, config)
        # Deliberately slice before entering decoder: tile cannot see other rows.
        latent_rows = full_latent[frame, latent_y0:latent_y1, :, :].copy()
        pixels, trace = decode_tile(latent_rows, frame, tile, decoder_fp, config, spec)
        row_path = self.rows_dir / f"frame_{frame:02d}_row_{tile:02d}.png"
        self._write_png_atomic(row_path, pixels, {"node_id": node["id"], "row_segment": trace["pixel_rows"]})
        trace["row_png"] = str(Path("row_png") / row_path.name)
        return {"pixels": pixels}, trace

    def _execute_output(
        self,
        node: dict[str, Any],
        config: RenderConfig,
        bundle: ConditionBundle,
        nodes: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        frame = int(node["frame"])
        tile_count = config.frame_height // config.tile_height
        bands: list[np.ndarray] = []
        tile_fingerprints: list[str] = []
        step_summaries = []
        for tile in range(tile_count):
            tile_id = f"tile:frame:{frame:02d}:row:{tile:02d}"
            pixels = self._upstream_array(nodes, tile_id, "pixels")
            bands.append(pixels)
            tile_fingerprints.append(str(nodes[tile_id]["content_fingerprint"]))
        final_pixels = np.concatenate(bands, axis=0).astype(np.uint8)
        output_path = self.frames_dir / f"frame_{frame:02d}.png"
        step_nodes = [nodes[key] for key in nodes if key.startswith("denoise:step:")]
        for metadata in step_nodes:
            trace = metadata.get("trace", {})
            step_summaries.append(
                {
                    "node_id": metadata.get("node_id"),
                    "content_fingerprint": metadata.get("content_fingerprint"),
                    "integer_fingerprint": trace.get("integer_fingerprint"),
                }
            )
        info = PngImagePlugin.PngInfo()
        sidecar = {
            "frame_index": frame,
            "condition_fingerprint": bundle.condition_fingerprint(),
            "parameter_fingerprint": config.parameter_fingerprint(),
            "steps": step_summaries,
            "tile_fingerprints": tile_fingerprints,
            "pixel_fingerprint": array_fingerprint(final_pixels),
            "png_path": str(Path("frames") / output_path.name),
        }
        info.add_text("fixed_render_condition", sidecar["condition_fingerprint"])
        info.add_text("fixed_render_params", sidecar["parameter_fingerprint"])
        info.add_text("fixed_render_steps", fingerprint_json(step_summaries))
        info.add_text("fixed_render_step_summary", json.dumps(step_summaries, sort_keys=True))
        info.add_text("fixed_render_sidecar", fingerprint_json(sidecar))
        self._write_png_atomic(output_path, final_pixels, pnginfo=info)
        atomic_write_json(self.frames_dir / f"frame_{frame:02d}.json", sidecar)
        return {"frame": final_pixels}, {**sidecar, "png_path": sidecar["png_path"]}

    def _write_png_atomic(
        self, path: Path, pixels: np.ndarray, metadata: dict[str, Any] | None = None, pnginfo: PngImagePlugin.PngInfo | None = None
    ) -> None:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        Image.fromarray(pixels, "RGB").save(
            temporary, format="PNG", compress_level=6, pnginfo=pnginfo
        )
        if metadata:
            atomic_write_json(path.with_suffix(".meta.json"), metadata)
        temporary.replace(path)

    def _upstream_array(
        self, nodes: dict[str, dict[str, Any]], node_id: str, array_name: str
    ) -> np.ndarray:
        metadata = nodes[node_id]
        artifact_key = metadata.get("artifact_cache_key") or metadata["cache_key"]
        arrays = self.store.load_arrays(artifact_key)
        if array_name not in arrays:
            raise StageFault(
                CODES["STAGE_FAILED"],
                f"artifact {node_id} does not contain {array_name}",
            )
        return arrays[array_name]

    def _find_artifact(self, content_key: str, lease_cache_key: str) -> tuple[str | None, str | None]:
        if self.store.exists(lease_cache_key):
            return lease_cache_key, "exact"
        index = self._read_content_index()
        record = index.get(content_key)
        if record and self.store.exists(record["artifact_cache_key"]):
            return record["artifact_cache_key"], "content_rebind"
        return None, None

    def _read_content_index(self) -> dict[str, Any]:
        if not self.content_index_path.exists():
            return {}
        try:
            return json.loads(self.content_index_path.read_text("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _register_content(self, content_key: str, artifact_key: str, content_fp: str) -> None:
        index = self._read_content_index()
        index[content_key] = {
            "artifact_cache_key": artifact_key,
            "content_fingerprint": content_fp,
            "updated_at_ms": int(time.time_ns() // 1_000_000),
        }
        atomic_write_json(self.content_index_path, index)

    def _output_fingerprint(self, arrays: dict[str, np.ndarray], extra: dict[str, Any]) -> str:
        if "values" in arrays:
            return array_fingerprint(arrays["values"])
        if "fused" in arrays:
            return array_fingerprint(arrays["fused"])
        if "latent" in arrays:
            return array_fingerprint(arrays["latent"])
        if "decoder_seed" in arrays:
            return array_fingerprint(arrays["decoder_seed"])
        if "pixels" in arrays:
            return array_fingerprint(arrays["pixels"])
        if "frame" in arrays:
            return array_fingerprint(arrays["frame"])
        return fingerprint_json(extra)

    def _invalidation_basis(
        self,
        node_id: str,
        identity: dict[str, Any],
        previous: dict[str, Any] | None,
        version: Any,
    ) -> dict[str, Any]:
        if not previous:
            return {"reason": "initial_render" if not version else "node_absent_in_previous_version"}
        changes: dict[str, Any] = {}
        if previous.get("content_key") != identity["content_key"]:
            old_upstream = previous.get("upstream_refs", {})
            for upstream_id, ref in identity["upstream_refs"].items():
                old_ref = old_upstream.get(upstream_id, {})
                if old_ref.get("content_fingerprint") != ref["content_fingerprint"]:
                    changes[upstream_id] = {
                        "old": old_ref.get("content_fingerprint"),
                        "new": ref["content_fingerprint"],
                    }
            if previous.get("params_fingerprint") != identity["params_fingerprint"]:
                changes["params_fingerprint"] = {
                    "old": previous.get("params_fingerprint"),
                    "new": identity["params_fingerprint"],
                }
            old_quant = previous.get("quant", {})
            new_quant = identity.get("params", {})
            # Quant is part of content_key; report spec via caller metadata if changed.
            changes["content_key"] = {
                "old": previous.get("content_key"),
                "new": identity["content_key"],
            }
        return changes or {"reason": "cache_miss_without_detected_component_change"}

    def _frame_results(
        self, plan: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        frames = sorted({node["frame"] for node in plan if "frame" in node})
        results: list[dict[str, Any]] = []
        for frame in frames:
            output_id = f"output:frame:{frame:02d}"
            metadata = nodes.get(output_id)
            if metadata and metadata.get("status") == "recomputed":
                trace = metadata.get("trace", {})
                results.append(
                    {
                        "frame_index": frame,
                        "status": "complete",
                        "pixel_fingerprint": trace.get("pixel_fingerprint") or metadata.get("content_fingerprint"),
                        "png_path": trace.get("png_path"),
                    }
                )
            elif metadata and metadata.get("status") == "reused":
                trace = metadata.get("trace", {})
                results.append(
                    {
                        "frame_index": frame,
                        "status": "complete",
                        "pixel_fingerprint": trace.get("pixel_fingerprint") or metadata.get("content_fingerprint"),
                        "png_path": trace.get("png_path"),
                    }
                )
            else:
                results.append({"frame_index": frame, "status": "incomplete", "pixel_fingerprint": None})
        return results

    def _affected_frames(
        self, current: dict[str, Any], frame_results: list[dict[str, Any]]
    ) -> list[int]:
        old = current.get("frame_fingerprints", {}) if current else {}
        changed: list[int] = []
        for frame in frame_results:
            index = str(frame["frame_index"])
            new_fp = frame.get("pixel_fingerprint")
            if new_fp and old.get(index) != new_fp:
                changed.append(int(frame["frame_index"]))
        if not changed:
            return []
        return [min(changed), max(changed)]

    def _tile_completion(self, config: RenderConfig, nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        total = config.frame_height // config.tile_height
        for frame in range(config.num_frames):
            completed: list[int] = []
            for tile in range(total):
                metadata = nodes.get(f"tile:frame:{frame:02d}:row:{tile:02d}")
                if metadata and metadata.get("status") in {"recomputed", "reused"}:
                    completed.append(tile)
            result[f"frame:{frame:02d}"] = {"total": total, "completed": completed}
        return result

    def _last_completed(self, entries: list[dict[str, Any]]) -> str | None:
        for entry in reversed(entries):
            if entry["status"] in {"recomputed", "reused"}:
                return entry["node_id"]
        return None


def derive_seed_safe(*parts: Any) -> int:
    from .hashing import derive_seed

    return derive_seed(*parts)
