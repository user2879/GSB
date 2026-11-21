"""Incremental deterministic execution engine for activation/permutation chains."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import Artifact, make_cache_key
from .cache import ArtifactCache
from .chain import Chain, Node, fusion_candidates, parse_chain
from .config import ExecutionConfig, parse_config
from .errors import (
    E_ACTIVATION_FAILED,
    E_CHAIN_INVALID,
    E_CYCLE_DETECTED,
    E_INPLACE_CONFLICT,
    E_PERM_INVALID,
    E_PERMUTE_FAILED,
    E_RECOMPUTE_BUDGET_PENDING,
    E_RUN_INTERRUPTED,
    E_SNAPSHOT_IO_ERROR,
    E_SNAPSHOT_STALE,
    E_UPSTREAM_FAILED,
    BusinessError,
)
from .input_spec import InputSpec, generate_input, parse_input_spec
from .numeric import (
    ACTIVATION_IMPL_VERSION,
    FUSION_IMPL_VERSION,
    PERMUTE_IMPL_VERSION,
    activation_approximation_metadata,
    activate,
    append_element_order,
    fused_boundary,
    permute_element_metadata,
    permute_values,
    unfused_boundary,
)
from .snapshot import SnapshotManager
from .util import (
    atomic_write_json,
    canonical_json,
    ensure_private_workspace,
    read_json,
    sha256_json,
    tensor_fingerprint,
    to_jsonable,
    workspace_lock,
)


def _quantization_labels(config: ExecutionConfig) -> tuple[list[str], int]:
    precision = config.precision
    labels: list[str] = []
    if config.quantization.scale is not None:
        labels.append(
            f"QUANT:fixed:{config.quantization_tier}:scale={config.quantization.scale:g}@q-v1"
        )
    labels.append(
        f"ROUND:decimal:{precision.tier}:places={precision.decimal_places}@round-v1"
    )
    return labels, len(labels)


def _input_element_metadata(spec: InputSpec, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        order[index] = [f"INPUT:{spec.name}@input-v1"]
    return order, np.zeros(values.shape, dtype=np.int64)


def _metadata_arrays(metadata: dict[str, Any], shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    order = np.empty(shape, dtype=object)
    raw_order = metadata.get("element_order")
    raw_rounds = metadata.get("round_count")
    if raw_order is None or raw_rounds is None:
        for index in np.ndindex(shape):
            order[index] = []
        return order, np.zeros(shape, dtype=np.int64)
    for index in np.ndindex(shape):
        value: Any = raw_order
        for position in index:
            value = value[position]
        order[index] = list(value)
    return order, np.asarray(raw_rounds, dtype=np.int64)


def _shape_list(shape: tuple[int, ...]) -> list[int]:
    return [int(dim) for dim in shape]


def _sequence_moved_ids(old_order: list[str], new_order: list[str]) -> set[str]:
    """Return IDs outside one longest-common-subsequence of the node order."""

    common = set(old_order) & set(new_order)
    left = [node_id for node_id in old_order if node_id in common]
    right = [node_id for node_id in new_order if node_id in common]
    rows, cols = len(left), len(right)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            if left[row - 1] == right[col - 1]:
                dp[row][col] = dp[row - 1][col - 1] + 1
            else:
                dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])
    kept: set[str] = set()
    row, col = rows, cols
    while row > 0 and col > 0:
        if left[row - 1] == right[col - 1]:
            kept.add(left[row - 1])
            row -= 1
            col -= 1
        elif dp[row - 1][col] >= dp[row][col - 1]:
            row -= 1
        else:
            col -= 1
    return common - kept


class OperatorChainEngine:
    """A workspace-backed incremental executor.

    The engine reads the supplied chain/input/config on every call. The workspace keeps
    content-addressed artifacts, the previous logical state, and atomic snapshots.
    """

    STATE_FILE = "state.json"
    LOCK_FILE = "workspace.lock"

    def __init__(
        self,
        chain: dict[str, Any] | Chain,
        input_spec: dict[str, Any] | InputSpec,
        config: dict[str, Any] | ExecutionConfig,
        workspace: str | os.PathLike[str],
    ):
        self.raw_chain = chain.raw if isinstance(chain, Chain) else dict(chain)
        self.raw_input = input_spec.raw if isinstance(input_spec, InputSpec) else dict(input_spec)
        self.raw_config = config.raw if isinstance(config, ExecutionConfig) else dict(config)
        if isinstance(config, ExecutionConfig) and self.raw_config is None:
            self.raw_config = config.to_json()
        self.config = parse_config(config)
        self.input_spec = parse_input_spec(input_spec)
        self.workspace = ensure_private_workspace(workspace)
        self.cache = ArtifactCache(self.workspace)
        self.snapshots = SnapshotManager(self.workspace)
        self.chain: Chain | None = None
        self.input_values = generate_input(self.input_spec)
        self.input_hash = tensor_fingerprint(self.input_values)

    @classmethod
    def from_files(
        cls,
        chain_path: str | os.PathLike[str],
        input_path: str | os.PathLike[str],
        config_path: str | os.PathLike[str],
        workspace: str | os.PathLike[str],
    ) -> "OperatorChainEngine":
        return cls(read_json(chain_path), read_json(input_path), read_json(config_path), workspace)

    @property
    def state_path(self) -> Path:
        return self.workspace / self.STATE_FILE

    def _read_state(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        try:
            data = read_json(self.state_path)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def _state_signature(self) -> str:
        return sha256_json(
            {
                "chain": self.raw_chain,
                "input": self.input_spec.canonical(),
                "config": self.config.to_json(),
            }
        )

    def _previous_chain_view(self, state: dict[str, Any] | None) -> dict[str, Any]:
        if not state:
            return {}
        view = state.get("chain_view")
        return view if isinstance(view, dict) else {}

    def _compute_dirty_set(self, state: dict[str, Any] | None, chain: Chain, fusion_pairs: dict[str, str]) -> set[str]:
        all_nodes = set(chain.topo_order)
        if state is None:
            return set(all_nodes)

        dirty: set[str] = set()
        old_chain_view = self._previous_chain_view(state)
        old_descriptors = old_chain_view.get("node_descriptors", {})
        if not isinstance(old_descriptors, dict):
            old_descriptors = {}
        current_descriptors = chain.node_descriptor_map()
        changed = {
            node_id
            for node_id, descriptor in current_descriptors.items()
            if old_descriptors.get(node_id) != descriptor
        }
        added = set(current_descriptors) - set(old_descriptors)
        removed = set(old_descriptors) - set(current_descriptors)
        old_order = old_chain_view.get("node_order", [])
        new_order = [node.id for node in chain.nodes]
        if not isinstance(old_order, list):
            old_order = []
        moved = _sequence_moved_ids([str(item) for item in old_order], new_order)
        seeds = changed | added | removed | moved
        dirty |= seeds & all_nodes
        dirty |= chain.downstream(seeds)

        old_consumers = old_chain_view.get("consumers", {})
        if isinstance(old_consumers, dict):
            for removed_id in removed:
                for child in old_consumers.get(removed_id, []):
                    if child in all_nodes:
                        dirty.add(child)
                        dirty |= chain.downstream([child])

        old_input = state.get("input_spec") if isinstance(state.get("input_spec"), dict) else None
        if old_input != self.input_spec.canonical():
            dirty = set(all_nodes)

        old_config = state.get("config") if isinstance(state.get("config"), dict) else {}
        old_precision = old_config.get("precision_tier", "fp64")
        old_quantization = old_config.get("quantization_tier", "none")
        if old_precision != self.config.precision_tier or old_quantization != self.config.quantization_tier:
            dirty = set(all_nodes)

        old_fusion = state.get("fusion_pairs", {})
        if not isinstance(old_fusion, dict):
            old_fusion = {}
        if old_fusion != fusion_pairs:
            fusion_nodes = set(old_fusion) | set(old_fusion.values()) | set(fusion_pairs) | set(fusion_pairs.values())
            fusion_nodes &= all_nodes
            dirty |= fusion_nodes
            dirty |= chain.downstream(fusion_nodes)

        for pending_id in state.get("pending_nodes", []) or []:
            if pending_id in all_nodes:
                dirty.add(pending_id)
                dirty |= chain.downstream([pending_id])
        return dirty

    def _activation_failure_record(self, node: Node, exc: BaseException) -> dict[str, Any]:
        if isinstance(exc, BusinessError):
            return exc.as_dict()
        return BusinessError(
            E_ACTIVATION_FAILED,
            f"{type(exc).__name__}: {exc}",
            stage="activation",
            node_id=node.id,
            details={"exception_type": type(exc).__name__},
        ).as_dict()

    def _permute_failure_record(self, node: Node, exc: BaseException) -> dict[str, Any]:
        if isinstance(exc, BusinessError):
            return exc.as_dict()
        return BusinessError(
            E_PERMUTE_FAILED,
            f"{type(exc).__name__}: {exc}",
            stage="permutation",
            node_id=node.id,
            details={"exception_type": type(exc).__name__},
        ).as_dict()

    def _inplace_conflicts(self, chain: Chain, node: Node) -> list[list[str]]:
        if not node.inplace:
            return []
        source = node.inputs[0]
        peers = [candidate for candidate in chain.consumers.get(source, []) if candidate != node.id]
        pairs: list[list[str]] = []
        for peer in peers:
            pair = sorted([node.id, peer])
            if pair not in pairs:
                pairs.append(pair)
        if source in chain.outputs and source != node.id:
            pair = sorted([node.id, f"output:{source}"])
            if pair not in pairs:
                pairs.append(pair)
        return sorted(pairs)

    def _expected_cache_key(
        self,
        node: Node,
        input_fingerprint: str,
        chain: Chain,
        fusion_map: dict[str, str],
        artifacts: dict[str, Artifact],
    ) -> tuple[str, dict[str, Any] | None]:
        fusion_descriptor: dict[str, Any] | None = None
        if node.is_permute and self.config.fusion_enabled:
            activation_id = next((act for act, perm in fusion_map.items() if perm == node.id), None)
            if activation_id is not None:
                activation = chain.by_id[activation_id]
                activation_artifact = artifacts.get(activation_id)
                activation_key = activation_artifact.cache_key if activation_artifact is not None else "pending"
                raw_hash = activation_artifact.raw_values_hash if activation_artifact is not None else None
                fusion_descriptor = {
                    "strategy": "auto",
                    "activation_node": activation.id,
                    "activation_op": activation.op,
                    "activation_params": activation.normalized_params,
                    "activation_cache_key": activation_key,
                    "activation_raw_fingerprint": raw_hash,
                    "implementation_version": FUSION_IMPL_VERSION,
                    "rounding_order": ["decimal", "fixed"] if self.config.quantization.scale is not None else ["decimal"],
                }
        return make_cache_key(node, [input_fingerprint], self.config, fusion_descriptor=fusion_descriptor), fusion_descriptor

    def _execute_node(
        self,
        node: Node,
        chain: Chain,
        artifacts: dict[str, Artifact],
        input_values: np.ndarray,
        input_fingerprint: str,
        input_order: np.ndarray,
        input_rounds: np.ndarray,
        cache_key: str,
        fusion_descriptor: dict[str, Any] | None,
    ) -> Artifact:
        precision = self.config.precision
        quant_labels, quant_rounds = _quantization_labels(self.config)
        if node.is_activation:
            raw = activate(input_values, node.op, node.normalized_params, self.config)
            boundary = unfused_boundary(raw, self.config)
            labels = [f"ACT:{node.op}@{ACTIVATION_IMPL_VERSION}"] + quant_labels
            element_order, round_count = append_element_order(
                input_order,
                labels,
                quant_rounds,
                input_rounds,
            )
            metadata: dict[str, Any] = {
                "operator": node.op,
                "params": node.normalized_params,
                "precision": {
                    "tier": precision.tier,
                    "dtype": precision.dtype.str,
                    "decimal_places": precision.decimal_places,
                },
                "quantization": {
                    "tier": self.config.quantization_tier,
                    "scale": self.config.quantization.scale,
                },
                "approximation": activation_approximation_metadata(node.op, node.normalized_params),
                "implementation_version": ACTIVATION_IMPL_VERSION,
                "rounding_order": list(boundary.rounding_order),
                "rounding_mode": "round-half-to-even",
                "element_order": to_jsonable(element_order),
                "round_count": to_jsonable(round_count),
                "input_fingerprints": [input_fingerprint],
                "inplace": {"requested": node.inplace, "writeback_applied": node.inplace},
                "fusion": {"fused": False},
            }
            return Artifact(
                node_id=node.id,
                op=node.op,
                values=boundary.values,
                cache_key=cache_key,
                metadata=metadata,
                raw_values=np.ascontiguousarray(raw),
            )

        perm = tuple(node.normalized_params["perm"])
        if fusion_descriptor is not None:
            activation_id = str(fusion_descriptor["activation_node"])
            activation_artifact = artifacts.get(activation_id)
            if activation_artifact is None or activation_artifact.raw_values is None:
                raise BusinessError(
                    E_UPSTREAM_FAILED,
                    "fused permutation requires the activation raw sidecar",
                    stage="fusion",
                    node_id=node.id,
                )
            activation = chain.by_id[activation_id]
            activation_source = activation.inputs[0]
            if activation_source in artifacts:
                source_artifact = artifacts[activation_source]
                source_order, source_rounds = _metadata_arrays(source_artifact.metadata, source_artifact.values.shape)
            elif activation_source == self.input_spec.name:
                source_order, source_rounds = _input_element_metadata(self.input_spec, self.input_values)
            else:
                source_order = input_order
                source_rounds = input_rounds
            reordered_order, reordered_rounds = permute_element_metadata(source_order, source_rounds, perm)
            raw_reordered = permute_values(activation_artifact.raw_values, perm)
            boundary = fused_boundary(raw_reordered, self.config)
            fused_labels = [
                f"FUSED_ACT_PERM:{activation.id}+{node.id}:{activation.op}:{perm}@{FUSION_IMPL_VERSION}"
            ] + quant_labels
            element_order, round_count = append_element_order(
                reordered_order,
                fused_labels,
                quant_rounds,
                reordered_rounds,
            )
            metadata = {
                "operator": node.op,
                "params": node.normalized_params,
                "precision": {
                    "tier": precision.tier,
                    "dtype": precision.dtype.str,
                    "decimal_places": precision.decimal_places,
                },
                "quantization": {
                    "tier": self.config.quantization_tier,
                    "scale": self.config.quantization.scale,
                },
                "approximation": activation_approximation_metadata(activation.op, activation.normalized_params),
                "implementation_version": FUSION_IMPL_VERSION,
                "traversal": "output_index_lexicographic",
                "perm": list(perm),
                "rounding_order": list(boundary.rounding_order),
                "rounding_mode": "round-half-to-even",
                "element_order": to_jsonable(element_order),
                "round_count": to_jsonable(round_count),
                "input_fingerprints": [input_fingerprint],
                "inplace": {"requested": node.inplace, "writeback_applied": node.inplace},
                "fusion": {
                    "fused": True,
                    "activation_node": activation.id,
                    "activation_op": activation.op,
                    "activation_params": activation.normalized_params,
                    "input_source": "activation_raw",
                    "rounding_order": list(boundary.rounding_order),
                    "unfused_rounding_order": ["fixed", "decimal"] if self.config.quantization.scale is not None else ["decimal"],
                    "round_count_delta": quant_rounds,
                    "unfused_total_boundary_rounds": quant_rounds * 2,
                },
            }
            return Artifact(node.id, node.op, boundary.values, cache_key, metadata)

        reordered_order, reordered_rounds = permute_element_metadata(input_order, input_rounds, perm)
        reordered = permute_values(input_values, perm)
        boundary = unfused_boundary(reordered, self.config)
        labels = [f"PERM:{node.id}:{perm}@{PERMUTE_IMPL_VERSION}"] + quant_labels
        element_order, round_count = append_element_order(
            reordered_order,
            labels,
            quant_rounds,
            reordered_rounds,
        )
        metadata = {
            "operator": node.op,
            "params": node.normalized_params,
            "precision": {
                "tier": precision.tier,
                "dtype": precision.dtype.str,
                "decimal_places": precision.decimal_places,
            },
            "quantization": {
                "tier": self.config.quantization_tier,
                "scale": self.config.quantization.scale,
            },
            "implementation_version": PERMUTE_IMPL_VERSION,
            "traversal": "output_index_lexicographic",
            "perm": list(perm),
            "rounding_order": list(boundary.rounding_order),
            "rounding_mode": "round-half-to-even",
            "element_order": to_jsonable(element_order),
            "round_count": to_jsonable(round_count),
            "input_fingerprints": [input_fingerprint],
            "inplace": {"requested": node.inplace, "writeback_applied": node.inplace},
            "fusion": {"fused": False},
        }
        return Artifact(node.id, node.op, boundary.values, cache_key, metadata)

    def _input_context(
        self,
        node: Node,
        chain: Chain,
        artifacts: dict[str, Artifact],
        base_input_order: np.ndarray,
        base_input_rounds: np.ndarray,
    ) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
        source = node.inputs[0]
        if source in artifacts:
            artifact = artifacts[source]
            order, rounds = _metadata_arrays(artifact.metadata, artifact.values.shape)
            return artifact.values, artifact.values_hash, order, rounds
        if source == self.input_spec.name:
            return self.input_values, self.input_hash, base_input_order, base_input_rounds
        raise BusinessError(
            E_CHAIN_INVALID,
            f"missing input tensor: {source}",
            stage="execution",
            node_id=node.id,
        )

    def _save_state(self, record: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, record)

    def _state_record(
        self,
        *,
        global_version: int,
        signature: str,
        chain: Chain,
        fusion_pairs: dict[str, str],
        pending_nodes: list[str],
        status: str,
    ) -> dict[str, Any]:
        return {
            "global_version": global_version,
            "signature": signature,
            "chain": self.raw_chain,
            "input_spec": self.input_spec.canonical(),
            "config": self.config.to_json(),
            "chain_view": {
                "node_descriptors": chain.node_descriptor_map(),
                "node_order": [node.id for node in chain.nodes],
                "topo_order": list(chain.topo_order),
                "consumers": chain.consumers,
            },
            "fusion_pairs": fusion_pairs,
            "pending_nodes": sorted(pending_nodes, key=lambda node_id: (chain.topo_rank.get(node_id, 10**9), node_id)),
            "last_run_status": status,
        }

    def _empty_result(
        self,
        *,
        global_version: int,
        status: str,
        failures: list[dict[str, Any]],
        diagnostics: list[dict[str, Any]] | None = None,
        strict_resume: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "global_version": global_version,
            "config_hash": self.config.config_hash(),
            "chain_hash": None,
            "input_hash": self.input_hash,
            "precision": self.config.precision_tier,
            "quantization": self.config.quantization_tier,
            "tolerance": self.config.tolerance,
            "fusion": {"enabled": self.config.fusion_enabled, "pairs": {}},
            "budget": {"limit": self.config.recompute_budget, "consumed": 0, "remaining": self.config.recompute_budget},
            "invalidated": [],
            "reused": [],
            "recomputed": [],
            "pending": [],
            "failures": failures,
            "diagnostics": diagnostics or [],
            "nodes": [],
            "snapshot": {},
            "strict_resume": strict_resume,
        }

    def run(
        self,
        *,
        resume: bool = True,
        strict_resume: bool = False,
        stop_after: int | None = None,
        include_values: bool = True,
    ) -> dict[str, Any]:
        with workspace_lock(self.workspace / self.LOCK_FILE):
            return self._run_locked(
                resume=resume,
                strict_resume=strict_resume,
                stop_after=stop_after,
                include_values=include_values,
            )

    def _run_locked(
        self,
        *,
        resume: bool,
        strict_resume: bool,
        stop_after: int | None,
        include_values: bool,
    ) -> dict[str, Any]:
        diagnostics: list[dict[str, Any]] = []
        try:
            chain = parse_chain(self.raw_chain, self.input_spec)
        except BusinessError as exc:
            code = exc.code
            if code not in {E_CHAIN_INVALID, E_CYCLE_DETECTED, E_PERM_INVALID}:
                code = E_CHAIN_INVALID
            failure = exc.as_dict() | {"code": code}
            return self._empty_result(global_version=0, status="failed", failures=[failure], diagnostics=diagnostics)
        self.chain = chain
        fusion_pairs = fusion_candidates(chain, self.config.fusion_enabled)

        state = self._read_state()
        signature = self._state_signature()
        if state is not None and state.get("signature") == signature:
            global_version = int(state.get("global_version", 1))
        elif state is not None:
            global_version = int(state.get("global_version", 1)) + 1
        else:
            global_version = 1

        resume_info = self.snapshots.check_resume(
            global_version=global_version,
            config_hash=self.config.config_hash(),
        ) if resume else {"status": "disabled", "resumable": False, "manifest": None}

        if resume and resume_info["status"] == "stale":
            diagnostic = {
                "code": E_SNAPSHOT_STALE,
                "stage": "snapshot",
                "message": "snapshot global version or config hash does not match the requested run",
                "details": {
                    "expected_global_version": resume_info.get("expected_global_version"),
                    "actual_global_version": resume_info.get("actual_global_version"),
                    "expected_config_hash": resume_info.get("expected_config_hash"),
                    "actual_config_hash": resume_info.get("actual_config_hash"),
                },
            }
            if strict_resume:
                return self._empty_result(
                    global_version=global_version,
                    status="failed",
                    failures=[diagnostic | {"message": diagnostic["message"]}],
                    diagnostics=[diagnostic],
                    strict_resume=True,
                )
            diagnostics.append(diagnostic)
        elif resume and resume_info["status"] == "resumable":
            diagnostics.append({
                "code": "SNAPSHOT_RESUMED",
                "stage": "snapshot",
                "message": "resuming from a matching partial snapshot",
            })

        dirty = self._compute_dirty_set(state, chain, fusion_pairs)

        # Detect a newly introduced in-place conflict even if the in-place node's own
        # descriptor did not change (for example, a sibling consumer was added).
        for node in chain.nodes:
            if node.inplace and self._inplace_conflicts(chain, node):
                dirty.add(node.id)
                dirty |= chain.downstream([node.id])

        pending_seed = set()
        if state is not None:
            pending_seed |= {node_id for node_id in state.get("pending_nodes", []) if node_id in chain.by_id}
        if resume_info.get("resumable") and isinstance(resume_info.get("manifest"), dict):
            pending_seed |= {
                node_id for node_id in resume_info["manifest"].get("pending_nodes", []) if node_id in chain.by_id
            }

        state_record = self._state_record(
            global_version=global_version,
            signature=signature,
            chain=chain,
            fusion_pairs=fusion_pairs,
            pending_nodes=sorted(pending_seed),
            status="running",
        )
        self._save_state(state_record)

        base_input_order, base_input_rounds = _input_element_metadata(self.input_spec, self.input_values)
        artifacts: dict[str, Artifact] = {}
        tensor_registry: dict[str, dict[str, Any]] = {
            self.input_spec.name: {
                "values_hash": self.input_hash,
                "shape": _shape_list(self.input_values.shape),
                "producer": "input",
                "inplace_node": None,
            }
        }
        status: dict[str, str] = {}
        failures: list[dict[str, Any]] = []
        recomputed: set[str] = set()
        reused: set[str] = set()
        pending: set[str] = set()
        budget_consumed = 0
        processed_count = 0
        last_snapshot_count = 0
        stopped = False
        validation_record = {
            "valid": True,
            "chain_hash": chain.signature(),
            "input_hash": self.input_hash,
            "config_hash": self.config.config_hash(),
            "topology": list(chain.topo_order),
            "fusion_pairs": fusion_pairs,
        }

        def artifact_sort_key(node_id: str) -> tuple[int, str]:
            return chain.topo_rank.get(node_id, 10**9), node_id

        def commit_snapshot(final: bool = False) -> None:
            unresolved = [node_id for node_id in chain.topo_order if node_id not in artifacts and status.get(node_id) != "failed"]
            complete = final and not unresolved and not failures
            snapshot_status = "complete" if complete else ("partial" if not final else "partial")
            try:
                self.snapshots.commit(
                    global_version=global_version,
                    config_hash=self.config.config_hash(),
                    chain_hash=chain.signature(),
                    input_hash=self.input_hash,
                    completed_nodes=sorted(artifacts, key=artifact_sort_key),
                    pending_nodes=sorted(set(unresolved) | pending, key=artifact_sort_key),
                    failures=failures,
                    artifacts=[artifacts[node_id] for node_id in sorted(artifacts, key=artifact_sort_key)],
                    status=snapshot_status,
                    validation=validation_record,
                )
            except Exception as exc:  # pragma: no cover - defensive filesystem path
                diagnostics.append(
                    BusinessError(
                        E_SNAPSHOT_IO_ERROR,
                        f"snapshot commit failed: {type(exc).__name__}: {exc}",
                        stage="snapshot",
                    ).as_dict()
                )

        def make_failure(node: Node, record: dict[str, Any]) -> None:
            status[node.id] = "failed"
            failure = dict(record)
            failure.setdefault("node_id", node.id)
            failure["topological_rank"] = chain.topo_rank[node.id]
            failure["stable_sort_key"] = [chain.topo_rank[node.id], node.id]
            failures.append(failure)

        def register_artifact(node: Node, artifact: Artifact) -> None:
            tensor_registry[node.id] = {
                "values_hash": artifact.values_hash,
                "shape": _shape_list(artifact.values.shape),
                "producer": node.id,
                "inplace_node": node.id if node.inplace else None,
            }
            if node.inplace:
                tensor_registry[node.inputs[0]] = {
                    "values_hash": artifact.values_hash,
                    "shape": _shape_list(artifact.values.shape),
                    "producer": node.id,
                    "inplace_node": node.id,
                    "writeback_of": node.inputs[0],
                }

        def make_upstream_failure(node: Node, upstream_id: str) -> None:
            make_failure(
                node,
                BusinessError(
                    E_UPSTREAM_FAILED,
                    f"upstream node {upstream_id!r} failed",
                    stage="execution.dependency",
                    node_id=node.id,
                    details={"upstream_node": upstream_id},
                ).as_dict(),
            )

        def make_pending(node: Node, code: str = E_RECOMPUTE_BUDGET_PENDING) -> None:
            status[node.id] = "pending"
            pending.add(node.id)
            diagnostics.append(
                {
                    "code": code,
                    "stage": "scheduler",
                    "node_id": node.id,
                    "message": (
                        "node retained in the pending queue because the recompute budget is exhausted"
                        if code == E_RECOMPUTE_BUDGET_PENDING
                        else "node retained in the pending queue because the run was interrupted"
                    ),
                    "topological_rank": chain.topo_rank[node.id],
                    "stable_sort_key": [chain.topo_rank[node.id], node.id],
                }
            )

        remaining = set(chain.topo_order)
        worker_count = max(1, min(self.config.thread_pool_size, max(1, len(chain.topo_order))))
        executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
        try:
            while remaining:
                ready: list[Node] = []
                for node_id in chain.topo_order:
                    if node_id not in remaining:
                        continue
                    source = chain.by_id[node_id].inputs[0]
                    if source in chain.by_id and source in remaining:
                        continue
                    ready.append(chain.by_id[node_id])
                if not ready:
                    # All remaining nodes are part of an already validated acyclic graph;
                    # this is only reachable if an upstream was left unresolved.
                    for node_id in sorted(remaining, key=artifact_sort_key):
                        make_pending(chain.by_id[node_id], E_UPSTREAM_FAILED)
                    break

                compute_jobs: list[tuple[Node, str, np.ndarray, str, np.ndarray, np.ndarray, str, dict[str, Any] | None]] = []
                for node in sorted(ready, key=lambda candidate: artifact_sort_key(candidate.id)):
                    upstream = node.inputs[0]
                    if upstream in chain.by_id and status.get(upstream) == "failed":
                        make_upstream_failure(node, upstream)
                        remaining.remove(node.id)
                        processed_count += 1
                        continue
                    if upstream in chain.by_id and status.get(upstream) == "pending":
                        make_pending(node, E_RECOMPUTE_BUDGET_PENDING)
                        remaining.remove(node.id)
                        processed_count += 1
                        continue

                    conflicts = self._inplace_conflicts(chain, node)
                    if conflicts:
                        dirty.add(node.id)
                        dirty |= chain.downstream([node.id])
                        make_failure(
                            node,
                            BusinessError(
                                E_INPLACE_CONFLICT,
                                "in-place writeback would invalidate another tensor consumer",
                                stage="execute.inplace",
                                node_id=node.id,
                                details={
                                    "conflict_pairs": conflicts,
                                    "input_tensor": node.inputs[0],
                                },
                            ).as_dict(),
                        )
                        remaining.remove(node.id)
                        processed_count += 1
                        continue

                    try:
                        input_values, input_fingerprint, input_order, input_rounds = self._input_context(
                            node,
                            chain,
                            artifacts,
                            base_input_order,
                            base_input_rounds,
                        )
                    except BusinessError as exc:
                        make_failure(node, exc.as_dict())
                        remaining.remove(node.id)
                        processed_count += 1
                        continue

                    cache_key, fusion_descriptor = self._expected_cache_key(
                        node,
                        input_fingerprint,
                        chain,
                        fusion_pairs,
                        artifacts,
                    )
                    if node.id not in dirty:
                        loaded = self.cache.load(node.id, cache_key)
                        if loaded is not None:
                            needs_raw = node.id in fusion_pairs and loaded.raw_values is None
                            if not needs_raw:
                                artifacts[node.id] = loaded
                                register_artifact(node, loaded)
                                status[node.id] = "reused"
                                reused.add(node.id)
                                remaining.remove(node.id)
                                processed_count += 1
                                continue
                        dirty.add(node.id)
                        dirty |= chain.downstream([node.id])

                    compute_jobs.append(
                        (
                            node,
                            cache_key,
                            input_values,
                            input_fingerprint,
                            input_order,
                            input_rounds,
                            cache_key,
                            fusion_descriptor,
                        )
                    )

                if stopped:
                    for node_id in sorted(remaining, key=artifact_sort_key):
                        make_pending(chain.by_id[node_id], E_RUN_INTERRUPTED)
                    remaining.clear()
                    break

                priority = {}
                for job in compute_jobs:
                    job_node = job[0]
                    priority[job_node.id] = 0 if job_node.id in pending_seed else 1
                compute_jobs.sort(key=lambda job: (priority[job[0].id], chain.topo_rank[job[0].id], job[0].id))

                scheduled_jobs: list[tuple[Node, str, np.ndarray, str, np.ndarray, np.ndarray, str, dict[str, Any] | None]] = []
                for job in compute_jobs:
                    job_node = job[0]
                    if stop_after is not None and processed_count + len(scheduled_jobs) >= stop_after:
                        make_pending(job_node, E_RUN_INTERRUPTED)
                        remaining.remove(job_node.id)
                        processed_count += 1
                        stopped = True
                        continue
                    if budget_consumed >= self.config.recompute_budget:
                        make_pending(job_node)
                        remaining.remove(job_node.id)
                        processed_count += 1
                        continue
                    budget_consumed += 1
                    scheduled_jobs.append(job)

                futures: dict[Any, tuple[Node, str]] = {}
                if executor is not None and scheduled_jobs:
                    for job in scheduled_jobs:
                        job_node, key, values, fingerprint, order, rounds, _, fusion = job
                        future = executor.submit(
                            self._execute_node,
                            job_node,
                            chain,
                            dict(artifacts),
                            values,
                            fingerprint,
                            order,
                            rounds,
                            key,
                            fusion,
                        )
                        futures[future] = (job_node, key)
                    future_results: list[tuple[Node, Artifact | BaseException, str]] = []
                    for future in as_completed(futures):
                        job_node, key = futures[future]
                        try:
                            future_results.append((job_node, future.result(), key))
                        except BaseException as exc:
                            future_results.append((job_node, exc, key))
                    future_results.sort(key=lambda item: artifact_sort_key(item[0].id))
                else:
                    future_results = []
                    for job in scheduled_jobs:
                        job_node, key, values, fingerprint, order, rounds, _, fusion = job
                        try:
                            artifact = self._execute_node(
                                job_node,
                                chain,
                                artifacts,
                                values,
                                fingerprint,
                                order,
                                rounds,
                                key,
                                fusion,
                            )
                            future_results.append((job_node, artifact, key))
                        except BaseException as exc:
                            future_results.append((job_node, exc, key))

                for job_node, result, key in future_results:
                    if isinstance(result, BaseException):
                        record = (
                            self._activation_failure_record(job_node, result)
                            if job_node.is_activation
                            else self._permute_failure_record(job_node, result)
                        )
                        make_failure(job_node, record)
                    else:
                        artifacts[job_node.id] = result
                        register_artifact(job_node, result)
                        status[job_node.id] = "computed"
                        recomputed.add(job_node.id)
                        try:
                            self.cache.save(result)
                        except Exception as exc:  # pragma: no cover - defensive filesystem path
                            diagnostics.append(
                                BusinessError(
                                    E_SNAPSHOT_IO_ERROR,
                                    f"artifact cache write failed for node {job_node.id}: {type(exc).__name__}: {exc}",
                                    stage="cache",
                                    node_id=job_node.id,
                                ).as_dict()
                            )
                    processed_count += 1
                    remaining.discard(job_node.id)

                if processed_count - last_snapshot_count >= self.config.snapshot_interval:
                    commit_snapshot(final=False)
                    last_snapshot_count = processed_count
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        # Anything left unresolved after a scheduler break remains pending.
        for node_id in sorted(remaining, key=artifact_sort_key):
            make_pending(chain.by_id[node_id], E_RUN_INTERRUPTED)
        pending = {node_id for node_id in chain.topo_order if status.get(node_id) == "pending"}

        failures.sort(
            key=lambda item: (
                chain.topo_rank.get(str(item.get("node_id", "")), 10**9),
                str(item.get("stage", "")),
                str(item.get("code", "")),
                str(item.get("node_id", "")),
            )
        )
        diagnostics.sort(
            key=lambda item: (
                chain.topo_rank.get(str(item.get("node_id", "")), 10**9),
                str(item.get("code", "")),
                str(item.get("node_id", "")),
            )
        )

        invalidated = sorted(dirty, key=artifact_sort_key)
        node_records: list[dict[str, Any]] = []
        for node_id in chain.topo_order:
            artifact = artifacts.get(node_id)
            if artifact is not None:
                node_status = status.get(node_id, "reused")
                record = artifact.as_record(
                    include_values=include_values,
                    status=node_status,
                    topo_rank=chain.topo_rank[node_id],
                )
                record["stable_sort_key"] = [chain.topo_rank[node_id], node_id]
                node_records.append(record)
            else:
                node_records.append(
                    {
                        "id": node_id,
                        "op": chain.by_id[node_id].op,
                        "status": status.get(node_id, "pending"),
                        "topological_rank": chain.topo_rank[node_id],
                        "stable_sort_key": [chain.topo_rank[node_id], node_id],
                        "values": None,
                    }
                )

        pending_ids = sorted(pending, key=artifact_sort_key)
        complete = len(artifacts) == len(chain.topo_order) and not failures
        result_status = "complete" if complete else "partial"
        state_record = self._state_record(
            global_version=global_version,
            signature=signature,
            chain=chain,
            fusion_pairs=fusion_pairs,
            pending_nodes=pending_ids,
            status=result_status,
        )
        self._save_state(state_record)
        commit_snapshot(final=True)
        outcome_manifest = self.snapshots.inspect() or {}

        canonical_material = {
            "artifacts": [
                {
                    "id": record["id"],
                    "op": record["op"],
                    "cache_key": record.get("cache_key"),
                    "values_hash": record.get("values_hash"),
                    "shape": record.get("shape"),
                    "dtype": record.get("dtype"),
                    "metadata": record.get("metadata"),
                }
                for record in node_records
                if record.get("values_hash") is not None
            ],
            "failures": failures,
            "pending": pending_ids,
        }

        return {
            "schema_version": 1,
            "status": result_status,
            "global_version": global_version,
            "config_hash": self.config.config_hash(),
            "chain_hash": chain.signature(),
            "input_hash": self.input_hash,
            "precision": self.config.precision_tier,
            "quantization": self.config.quantization_tier,
            "tolerance": self.config.tolerance,
            "fusion": {"enabled": self.config.fusion_enabled, "pairs": fusion_pairs},
            "budget": {
                "limit": self.config.recompute_budget,
                "consumed": budget_consumed,
                "remaining": self.config.recompute_budget - budget_consumed,
            },
            "invalidated": invalidated,
            "reused": sorted(reused, key=artifact_sort_key),
            "recomputed": sorted(recomputed, key=artifact_sort_key),
            "pending": pending_ids,
            "failures": failures,
            "diagnostics": diagnostics,
            "nodes": node_records,
            "tensor_registry": {key: tensor_registry[key] for key in sorted(tensor_registry)},
            "snapshot": {
                "status": outcome_manifest.get("status"),
                "completed_nodes": outcome_manifest.get("completed_nodes", []),
                "artifact_versions": outcome_manifest.get("artifact_versions", {}),
                "global_version": outcome_manifest.get("global_version"),
                "config_hash": outcome_manifest.get("config_hash"),
            },
            "canonical_signature": sha256_json(canonical_material),
        }


def compare_execution_artifacts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Compare the run-invariant portion of two execution results.

    Run-local invalidation/reuse lists are intentionally excluded: a cold full rebuild
    and an incremental edit cannot have the same cache disposition by definition.
    """

    left_material = {
        "nodes": [
            {
                key: node.get(key)
                for key in ("id", "op", "cache_key", "values_hash", "shape", "dtype", "metadata", "values")
            }
            for node in left.get("nodes", [])
        ],
        "failures": left.get("failures", []),
        "pending": left.get("pending", []),
        "tensor_registry": left.get("tensor_registry", {}),
    }
    right_material = {
        "nodes": [
            {
                key: node.get(key)
                for key in ("id", "op", "cache_key", "values_hash", "shape", "dtype", "metadata", "values")
            }
            for node in right.get("nodes", [])
        ],
        "failures": right.get("failures", []),
        "pending": right.get("pending", []),
        "tensor_registry": right.get("tensor_registry", {}),
    }
    equal = canonical_json(left_material) == canonical_json(right_material)
    return {
        "equal": equal,
        "left_signature": sha256_json(left_material),
        "right_signature": sha256_json(right_material),
    }
