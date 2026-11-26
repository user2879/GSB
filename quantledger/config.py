"""Configuration loading, validation and deterministic model state."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import (
    E_BAD_BATCH,
    E_BAD_COMMAND,
    E_BAD_DIMENSION,
    E_BAD_FRAC_BITS,
    E_BAD_GROUP_SIZE,
    E_BAD_LEASE_LIMIT,
    E_BAD_QUANT_LEVEL,
    E_BAD_QUOTA,
    E_BAD_SAMPLE,
    E_CONFIG,
    E_DUP_BATCH,
    E_DUP_PORT,
    E_DUP_WORKER_ID,
    E_GROUP_NOT_DIVISIBLE,
    E_NONFINITE_WEIGHT,
    E_UNKNOWN_LAYER,
    BusinessIssue,
    issue,
)
from .util import digest_array, digest_json, read_json

ALLOWED_QUANT_LEVELS = (4, 8, 16)
ALLOWED_FRAC_BITS = (4, 6, 8, 10, 12)
LOOPBACK_HOST = "127.0.0.1"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def generate_tensor(shape: tuple[int, ...], seed: int, *, scale: float = 1.0) -> np.ndarray:
    """Generate the exact same tensor across Python/NumPy runs."""

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return (rng.random(shape, dtype=np.float64) * 2.0 - 1.0) * float(scale)


@dataclass
class WorkerSpec:
    worker_id: str
    port: int
    concurrency_quota: int
    lease_limit: int
    frac_bits: int
    quant_level: int
    host: str = LOOPBACK_HOST

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "port": self.port,
            "concurrency_quota": self.concurrency_quota,
            "lease_limit": self.lease_limit,
            "frac_bits": self.frac_bits,
            "quant_level": self.quant_level,
        }


@dataclass
class LayerSpec:
    name: str
    rows: int
    cols: int
    group_size: int
    weight: np.ndarray
    weight_seed: int | None = None
    bias: np.ndarray | None = None

    def clone(self) -> "LayerSpec":
        return LayerSpec(
            name=self.name,
            rows=self.rows,
            cols=self.cols,
            group_size=self.group_size,
            weight=self.weight.copy(),
            weight_seed=self.weight_seed,
            bias=None if self.bias is None else self.bias.copy(),
        )

    def content_dict(self, *, include_weight: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "group_size": self.group_size,
            "weight_digest": digest_array(self.weight),
            "bias_digest": "__NONE__" if self.bias is None else digest_array(self.bias),
        }
        if include_weight:
            payload["weight"] = self.weight.astype(float).tolist()
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "group_size": self.group_size,
            "weight": self.weight.astype(float).tolist(),
        }
        if self.weight_seed is not None:
            payload["weight_seed"] = self.weight_seed
        if self.bias is not None:
            payload["bias"] = self.bias.astype(float).tolist()
        return payload


@dataclass
class BatchSpec:
    batch_id: str
    samples: np.ndarray
    sample_seed: int | None = None

    def clone(self) -> "BatchSpec":
        return BatchSpec(self.batch_id, self.samples.copy(), self.sample_seed)

    def digest(self) -> str:
        return digest_array(self.samples)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "batch_id": self.batch_id,
            "shape": list(self.samples.shape),
            "samples": self.samples.astype(float).tolist(),
        }
        if self.sample_seed is not None:
            payload["sample_seed"] = self.sample_seed
        return payload


@dataclass
class RuntimeState:
    workers: list[WorkerSpec]
    layers: list[LayerSpec]
    batches: list[BatchSpec]
    revision: int
    frac_bits: int
    quant_level: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "RuntimeState":
        return RuntimeState(
            workers=[copy.deepcopy(worker) for worker in self.workers],
            layers=[layer.clone() for layer in self.layers],
            batches=[batch.clone() for batch in self.batches],
            revision=self.revision,
            frac_bits=self.frac_bits,
            quant_level=self.quant_level,
            metadata=copy.deepcopy(self.metadata),
        )

    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.layers]

    def find_layer(self, name: str) -> tuple[int, LayerSpec] | None:
        for index, layer in enumerate(self.layers):
            if layer.name == name:
                return index, layer
        return None

    def find_worker(self, worker_id: str) -> WorkerSpec | None:
        for worker in self.workers:
            if worker.worker_id == worker_id:
                return worker
        return None

    def input_width(self) -> int:
        return self.layers[0].cols if self.layers else 0

    def operator_dict(self, *, include_weights: bool = False) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "frac_bits": self.frac_bits,
            "quant_level": self.quant_level,
            "layers": [layer.content_dict(include_weight=include_weights) for layer in self.layers],
        }

    def operator_config_hash(self) -> str:
        return digest_json(self.operator_dict(include_weights=True))

    def runtime_config_hash(self) -> str:
        return digest_json(
            {
                "operator": self.operator_config_hash(),
                "workers": [worker.to_dict() for worker in self.workers],
                "batches": [
                    {"batch_id": batch.batch_id, "digest": batch.digest()} for batch in self.batches
                ],
            }
        )


@dataclass
class LoadedConfig:
    state: RuntimeState
    issues: list[BusinessIssue]
    raw: dict[str, Any]
    source_path: str | None = None
    receipt_paths: list[str] = field(default_factory=list)
    commands: list[dict[str, Any]] = field(default_factory=list)

    def issue_dicts(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self.issues]


def _make_weight(layer_raw: dict[str, Any], rows: int, cols: int, path: str, issues: list[BusinessIssue]) -> tuple[np.ndarray | None, int | None]:
    explicit = layer_raw.get("weight")
    seed = layer_raw.get("weight_seed")
    if explicit is None:
        if not _is_int(seed):
            issues.append(issue(E_CONFIG, str(layer_raw.get("name", path)), path, "weight or integer weight_seed is required"))
            return None, None
        weight = generate_tensor((rows, cols), int(seed))
    else:
        try:
            weight = np.asarray(explicit, dtype=np.float64)
        except (TypeError, ValueError):
            issues.append(issue(E_CONFIG, str(layer_raw.get("name", path)), path + ".weight", "weight must be a numeric matrix"))
            return None, None
        if weight.shape != (rows, cols):
            issues.append(
                issue(
                    E_BAD_DIMENSION,
                    str(layer_raw.get("name", path)),
                    path + ".weight",
                    f"weight shape {list(weight.shape)} does not match {(rows, cols)}",
                )
            )
            return None, None
    if not np.all(np.isfinite(weight)):
        issues.append(issue(E_NONFINITE_WEIGHT, str(layer_raw.get("name", path)), path + ".weight", "weight contains non-finite values"))
        return None, None
    return weight, int(seed) if _is_int(seed) else None


def _make_bias(layer_raw: dict[str, Any], rows: int, path: str, issues: list[BusinessIssue]) -> np.ndarray | None:
    raw = layer_raw.get("bias")
    if raw is None:
        return None
    try:
        bias = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        issues.append(issue(E_CONFIG, str(layer_raw.get("name", path)), path + ".bias", "bias must be numeric"))
        return None
    if bias.shape != (rows,):
        issues.append(issue(E_BAD_DIMENSION, str(layer_raw.get("name", path)), path + ".bias", f"bias shape must be [{rows}]"))
        return None
    if not np.all(np.isfinite(bias)):
        issues.append(issue(E_NONFINITE_WEIGHT, str(layer_raw.get("name", path)), path + ".bias", "bias contains non-finite values"))
        return None
    return bias


def _validate_layers(raw_layers: Any, issues: list[BusinessIssue]) -> list[LayerSpec]:
    if not isinstance(raw_layers, list):
        issues.append(issue(E_CONFIG, "operator.layers", "operator.layers", "layers must be a list"))
        return []
    layers: list[LayerSpec] = []
    names: set[str] = set()
    previous_rows: int | None = None
    for index, raw in enumerate(raw_layers):
        path = f"operator.layers[{index}]"
        if not isinstance(raw, dict):
            issues.append(issue(E_CONFIG, str(index), path, "layer must be an object"))
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            issues.append(issue(E_CONFIG, str(index), path + ".name", "layer name must be a non-empty string"))
            continue
        if name in names:
            issues.append(issue(E_CONFIG, name, path + ".name", "duplicate layer name", skipped=True))
            continue
        rows, cols, group_size = raw.get("rows"), raw.get("cols"), raw.get("group_size")
        if not _positive_int(rows) or not _positive_int(cols):
            issues.append(issue(E_BAD_DIMENSION, name, path, "rows and cols must be positive integers"))
            continue
        if not _positive_int(group_size):
            issues.append(issue(E_BAD_GROUP_SIZE, name, path + ".group_size", "group_size must be a positive integer"))
            continue
        if (rows * cols) % group_size != 0:
            issues.append(
                issue(
                    E_GROUP_NOT_DIVISIBLE,
                    name,
                    path + ".group_size",
                    f"group_size {group_size} does not divide {rows * cols} layer elements",
                )
            )
            continue
        if previous_rows is not None and cols != previous_rows:
            issues.append(
                issue(
                    E_BAD_DIMENSION,
                    name,
                    path + ".cols",
                    f"layer input width {cols} does not match previous output width {previous_rows}",
                )
            )
            continue
        weight, weight_seed = _make_weight(raw, rows, cols, path, issues)
        if weight is None:
            continue
        bias = _make_bias(raw, rows, path, issues)
        if raw.get("bias") is not None and bias is None:
            continue
        layers.append(
            LayerSpec(
                name=name,
                rows=rows,
                cols=cols,
                group_size=group_size,
                weight=weight,
                weight_seed=weight_seed,
                bias=bias,
            )
        )
        names.add(name)
        previous_rows = rows
    return layers


def _make_samples(raw: dict[str, Any], input_width: int, path: str, issues: list[BusinessIssue], item: str) -> tuple[np.ndarray | None, int | None]:
    explicit = raw.get("samples")
    seed = raw.get("sample_seed")
    size = raw.get("batch_size")
    if explicit is not None:
        try:
            samples = np.asarray(explicit, dtype=np.float64)
        except (TypeError, ValueError):
            issues.append(issue(E_BAD_SAMPLE, item, path + ".samples", "samples must be a numeric matrix"))
            return None, None
    else:
        if not _is_int(seed) or not _positive_int(size):
            issues.append(issue(E_BAD_SAMPLE, item, path, "samples or integer sample_seed + positive batch_size are required"))
            return None, None
        samples = generate_tensor((int(size), input_width), int(seed))
    if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] != input_width:
        issues.append(
            issue(
                E_BAD_DIMENSION,
                item,
                path + ".samples",
                f"sample shape {list(samples.shape)} must be [n, {input_width}]",
            )
        )
        return None, None
    if not np.all(np.isfinite(samples)):
        issues.append(issue(E_BAD_SAMPLE, item, path + ".samples", "samples contain non-finite values"))
        return None, None
    return samples, int(seed) if _is_int(seed) else None


def _validate_batches(raw_batches: Any, input_width: int, issues: list[BusinessIssue]) -> list[BatchSpec]:
    if raw_batches is None:
        return []
    if not isinstance(raw_batches, list):
        issues.append(issue(E_CONFIG, "batches", "batches", "batches must be a list"))
        return []
    batches: list[BatchSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_batches):
        path = f"batches[{index}]"
        if not isinstance(raw, dict):
            issues.append(issue(E_BAD_BATCH, str(index), path, "batch must be an object"))
            continue
        batch_id = raw.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            issues.append(issue(E_BAD_BATCH, str(index), path + ".batch_id", "batch_id must be a non-empty string"))
            continue
        if batch_id in seen:
            issues.append(issue(E_DUP_BATCH, batch_id, path, "duplicate batch_id"))
            continue
        samples, seed = _make_samples(raw, input_width, path, issues, batch_id)
        if samples is None:
            continue
        batches.append(BatchSpec(batch_id, samples, seed))
        seen.add(batch_id)
    return batches


def validate_base_config(raw: Any, source_path: str | Path | None = None) -> LoadedConfig:
    issues: list[BusinessIssue] = []
    if not isinstance(raw, dict):
        issues.append(issue(E_CONFIG, "root", "$", "configuration root must be an object"))
        return LoadedConfig(RuntimeState([], [], [], 1, 8, 8), issues, {}, str(source_path) if source_path else None)

    workers: list[WorkerSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    raw_workers = raw.get("workers", [])
    if not isinstance(raw_workers, list):
        issues.append(issue(E_CONFIG, "workers", "workers", "workers must be a list"))
        raw_workers = []
    for index, item in enumerate(raw_workers):
        path = f"workers[{index}]"
        if not isinstance(item, dict):
            issues.append(issue(E_CONFIG, str(index), path, "worker must be an object"))
            continue
        worker_id = item.get("worker_id", item.get("id"))
        if not isinstance(worker_id, str) or not worker_id:
            issues.append(issue(E_CONFIG, str(index), path + ".worker_id", "worker_id must be a non-empty string"))
            continue
        if worker_id in seen_ids:
            issues.append(issue(E_DUP_WORKER_ID, worker_id, path + ".worker_id", "duplicate worker_id"))
            continue
        host = item.get("host", LOOPBACK_HOST)
        if host != LOOPBACK_HOST:
            issues.append(issue(E_CONFIG, worker_id, path + ".host", "workers must bind 127.0.0.1"))
            continue
        port = item.get("port")
        if not _positive_int(port) or int(port) > 65535:
            issues.append(issue(E_CONFIG, worker_id, path + ".port", "port must be an integer in 1..65535"))
            continue
        if int(port) in seen_ports:
            issues.append(issue(E_DUP_PORT, worker_id, path + ".port", f"duplicate loopback port {port}"))
            continue
        quota = item.get("concurrency_quota")
        if not _positive_int(quota):
            issues.append(issue(E_BAD_QUOTA, worker_id, path + ".concurrency_quota", "concurrency_quota must be a positive integer"))
            continue
        lease_limit = item.get("lease_limit")
        if not _positive_int(lease_limit):
            issues.append(issue(E_BAD_LEASE_LIMIT, worker_id, path + ".lease_limit", "lease_limit must be a positive integer"))
            continue
        frac_bits = item.get("frac_bits")
        if frac_bits not in ALLOWED_FRAC_BITS:
            issues.append(issue(E_BAD_FRAC_BITS, worker_id, path + ".frac_bits", f"frac_bits must be one of {list(ALLOWED_FRAC_BITS)}"))
            continue
        quant_level = item.get("quant_level")
        if quant_level not in ALLOWED_QUANT_LEVELS:
            issues.append(issue(E_BAD_QUANT_LEVEL, worker_id, path + ".quant_level", f"quant_level must be one of {list(ALLOWED_QUANT_LEVELS)}"))
            continue
        workers.append(
            WorkerSpec(
                worker_id=worker_id,
                host=host,
                port=int(port),
                concurrency_quota=int(quota),
                lease_limit=int(lease_limit),
                frac_bits=int(frac_bits),
                quant_level=int(quant_level),
            )
        )
        seen_ids.add(worker_id)
        seen_ports.add(int(port))
    if not workers:
        issues.append(issue(E_CONFIG, "workers", "workers", "at least one valid worker is required"))

    operator = raw.get("operator", {})
    if not isinstance(operator, dict):
        issues.append(issue(E_CONFIG, "operator", "operator", "operator must be an object"))
        operator = {}
    revision = operator.get("revision", raw.get("revision", 1))
    if not _positive_int(revision):
        issues.append(issue(E_CONFIG, "operator.revision", "operator.revision", "revision must be a positive integer"))
        revision = 1
    frac_bits = operator.get("frac_bits", raw.get("frac_bits", 8))
    if frac_bits not in ALLOWED_FRAC_BITS:
        issues.append(issue(E_BAD_FRAC_BITS, "operator", "operator.frac_bits", f"frac_bits must be one of {list(ALLOWED_FRAC_BITS)}"))
        frac_bits = 8
    quant_level = operator.get("quant_level", raw.get("quant_level", 8))
    if quant_level not in ALLOWED_QUANT_LEVELS:
        issues.append(issue(E_BAD_QUANT_LEVEL, "operator", "operator.quant_level", f"quant_level must be one of {list(ALLOWED_QUANT_LEVELS)}"))
        quant_level = 8
    layers = _validate_layers(operator.get("layers", []), issues)
    if not layers:
        issues.append(issue(E_CONFIG, "operator.layers", "operator.layers", "at least one valid layer is required"))
    batches = _validate_batches(raw.get("batches", []), layers[0].cols if layers else 0, issues)

    receipt_paths: list[str] = []
    raw_receipts = raw.get("receipts")
    if isinstance(raw_receipts, str):
        receipt_paths = [raw_receipts]
    elif isinstance(raw_receipts, list):
        receipt_paths = [str(item) for item in raw_receipts]
    elif source_path is not None:
        candidate = Path(source_path).parent / "receipts"
        if candidate.is_dir():
            receipt_paths = [str(path) for path in sorted(candidate.glob("*.json*"))]

    state = RuntimeState(
        workers=workers,
        layers=layers,
        batches=batches,
        revision=int(revision),
        frac_bits=int(frac_bits),
        quant_level=int(quant_level),
        metadata={key: value for key, value in raw.items() if key not in {"workers", "operator", "batches", "receipts", "commands"}},
    )
    commands: list[dict[str, Any]] = []
    if raw.get("commands") is not None:
        commands, command_issues = validate_commands(raw.get("commands"), state)
        issues.extend(command_issues)
    return LoadedConfig(
        state=state,
        issues=issues,
        raw=raw,
        source_path=str(source_path) if source_path else None,
        receipt_paths=receipt_paths,
        commands=commands,
    )


def load_base_config(path: str | Path) -> LoadedConfig:
    return validate_base_config(read_json(path), path)


def _append_batch(state: RuntimeState, raw: dict[str, Any], path: str, issues: list[BusinessIssue]) -> BatchSpec | None:
    batch_id = raw.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        issues.append(issue(E_BAD_BATCH, path, path + ".batch_id", "batch_id must be a non-empty string"))
        return None
    if any(batch.batch_id == batch_id for batch in state.batches):
        issues.append(issue(E_DUP_BATCH, batch_id, path, "duplicate batch_id"))
        return None
    samples, seed = _make_samples(raw, state.input_width(), path, issues, batch_id)
    if samples is None:
        return None
    return BatchSpec(batch_id, samples, seed)


def _replace_weight(state: RuntimeState, raw: dict[str, Any], path: str, issues: list[BusinessIssue]) -> LayerSpec | None:
    name = raw.get("layer", raw.get("layer_name"))
    found = state.find_layer(name) if isinstance(name, str) else None
    if found is None:
        issues.append(issue(E_UNKNOWN_LAYER, str(name), path + ".layer", "unknown layer name"))
        return None
    _, layer = found
    explicit = raw.get("weight")
    seed = raw.get("weight_seed")
    if explicit is None:
        if not _is_int(seed):
            issues.append(issue(E_CONFIG, layer.name, path, "weight or integer weight_seed is required"))
            return None
        weight = generate_tensor((layer.rows, layer.cols), int(seed))
    else:
        try:
            weight = np.asarray(explicit, dtype=np.float64)
        except (TypeError, ValueError):
            issues.append(issue(E_CONFIG, layer.name, path + ".weight", "weight must be a numeric matrix"))
            return None
        if weight.shape != (layer.rows, layer.cols):
            issues.append(
                issue(
                    E_BAD_DIMENSION,
                    layer.name,
                    path + ".weight",
                    f"weight shape {list(weight.shape)} does not match {(layer.rows, layer.cols)}",
                )
            )
            return None
    if not np.all(np.isfinite(weight)):
        issues.append(issue(E_NONFINITE_WEIGHT, layer.name, path + ".weight", "weight contains non-finite values"))
        return None
    replacement = layer.clone()
    replacement.weight = weight.astype(np.float64)
    replacement.weight_seed = int(seed) if _is_int(seed) else None
    return replacement


def validate_commands(raw: Any, state: RuntimeState) -> tuple[list[dict[str, Any]], list[BusinessIssue]]:
    """Validate a command stream against a cloned state.

    Returned commands are normalized and accepted in order. Invalid items are
    skipped with a stable business code, so one malformed mutation does not
    poison unrelated work.
    """

    issues: list[BusinessIssue] = []
    if raw is None:
        return [], issues
    if isinstance(raw, dict):
        raw = raw.get("commands", [])
    if not isinstance(raw, list):
        issues.append(issue(E_BAD_COMMAND, "commands", "commands", "command stream must be a list"))
        return [], issues
    work = state.clone()
    accepted: list[dict[str, Any]] = []
    seen_seqs: set[int] = set()
    for index, item in enumerate(raw):
        path = f"commands[{index}]"
        if not isinstance(item, dict):
            issues.append(issue(E_BAD_COMMAND, str(index), path, "command must be an object"))
            continue
        seq = item.get("seq", index + 1)
        if not _is_int(seq) or int(seq) in seen_seqs:
            issues.append(issue(E_BAD_COMMAND, str(seq), path + ".seq", "seq must be a unique integer"))
            continue
        command_type = item.get("type", item.get("op"))
        normalized: dict[str, Any] = {"seq": int(seq), "type": command_type}
        mutation_types = {"append_batch", "replace_weight", "set_weight", "set_quant_level", "set_frac_bits"}
        if command_type in mutation_types and "revision" in item:
            revision_value = item.get("revision")
            if not _positive_int(revision_value):
                issues.append(issue(E_CONFIG, str(revision_value), path + ".revision", "revision must be a positive integer"))
                continue
            work.revision = int(revision_value)
            normalized["revision"] = int(revision_value)
        if command_type == "append_batch":
            batch_payload = item.get("batch", item)
            if not isinstance(batch_payload, dict):
                issues.append(issue(E_BAD_BATCH, str(index), path + ".batch", "batch payload must be an object"))
                continue
            batch = _append_batch(work, batch_payload, path, issues)
            if batch is None:
                continue
            work.batches.append(batch)
            normalized["batch_id"] = batch.batch_id
            normalized["batch"] = batch.to_dict()
        elif command_type in {"replace_weight", "set_weight"}:
            layer = _replace_weight(work, item, path, issues)
            if layer is None:
                continue
            index_in_chain, _ = work.find_layer(layer.name)  # type: ignore[misc]
            work.layers[index_in_chain] = layer
            normalized.update({"type": "replace_weight", "layer": layer.name, "weight_digest": digest_array(layer.weight)})
            if layer.weight_seed is not None:
                normalized["weight_seed"] = layer.weight_seed
            normalized["weight"] = layer.weight.astype(float).tolist()
        elif command_type == "set_quant_level":
            value = item.get("value", item.get("quant_level"))
            if value not in ALLOWED_QUANT_LEVELS:
                issues.append(issue(E_BAD_QUANT_LEVEL, str(value), path + ".value", f"quant_level must be one of {list(ALLOWED_QUANT_LEVELS)}"))
                continue
            work.quant_level = int(value)
            normalized.update({"quant_level": int(value)})
        elif command_type == "set_frac_bits":
            value = item.get("value", item.get("frac_bits"))
            if value not in ALLOWED_FRAC_BITS:
                issues.append(issue(E_BAD_FRAC_BITS, str(value), path + ".value", f"frac_bits must be one of {list(ALLOWED_FRAC_BITS)}"))
                continue
            work.frac_bits = int(value)
            normalized.update({"frac_bits": int(value)})
        elif command_type in {"set_revision", "rollback_revision"}:
            value = item.get("value", item.get("revision"))
            if not _positive_int(value):
                issues.append(issue(E_CONFIG, str(value), path + ".value", "revision must be a positive integer"))
                continue
            work.revision = int(value)
            normalized.update({"revision": int(value)})
        elif command_type == "set_lease_limit":
            worker_id = item.get("worker_id")
            worker = work.find_worker(worker_id) if isinstance(worker_id, str) else None
            if worker is None:
                issues.append(issue(E_UNKNOWN_LAYER, str(worker_id), path + ".worker_id", "unknown worker_id"))
                continue
            value = item.get("value", item.get("lease_limit"))
            if not _positive_int(value):
                issues.append(issue(E_BAD_LEASE_LIMIT, worker.worker_id, path + ".value", "lease_limit must be a positive integer"))
                continue
            worker.lease_limit = int(value)
            normalized.update({"worker_id": worker.worker_id, "lease_limit": int(value)})
        elif command_type == "remove_batch":
            batch_id = item.get("batch_id")
            before = len(work.batches)
            work.batches = [batch for batch in work.batches if batch.batch_id != batch_id]
            if len(work.batches) == before:
                issues.append(issue(E_UNKNOWN_LAYER, str(batch_id), path + ".batch_id", "unknown batch_id"))
                continue
            normalized["batch_id"] = batch_id
        elif command_type == "noop":
            pass
        else:
            issues.append(issue(E_BAD_COMMAND, str(command_type), path + ".type", "unknown command type"))
            continue
        accepted.append(normalized)
        seen_seqs.add(int(seq))
    return accepted, issues


def issues_after_commands(base_issues: Iterable[BusinessIssue], commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item.as_dict() for item in base_issues]


def apply_command(state: RuntimeState, command: dict[str, Any]) -> dict[str, Any]:
    """Apply one normalized/accepted command to a cloned runtime state."""

    command_type = command.get("type")
    if command_type in {"append_batch", "replace_weight", "set_quant_level", "set_frac_bits"} and "revision" in command:
        state.revision = int(command["revision"])
    if command_type == "append_batch":
        payload = command.get("batch")
        if not isinstance(payload, dict):
            raise ValueError("append_batch command lacks replayable batch payload")
        samples, seed = _make_samples(payload, state.input_width(), "command.batch", [], str(payload.get("batch_id")))
        if samples is None:
            raise ValueError("append_batch command contains invalid batch")
        batch = BatchSpec(str(payload["batch_id"]), samples, seed)
        state.batches.append(batch)
        return {"batch_id": batch.batch_id, "batch": batch}
    if command_type == "replace_weight":
        name = command.get("layer")
        found = state.find_layer(str(name))
        if found is None:
            raise ValueError(f"unknown layer {name}")
        index, old = found
        if "weight" in command:
            weight = np.asarray(command["weight"], dtype=np.float64)
        elif "weight_seed" in command:
            weight = generate_tensor((old.rows, old.cols), int(command["weight_seed"]))
        else:
            raise ValueError("replace_weight command lacks weight payload")
        if weight.shape != (old.rows, old.cols):
            raise ValueError("replace_weight payload has a mismatched shape")
        replacement = old.clone()
        replacement.weight = weight
        replacement.weight_seed = int(command["weight_seed"]) if "weight_seed" in command else None
        state.layers[index] = replacement
        return {"layer_index": index, "layer": replacement}
    if command_type == "set_quant_level":
        state.quant_level = int(command["quant_level"])
        return {"quant_level": state.quant_level}
    if command_type == "set_frac_bits":
        state.frac_bits = int(command["frac_bits"])
        return {"frac_bits": state.frac_bits}
    if command_type in {"set_revision", "rollback_revision"}:
        state.revision = int(command["revision"])
        return {"revision": state.revision}
    if command_type == "set_lease_limit":
        worker = state.find_worker(str(command.get("worker_id")))
        if worker is None:
            raise ValueError(f"unknown worker {command.get('worker_id')}")
        worker.lease_limit = int(command["lease_limit"])
        return {"worker_id": worker.worker_id, "lease_limit": worker.lease_limit}
    if command_type == "remove_batch":
        batch_id = str(command.get("batch_id"))
        state.batches = [batch for batch in state.batches if batch.batch_id != batch_id]
        return {"batch_id": batch_id}
    if command_type == "noop":
        return {}
    raise ValueError(f"unknown normalized command {command_type!r}")
