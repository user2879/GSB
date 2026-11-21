"""Content-fingerprint addressed artifact store used for exact reuse."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import PreprocessResult, preprocess_artifact_fingerprint
from .utils import (
    atomic_write_json,
    atomic_write_npz,
    hash_array,
    hash_object,
    read_json,
)


def preprocess_payload(result: PreprocessResult) -> dict[str, Any]:
    return result.to_metadata()


def preprocess_from_payload(payload: dict[str, Any], X: np.ndarray) -> PreprocessResult:
    shape = tuple(int(value) for value in payload.get("shape", X.shape))
    return PreprocessResult(
        X=np.asarray(X, dtype=np.float64),
        sample_ids=[str(value) for value in payload.get("sample_ids", [])],
        feature_names=[str(value) for value in payload.get("feature_names", [])],
        column_types=dict(payload.get("column_types", {})),
        category_mappings=dict(payload.get("category_mappings", {})),
        statistics=dict(payload.get("statistics", {})),
        dropped_columns=list(payload.get("dropped_columns", [])),
        train_row_count=int(payload.get("train_row_count", 0)),
        train_mask=[bool(value) for value in payload.get("train_mask", [])],
        method=str(payload.get("method", "standard")),
        parameter_fingerprint=str(payload.get("parameter_fingerprint", "")),
        content_fingerprint=str(payload.get("content_fingerprint", "")),
        artifact_fingerprint=str(payload.get("artifact_fingerprint", "")),
        shape=(int(shape[0]), int(shape[1])),
    )


class ArtifactStore:
    def __init__(self, output_root: Path):
        self.root = Path(output_root) / "cache" / "artifacts"
        self.preprocess_root = self.root / "preprocess"
        self.cluster_root = self.root / "clusters"
        self.evaluation_root = self.root / "evaluations"
        self.ledger_path = Path(output_root) / "cache" / "ledger.json"
        for path in (self.preprocess_root, self.cluster_root, self.evaluation_root):
            path.mkdir(parents=True, exist_ok=True)

    def save_preprocess(self, result: PreprocessResult) -> None:
        prefix = result.content_fingerprint
        atomic_write_json(self.preprocess_root / f"{prefix}.json", preprocess_payload(result))
        atomic_write_npz(self.preprocess_root / f"{prefix}.npz", X=result.X)

    def load_preprocess(self, content_fingerprint: str) -> PreprocessResult | None:
        meta_path = self.preprocess_root / f"{content_fingerprint}.json"
        matrix_path = self.preprocess_root / f"{content_fingerprint}.npz"
        payload = read_json(meta_path)
        if not isinstance(payload, dict) or not matrix_path.exists():
            return None
        try:
            with np.load(matrix_path, allow_pickle=False) as loaded:
                X = loaded["X"].copy()
        except (OSError, ValueError, KeyError):
            return None
        result = preprocess_from_payload(payload, X)
        if result.content_fingerprint != content_fingerprint:
            return None
        expected_artifact = preprocess_artifact_fingerprint(
            result.content_fingerprint,
            result.X,
            result.feature_names,
            result.statistics,
            result.dropped_columns,
            result.train_mask,
        )
        if result.artifact_fingerprint != expected_artifact:
            return None
        return result

    def save_cluster(self, identity_fingerprint: str, record: dict[str, Any]) -> None:
        atomic_write_json(self.cluster_root / f"{identity_fingerprint}.json", record)

    def load_cluster(self, identity_fingerprint: str) -> dict[str, Any] | None:
        record = read_json(self.cluster_root / f"{identity_fingerprint}.json")
        if not isinstance(record, dict):
            return None
        if record.get("identity_fingerprint") != identity_fingerprint:
            return None
        expected_artifact = hash_object(
            "clusterart",
            {
                "identity": identity_fingerprint,
                "label_hash": hash_array(np.asarray(record.get("labels", []), dtype=np.float64)),
                "status": record.get("status"),
                "iterations_executed": record.get("iterations_executed", 0),
                "trajectory": record.get("trajectory", []),
                "diagnostics": record.get("diagnostics", {}),
                "flags": record.get("diagnostic_flags", []),
                "error_code": record.get("error_code"),
            },
        )
        if record.get("artifact_fingerprint") != expected_artifact:
            return None
        return record

    def save_evaluation(self, identity_fingerprint: str, record: dict[str, Any]) -> None:
        atomic_write_json(self.evaluation_root / f"{identity_fingerprint}.json", record)

    def load_evaluation(self, identity_fingerprint: str) -> dict[str, Any] | None:
        record = read_json(self.evaluation_root / f"{identity_fingerprint}.json")
        if not isinstance(record, dict):
            return None
        if record.get("identity_fingerprint") != identity_fingerprint:
            return None
        evaluation = record.get("evaluation", {})
        expected_artifact = hash_object(
            "evalart",
            {
                "identity": identity_fingerprint,
                "labels": record.get("label_hash"),
                "metrics": evaluation.get("metrics"),
                "flags": evaluation.get("diagnostic_flags"),
                "boundary_check": evaluation.get("boundary_check"),
            },
        )
        if record.get("artifact_fingerprint") != expected_artifact:
            return None
        return record

    def load_ledger(self) -> dict[str, Any]:
        value = read_json(self.ledger_path, {})
        return value if isinstance(value, dict) else {}

    def save_ledger(self, ledger: dict[str, Any]) -> None:
        atomic_write_json(self.ledger_path, ledger)
