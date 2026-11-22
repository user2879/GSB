"""Content-addressed model and conversion artifact cache."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import atomic_write_json, content_hash, read_json, short_hash, to_jsonable


TRAIN_CACHE_SCHEMA = "train-cache/1"
CONVERSION_CACHE_SCHEMA = "conversion-cache/1"


@dataclass(frozen=True)
class CandidateIdentity:
    name: str
    scheme: str
    penalty: float
    mix: float | None
    feature_names: tuple[str, ...]
    feature_indices: tuple[int, ...]
    data_slice_fingerprint: str
    standardization_fingerprint: str
    sample_fingerprint: str
    penalty_fingerprint: str
    training_params_fingerprint: str
    train_key: str
    stable_order_key: str
    quota: int
    requested_epochs: int

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "name": self.name,
                "scheme": self.scheme,
                "penalty": self.penalty,
                "mix": self.mix,
                "feature_names": list(self.feature_names),
                "feature_indices": list(self.feature_indices),
                "data_slice_fingerprint": self.data_slice_fingerprint,
                "standardization_fingerprint": self.standardization_fingerprint,
                "sample_fingerprint": self.sample_fingerprint,
                "penalty_fingerprint": self.penalty_fingerprint,
                "training_params_fingerprint": self.training_params_fingerprint,
                "train_key": self.train_key,
                "stable_order_key": self.stable_order_key,
                "quota": self.quota,
                "requested_epochs": self.requested_epochs,
            }
        )


def penalty_fingerprint(scheme: str, penalty: float, mix: float | None) -> str:
    return short_hash("pen", {"scheme": scheme, "penalty": float(penalty), "mix": mix}, length=20)


def stable_model_order_key(scheme: str, penalty_fp: str, name: str) -> str:
    return f"{scheme}|{penalty_fp}|{name}"


def build_candidate_identity(
    *,
    model_name: str,
    scheme: str,
    penalty: float,
    mix: float | None,
    feature_names: list[str],
    feature_indices: list[int],
    selected_data_slice_fingerprint: str,
    standardization_fingerprint: str,
    sample_fingerprint: str,
    training_hyperparameters: dict[str, Any],
    quota: int,
    requested_epochs: int,
    seed: int,
    fault: dict[str, Any] | None,
) -> CandidateIdentity:
    penalty_fp = penalty_fingerprint(scheme, penalty, mix)
    training_params_fp = content_hash(
        "params",
        {
            "scheme": scheme,
            "penalty": float(penalty),
            "mix": mix,
            "training": training_hyperparameters,
            "quota": int(quota),
            "seed": int(seed),
            "fault": fault,
        },
    )
    train_key = content_hash(
        "train",
        {
            "schema": 1,
            "name": model_name,
            "scheme": scheme,
            "penalty": float(penalty),
            "mix": mix,
            "feature_names": list(feature_names),
            "data_slice_fingerprint": selected_data_slice_fingerprint,
            "standardization_fingerprint": standardization_fingerprint,
            "sample_fingerprint": sample_fingerprint,
            "training_hyperparameters": training_hyperparameters,
            "quota": int(quota),
            "seed": int(seed),
            "fault": fault,
        },
    )
    return CandidateIdentity(
        name=model_name,
        scheme=scheme,
        penalty=float(penalty),
        mix=mix,
        feature_names=tuple(feature_names),
        feature_indices=tuple(feature_indices),
        data_slice_fingerprint=selected_data_slice_fingerprint,
        standardization_fingerprint=standardization_fingerprint,
        sample_fingerprint=sample_fingerprint,
        penalty_fingerprint=penalty_fp,
        training_params_fingerprint=training_params_fp,
        train_key=train_key,
        stable_order_key=stable_model_order_key(scheme, penalty_fp, model_name),
        quota=int(quota),
        requested_epochs=int(requested_epochs),
    )


def build_conversion_key(
    identity: CandidateIdentity,
    training_artifact_fingerprint: str,
    exchange_version: str,
    schema_fingerprint: str,
    conversion_fault: dict[str, Any] | None,
    tolerance: float,
) -> str:
    return content_hash(
        "conv",
        {
            "schema": 1,
            "training_artifact_fingerprint": training_artifact_fingerprint,
            "version": exchange_version,
            "schema_fingerprint": schema_fingerprint,
            "feature_indices": list(identity.feature_indices),
            "feature_names": list(identity.feature_names),
            "conversion_fault": conversion_fault,
            "tolerance": float(tolerance),
        },
    )


class ArtifactCache:
    """JSON cache whose path and payload are validated against the cache key."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.train_root = self.root / "train"
        self.conversion_root = self.root / "conversion"
        self._lock = threading.Lock()

    @staticmethod
    def _path(root: Path, key: str) -> Path:
        return root / key[:10] / f"{key}.json"

    def train_path(self, key: str) -> Path:
        return self._path(self.train_root, key)

    def conversion_path(self, key: str) -> Path:
        return self._path(self.conversion_root, key)

    def load_train(self, key: str) -> dict[str, Any] | None:
        path = self.train_path(key)
        if not path.exists():
            return None
        try:
            payload = read_json(path)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != TRAIN_CACHE_SCHEMA or payload.get("key") != key:
            return None
        artifact = payload.get("artifact")
        fingerprint = payload.get("artifact_fingerprint")
        if not isinstance(artifact, dict) or not isinstance(fingerprint, str):
            return None
        if content_hash("artifact", artifact) != fingerprint:
            return None
        return payload

    def save_train(
        self,
        key: str,
        artifact: dict[str, Any],
        identity: CandidateIdentity,
    ) -> dict[str, Any]:
        fingerprint = content_hash("artifact", artifact)
        payload = {
            "schema_version": TRAIN_CACHE_SCHEMA,
            "key": key,
            "identity": identity.to_dict(),
            "artifact_fingerprint": fingerprint,
            "artifact": artifact,
        }
        path = self.train_path(key)
        with self._lock:
            atomic_write_json(path, payload)
        return payload

    def load_conversion(self, key: str) -> dict[str, Any] | None:
        path = self.conversion_path(key)
        if not path.exists():
            return None
        try:
            payload = read_json(path)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != CONVERSION_CACHE_SCHEMA or payload.get("key") != key:
            return None
        conversion = payload.get("conversion")
        fingerprint = payload.get("conversion_fingerprint")
        if not isinstance(conversion, dict) or not isinstance(fingerprint, str):
            return None
        if content_hash("conversion", conversion) != fingerprint:
            return None
        return payload

    def save_conversion(
        self,
        key: str,
        conversion: dict[str, Any],
        *,
        training_artifact_fingerprint: str,
        exchange_version: str,
    ) -> dict[str, Any]:
        fingerprint = content_hash("conversion", conversion)
        payload = {
            "schema_version": CONVERSION_CACHE_SCHEMA,
            "key": key,
            "training_artifact_fingerprint": training_artifact_fingerprint,
            "exchange_version": exchange_version,
            "conversion_fingerprint": fingerprint,
            "conversion": conversion,
        }
        path = self.conversion_path(key)
        with self._lock:
            atomic_write_json(path, payload)
        return payload
