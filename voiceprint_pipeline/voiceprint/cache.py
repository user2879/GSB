"""Content-fingerprint cache keys and versioned atomic snapshot storage."""
from __future__ import annotations

import os
from typing import Any

from .util import atomic_write_json, fingerprint, read_json, sha256_text, stable_seed

STAGE_FEATURES = "features"
STAGE_AGGREGATION = "aggregation"
STAGE_EMBEDDING = "embedding"
STAGE_SIMILARITY = "similarity"
STAGE_ORDER = (STAGE_FEATURES, STAGE_AGGREGATION, STAGE_EMBEDDING, STAGE_SIMILARITY)

MANIFEST_NAME = "manifest.json"
VERSION_DIR_PREFIX = "v"


def feature_cache_key(sample_id: str, pcm_fp: str, feature_scope: str) -> str:
    return sha256_text(f"features|{sample_id}|{pcm_fp}|{feature_scope}")[:16]


def aggregation_cache_key(feature_key: str, stats_scope: str) -> str:
    return sha256_text(f"aggregation|{feature_key}|{stats_scope}")[:16]


def embedding_cache_key(agg_key: str, quant_scope: str, embed_scope: str, width: int,
                        levels: int) -> str:
    return sha256_text(
        f"embedding|{agg_key}|{quant_scope}|{embed_scope}|w{width}|L{levels}"
    )[:16]


def similarity_cache_key(emb_key_a: str, emb_key_b: str, sim_scope: str, width: int) -> str:
    return sha256_text(
        f"similarity|{emb_key_a}|{emb_key_b}|{sim_scope}|w{width}"
    )[:16]


class SnapshotStore:
    """Versioned snapshot store; each commit lands as a brand-new version directory.

    Recovery recognizes version changes via the manifest pointer and never mutates
    a previously committed version in place.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self.versions_dir = os.path.join(root, "versions")
        os.makedirs(self.versions_dir, exist_ok=True)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, MANIFEST_NAME)

    def has_manifest(self) -> bool:
        return os.path.isfile(self.manifest_path)

    def read_manifest(self) -> dict[str, Any]:
        return read_json(self.manifest_path)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def _version_dir(self, version: int) -> str:
        path = os.path.join(self.versions_dir, f"{VERSION_DIR_PREFIX}{version:08d}")
        os.makedirs(path, exist_ok=True)
        return path

    def artifact_path(self, version: int, stage: str, key: str) -> str:
        return os.path.join(self._version_dir(version), stage, f"{key}.json")

    def write_artifact(self, version: int, stage: str, key: str, payload: Any) -> str:
        path = self.artifact_path(version, stage, key)
        atomic_write_json(path, payload)
        return path

    def read_artifact(self, version: int, stage: str, key: str) -> Any:
        return read_json(self.artifact_path(version, stage, key))

    def artifact_exists(self, version: int, stage: str, key: str) -> bool:
        return os.path.isfile(self.artifact_path(version, stage, key))

    def commit(self, manifest: dict[str, Any]) -> None:
        """Whole-stage commit: bump version, write artifacts pointer, atomically swap manifest."""
        manifest["manifest_fp"] = sha256_text(fingerprint(
            {k: v for k, v in manifest.items() if k != "manifest_fp"}
        ))
        self.write_manifest(manifest)

    def prune_old_versions(self, keep: int = 3,
                           referenced_versions: set[int] | None = None) -> None:
        """Reclaim version directories that no manifest pointer references.

        Referenced versions are always retained, even if more than ``keep``
        directories exist, so recovery can still resolve every artifact.
        """
        referenced = set(referenced_versions or set())
        versions = sorted(
            name for name in os.listdir(self.versions_dir)
            if name.startswith(VERSION_DIR_PREFIX)
        )
        retained_tail = set(versions[-keep:])
        for name in versions:
            version_number = int(name[len(VERSION_DIR_PREFIX):])
            if version_number in referenced or name in retained_tail:
                continue
            path = os.path.join(self.versions_dir, name)
            for stage in STAGE_ORDER:
                stage_dir = os.path.join(path, stage)
                if not os.path.isdir(stage_dir):
                    continue
                for file_name in os.listdir(stage_dir):
                    os.remove(os.path.join(stage_dir, file_name))
                os.rmdir(stage_dir)
            if os.path.isdir(path):
                os.rmdir(path)


def stable_task_key(stage: str, identifier: str) -> str:
    return f"{stage}|{identifier}"


def pair_references(pair_id: str, sample_a: str, sample_b: str) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "sample_a": sample_a,
        "sample_b": sample_b,
        "reference_seed": stable_seed(f"pair-ref|{pair_id}|{sample_a}|{sample_b}"),
    }
