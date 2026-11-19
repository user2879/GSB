from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .canonical import atomic_write_bytes, canonical_bytes, digest_object, read_json
from .config import BudgetConfig
from .errors import WorkbenchError


INDEX_SCHEMA = 1
SHARD_SCHEMA = 1
SHARD_COUNT = 16


def shard_id_for_hash(fingerprint_hash: str) -> str:
    if len(fingerprint_hash) < 1 or any(character not in "0123456789abcdef" for character in fingerprint_hash):
        raise WorkbenchError(
            "E_PARTITION_HASH",
            "fingerprint hash is not lowercase hexadecimal",
            {"hash": fingerprint_hash},
        )
    return fingerprint_hash[0]


def _finalize(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    content_hash = digest_object(payload)
    document = dict(payload)
    document["content_hash"] = content_hash
    return document, content_hash


def make_shard_payload(
    shard: str,
    version_stamp: int,
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["hash"]].append(
            {
                "track_id": record["track_id"],
                "kind": record["kind"],
                "anchor_frame": int(record["anchor_frame"]),
                "target_frame": int(record["target_frame"]),
                "anchor_bin": int(record["anchor_bin"]),
                "target_bin": int(record["target_bin"]),
            }
        )
    entries = []
    occurrence_count = 0
    for fingerprint_hash in sorted(grouped):
        occurrences = sorted(
            grouped[fingerprint_hash],
            key=lambda item: (
                item["track_id"],
                item["kind"],
                item["anchor_frame"],
                item["target_frame"],
                item["anchor_bin"],
                item["target_bin"],
            ),
        )
        occurrence_count += len(occurrences)
        entries.append({"hash": fingerprint_hash, "occurrences": occurrences})
    return {
        "schema": SHARD_SCHEMA,
        "shard": shard,
        "version_stamp": int(version_stamp),
        "entry_count": len(entries),
        "occurrence_count": occurrence_count,
        "entries": entries,
    }


def entries_to_mapping(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {entry["hash"]: list(entry["occurrences"]) for entry in entries}


def mapping_to_records(shard: str, mapping: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fingerprint_hash, occurrences in mapping.items():
        if shard_id_for_hash(fingerprint_hash) != shard:
            raise WorkbenchError(
                "E_PARTITION_HASH",
                "fingerprint hash was assigned to the wrong shard",
                {"hash": fingerprint_hash, "shard": shard},
            )
        for occurrence in occurrences:
            records.append({"hash": fingerprint_hash, **occurrence})
    return records


def make_manifest_payload(
    *,
    version: int,
    state_digest: str,
    fingerprint_config_digest: str,
    fingerprints: dict[str, str],
    member_count: int,
    shards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": INDEX_SCHEMA,
        "version": int(version),
        "state_digest": state_digest,
        "fingerprint_config_digest": fingerprint_config_digest,
        "fingerprints": {key: fingerprints[key] for key in sorted(fingerprints)},
        "member_count": int(member_count),
        "shards": {key: shards[key] for key in sorted(shards)},
    }


class IndexStore:
    def __init__(
        self,
        root: str | Path,
        budget: BudgetConfig,
        crash_hook: Callable[[str, int, Path], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.partitions_dir = self.root / "indexes" / "partitions"
        self.manifests_dir = self.root / "indexes" / "manifests"
        self.budget = budget
        self.crash_hook = crash_hook
        self.shard_write_count = 0
        self.partitions_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def begin_commit(self) -> None:
        self.shard_write_count = 0

    def _piece_path(self, directory: Path, content_hash: str) -> Path:
        return directory / f"{content_hash}.json"

    def save_shard(
        self,
        payload: dict[str, Any],
        *,
        crashable: bool = False,
    ) -> dict[str, Any]:
        document, content_hash = _finalize(payload)
        data = canonical_bytes(document) + b"\n"
        maximum = int(self.budget.index_partition_memory_mb * 1024 * 1024)
        if len(data) > maximum:
            raise WorkbenchError(
                "E_BUDGET_DENIED",
                "serialized index shard exceeds the maintenance memory budget",
                {
                    "shard": payload.get("shard"),
                    "bytes": len(data),
                    "budget_bytes": maximum,
                },
            )
        path = self._piece_path(self.partitions_dir, content_hash)
        if not path.exists():
            self.shard_write_count += 1
            write_index = self.shard_write_count

            def before_replace(temporary: Path) -> None:
                if crashable and self.crash_hook is not None:
                    self.crash_hook("shard_temp", write_index, temporary)

            atomic_write_bytes(path, data, before_replace=before_replace)
        return {
            "file": f"{content_hash}.json",
            "content_hash": content_hash,
            "version_stamp": int(payload["version_stamp"]),
            "entry_count": int(payload["entry_count"]),
            "occurrence_count": int(payload["occurrence_count"]),
            "size_bytes": len(data),
        }

    def save_manifest(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        document, content_hash = _finalize(payload)
        path = self._piece_path(self.manifests_dir, content_hash)
        data = canonical_bytes(document) + b"\n"
        if path.exists():
            existing = path.read_bytes()
            if existing != data:
                raise WorkbenchError(
                    "E_PARTITION_HASH",
                    "content-addressed manifest path contains different bytes",
                    {"manifest_hash": content_hash},
                )
        else:
            atomic_write_bytes(path, data)
        return document, content_hash

    def load_manifest(self, manifest_hash: str) -> dict[str, Any]:
        path = self._piece_path(self.manifests_dir, manifest_hash)
        if not path.exists():
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "referenced index manifest is missing",
                {"manifest_hash": manifest_hash},
            )
        try:
            document = read_json(path)
        except Exception as exc:
            raise WorkbenchError(
                "E_PARTITION_HASH",
                "index manifest is not valid JSON",
                {"manifest_hash": manifest_hash, "reason": str(exc)},
            ) from exc
        if document.get("schema") != INDEX_SCHEMA:
            raise WorkbenchError(
                "E_INDEX_VERSION",
                "unsupported index manifest schema",
                {"schema": document.get("schema"), "expected": INDEX_SCHEMA},
            )
        payload = {key: value for key, value in document.items() if key != "content_hash"}
        actual = digest_object(payload)
        if actual != manifest_hash or document.get("content_hash") != manifest_hash:
            raise WorkbenchError(
                "E_PARTITION_HASH",
                "index manifest content hash mismatch",
                {
                    "manifest_hash": manifest_hash,
                    "document_hash": document.get("content_hash"),
                    "actual": actual,
                },
            )
        return document

    def load_shard(self, reference: dict[str, Any], expected_shard: str | None = None) -> dict[str, Any]:
        content_hash = str(reference["content_hash"])
        path = self._piece_path(self.partitions_dir, content_hash)
        if not path.exists():
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "referenced partition is missing",
                {"content_hash": content_hash},
            )
        try:
            document = read_json(path)
        except Exception as exc:
            raise WorkbenchError(
                "E_PARTITION_HASH",
                "partition is not valid JSON",
                {"content_hash": content_hash, "reason": str(exc)},
            ) from exc
        if document.get("schema") != SHARD_SCHEMA:
            raise WorkbenchError(
                "E_INDEX_VERSION",
                "unsupported partition schema",
                {"schema": document.get("schema"), "expected": SHARD_SCHEMA},
            )
        payload = {key: value for key, value in document.items() if key != "content_hash"}
        actual = digest_object(payload)
        if actual != content_hash or document.get("content_hash") != content_hash:
            raise WorkbenchError(
                "E_PARTITION_HASH",
                "partition content hash mismatch",
                {
                    "content_hash": content_hash,
                    "document_hash": document.get("content_hash"),
                    "actual": actual,
                },
            )
        if int(document.get("version_stamp", -1)) != int(reference["version_stamp"]):
            raise WorkbenchError(
                "E_INDEX_VERSION",
                "partition version stamp mismatch",
                {
                    "content_hash": content_hash,
                    "document": document.get("version_stamp"),
                    "manifest": reference["version_stamp"],
                },
            )
        if expected_shard is not None and document.get("shard") != expected_shard:
            raise WorkbenchError(
                "E_PARTITION_HASH",
                "partition shard identifier mismatch",
                {"expected": expected_shard, "actual": document.get("shard")},
            )
        for entry in document.get("entries", []):
            if shard_id_for_hash(entry["hash"]) != document.get("shard"):
                raise WorkbenchError(
                    "E_PARTITION_HASH",
                    "partition contains a hash outside its shard range",
                    {"hash": entry["hash"], "shard": document.get("shard")},
                )
        return document
