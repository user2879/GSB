from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from .audio import audio_content_hash, read_wav_bytes, validate_audio_bytes
from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_bytes,
    digest_object,
    read_json,
    sha256_file,
)
from .config import BudgetConfig, FingerprintConfig
from .errors import WorkbenchError
from .fingerprint import FingerprintResult, extract_fingerprint, validate_fingerprint_document
from .index import (
    IndexStore,
    entries_to_mapping,
    make_manifest_payload,
    make_shard_payload,
    mapping_to_records,
    shard_id_for_hash,
)
from .state import LibraryState, Member, Operation


CURRENT_SCHEMA = 1
STOP_SCHEMA = 1
OP_FILE_PATTERN = re.compile(r"^v(\d{6})\.json$")


def _version_filename(version: int) -> str:
    return f"v{version:06d}.json"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


class Workbench:
    """Versioned library, deterministic fingerprints, and crash-consistent index."""

    def __init__(
        self,
        root: str | Path,
        *,
        budget: BudgetConfig | None = None,
        crash_hook: Callable[[str, int, Path], None] | None = None,
        create: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = budget or BudgetConfig()
        self.crash_hook = crash_hook
        self.ops_dir = self.root / "ops"
        self.fingerprints_dir = self.root / "fingerprints"
        self.audio_dir = self.root / "audio"
        self.recovery_dir = self.root / "recovery"
        self.cache_dir = self.root / "cache"
        for directory in (
            self.ops_dir,
            self.fingerprints_dir,
            self.audio_dir,
            self.recovery_dir,
            self.cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.current_path = self.root / "CURRENT.json"
        self.stop_path = self.root / "STOP.json"
        self.lock_path = self.root / "WORKBENCH.lock"
        self._lock_handle: Any | None = None
        self._closed = False
        self._acquire_lock()
        self.recovery_report = self._recover()
        if self.current_path.exists():
            self.current = self._load_current()
            self._verify_current_manifest()
        elif create:
            self.current = self._initialize_empty()
            self.recovery_report = self._write_recovery_report(
                status="cold_start",
                clean_previous_exit=False,
                recovered_version=0,
                manifest_hash=self.current.manifest_hash,
            )
        else:
            raise WorkbenchError("E_SNAPSHOT_STALE", "workbench has no committed state")
        self.index = IndexStore(self.root, self.budget, self.crash_hook)

    def __enter__(self) -> "Workbench":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _acquire_lock(self) -> None:
        handle = open(self.lock_path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise WorkbenchError(
                "E_LOCKED",
                "workbench is already open by another process",
                {"lock": _relative(self.lock_path, self.root)},
            ) from exc
        self._lock_handle = handle

    def _release_lock(self) -> None:
        if self._lock_handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._lock_handle.seek(0)
                msvcrt.locking(self._lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_handle.close()
            self._lock_handle = None

    def close(self) -> None:
        if self._closed:
            return
        stop_document = {
            "schema": STOP_SCHEMA,
            "clean": True,
            "version": self.current.version,
            "manifest_hash": self.current.manifest_hash,
            "state_digest": self.current.state_digest,
        }
        atomic_write_json(self.stop_path, stop_document)
        self._closed = True
        self._release_lock()

    def _read_clean_stop(self) -> dict[str, Any] | None:
        if not self.stop_path.exists():
            return None
        try:
            document = read_json(self.stop_path)
        except Exception:
            document = None
        self.stop_path.unlink(missing_ok=True)
        return document if isinstance(document, dict) else None

    def _load_operation(self, version: int) -> Operation:
        path = self.ops_dir / _version_filename(version)
        if not path.exists():
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "operation record is missing",
                {"version": version, "file": _relative(path, self.root)},
            )
        try:
            document = read_json(path)
        except Exception as exc:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "operation record is unreadable",
                {"version": version, "reason": str(exc)},
            ) from exc
        operation = Operation.from_document(document)
        if operation.version != version:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "operation file name/version mismatch",
                {"expected": version, "actual": operation.version},
            )
        return operation

    def _load_current(self) -> LibraryState:
        try:
            pointer = read_json(self.current_path)
        except Exception as exc:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "CURRENT pointer is unreadable",
                {"reason": str(exc)},
            ) from exc
        if pointer.get("schema") != CURRENT_SCHEMA:
            raise WorkbenchError(
                "E_INDEX_VERSION",
                "unsupported CURRENT pointer schema",
                {"schema": pointer.get("schema"), "expected": CURRENT_SCHEMA},
            )
        operation = self._load_operation(int(pointer["version"]))
        state = operation.state
        if (
            state.manifest_hash != pointer.get("manifest_hash")
            or state.state_digest != pointer.get("state_digest")
        ):
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "CURRENT pointer does not match committed operation",
                {
                    "pointer_version": pointer.get("version"),
                    "state_version": state.version,
                },
            )
        return state

    def _committed_chain(self) -> dict[int, Operation]:
        current = self._load_current()
        chain: dict[int, Operation] = {}
        version = current.version
        expected_state = current
        while version >= 0:
            operation = self._load_operation(version)
            if operation.state.to_document() != expected_state.to_document():
                raise WorkbenchError(
                    "E_SNAPSHOT_STALE",
                    "operation chain state mismatch",
                    {"version": version},
                )
            if version in chain:
                raise WorkbenchError(
                    "E_SNAPSHOT_STALE",
                    "cycle detected in operation chain",
                    {"version": version},
                )
            chain[version] = operation
            if version == 0:
                if operation.parent_version != -1:
                    raise WorkbenchError(
                        "E_SNAPSHOT_STALE",
                        "initial operation has an invalid parent",
                        {"parent_version": operation.parent_version},
                    )
                break
            if operation.parent_version != version - 1:
                raise WorkbenchError(
                    "E_SNAPSHOT_STALE",
                    "operation chain has a version gap",
                    {"version": version, "parent_version": operation.parent_version},
                )
            expected_state = self._load_operation(operation.parent_version).state
            version = operation.parent_version
        return chain

    def _verify_manifest_for_state(self, state: LibraryState, *, verify_shards: bool) -> dict[str, Any]:
        manifest = self.index.load_manifest(state.manifest_hash) if hasattr(self, "index") else self._load_manifest_bootstrap(state.manifest_hash)
        if manifest.get("state_digest") != state.state_digest:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "index manifest belongs to a different state",
                {
                    "manifest_state": manifest.get("state_digest"),
                    "state_digest": state.state_digest,
                },
            )
        if manifest.get("fingerprint_config_digest") != state.fingerprint_config.digest:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "index manifest fingerprint configuration mismatch",
                {"version": state.version},
            )
        if manifest.get("fingerprints") != state.fingerprint_map:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "index manifest fingerprint mapping mismatch",
                {"version": state.version},
            )
        if verify_shards:
            for shard, reference in manifest.get("shards", {}).items():
                self._load_shard_bootstrap(reference, shard)
        return manifest

    def _verify_current_manifest(self) -> None:
        bootstrap = IndexStore(self.root, self.budget, self.crash_hook)
        self.index = bootstrap
        self._verify_manifest_for_state(self.current, verify_shards=True)

    def _load_manifest_bootstrap(self, manifest_hash: str) -> dict[str, Any]:
        bootstrap = IndexStore(self.root, self.budget, self.crash_hook)
        return bootstrap.load_manifest(manifest_hash)

    def _load_shard_bootstrap(self, reference: dict[str, Any], shard: str) -> dict[str, Any]:
        bootstrap = IndexStore(self.root, self.budget, self.crash_hook)
        return bootstrap.load_shard(reference, shard)

    def _collect_committed_references(
        self, chain: dict[int, Operation]
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        manifests: set[str] = set()
        shards: set[str] = set()
        fingerprints: set[str] = set()
        audio: set[str] = set()
        bootstrap = IndexStore(self.root, self.budget, self.crash_hook)
        for version in sorted(chain):
            operation = chain[version]
            state = operation.state
            manifests.add(state.manifest_hash)
            fingerprints.update(state.fingerprint_map.values())
            audio.update(member.audio_sha256 for member in state.members)
            manifest = bootstrap.load_manifest(state.manifest_hash)
            if manifest.get("state_digest") != state.state_digest:
                raise WorkbenchError(
                    "E_SNAPSHOT_STALE",
                    "historical manifest state digest mismatch",
                    {"version": version},
                )
            for shard, reference in manifest.get("shards", {}).items():
                shards.add(str(reference["content_hash"]))
                bootstrap.load_shard(reference, shard)
        return manifests, shards, fingerprints, audio

    def _scan_orphans(self, directory: Path, expected_stems: set[str], suffix: str) -> list[dict[str, Any]]:
        orphans: list[dict[str, Any]] = []
        if not directory.exists():
            return orphans
        for path in sorted(directory.glob(f"*{suffix}")):
            if not path.is_file():
                continue
            stem = path.name[: -len(suffix)]
            if stem not in expected_stems:
                item = {"file": _relative(path, self.root), "bytes": path.stat().st_size}
                orphans.append(item)
                path.unlink()
        return orphans

    def _recover(self) -> dict[str, Any]:
        clean_stop = self._read_clean_stop()
        preexisting_ops = list(self.ops_dir.glob("*.json"))
        if not self.current_path.exists():
            if preexisting_ops:
                raise WorkbenchError(
                    "E_SNAPSHOT_STALE",
                    "operation records exist but CURRENT pointer is missing",
                )
            return self._write_recovery_report(
                status="cold_start",
                clean_previous_exit=False,
                recovered_version=None,
                manifest_hash=None,
            )
        chain = self._committed_chain()
        manifests, shards, fingerprints, audio = self._collect_committed_references(chain)
        current = chain[max(chain)].state
        clean_previous = bool(
            clean_stop
            and clean_stop.get("clean") is True
            and int(clean_stop.get("version", -1)) == current.version
            and clean_stop.get("manifest_hash") == current.manifest_hash
            and clean_stop.get("state_digest") == current.state_digest
        )

        orphan_temps: list[dict[str, Any]] = []
        for path in sorted(self.root.rglob("*.tmp")):
            if path.is_file():
                orphan_temps.append({"file": _relative(path, self.root), "bytes": path.stat().st_size})
                path.unlink()
        orphan_partitions = self._scan_orphans(
            self.root / "indexes" / "partitions", shards, ".json"
        )
        orphan_manifests = self._scan_orphans(
            self.root / "indexes" / "manifests", manifests, ".json"
        )
        orphan_ops: list[dict[str, Any]] = []
        for path in sorted(self.ops_dir.glob("v*.json")):
            match = OP_FILE_PATTERN.match(path.name)
            if match is None or int(match.group(1)) not in chain:
                orphan_ops.append({"file": _relative(path, self.root), "bytes": path.stat().st_size})
                path.unlink()
        orphan_fingerprints = self._scan_orphans(self.fingerprints_dir, fingerprints, ".json")
        orphan_audio = self._scan_orphans(self.audio_dir, audio, ".wav")
        return self._write_recovery_report(
            status="clean" if clean_previous else "recovered",
            clean_previous_exit=clean_previous,
            recovered_version=current.version,
            manifest_hash=current.manifest_hash,
            orphan_temp_files=orphan_temps,
            orphan_partitions=orphan_partitions,
            orphan_manifests=orphan_manifests,
            orphan_operations=orphan_ops,
            orphan_fingerprints=orphan_fingerprints,
            orphan_audio=orphan_audio,
        )

    def _write_recovery_report(
        self,
        *,
        status: str,
        clean_previous_exit: bool,
        recovered_version: int | None,
        manifest_hash: str | None,
        orphan_temp_files: list[dict[str, Any]] | None = None,
        orphan_partitions: list[dict[str, Any]] | None = None,
        orphan_manifests: list[dict[str, Any]] | None = None,
        orphan_operations: list[dict[str, Any]] | None = None,
        orphan_fingerprints: list[dict[str, Any]] | None = None,
        orphan_audio: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        report = {
            "schema": 1,
            "status": status,
            "clean_previous_exit": clean_previous_exit,
            "recovered_version": recovered_version,
            "manifest_hash": manifest_hash,
            "orphan_temp_files": orphan_temp_files or [],
            "orphan_partitions": orphan_partitions or [],
            "orphan_manifests": orphan_manifests or [],
            "orphan_operations": orphan_operations or [],
            "orphan_fingerprints": orphan_fingerprints or [],
            "orphan_audio": orphan_audio or [],
            "action": "removed_after_stable_pointer_check",
        }
        atomic_write_json(self.recovery_dir / "last_recovery.json", report)
        return report

    def _initialize_empty(self) -> LibraryState:
        fingerprint_config = FingerprintConfig()
        provisional = LibraryState.create(
            version=0,
            members=(),
            fingerprint_config=fingerprint_config,
            similarity_threshold=0.08,
            fingerprints={},
            manifest_hash="",
        )
        manifest_payload = make_manifest_payload(
            version=0,
            state_digest=provisional.state_digest,
            fingerprint_config_digest=fingerprint_config.digest,
            fingerprints={},
            member_count=0,
            shards={},
        )
        bootstrap = IndexStore(self.root, self.budget, self.crash_hook)
        _, manifest_hash = bootstrap.save_manifest(manifest_payload)
        state = LibraryState.create(
            version=0,
            members=(),
            fingerprint_config=fingerprint_config,
            similarity_threshold=0.08,
            fingerprints={},
            manifest_hash=manifest_hash,
        )
        operation = Operation(
            version=0,
            parent_version=-1,
            operation={"type": "init", "detail": "empty deterministic workbench"},
            state=state,
        )
        atomic_write_json(self.ops_dir / _version_filename(0), operation.to_document())
        atomic_write_json(
            self.current_path,
            {
                "schema": CURRENT_SCHEMA,
                "version": state.version,
                "state_digest": state.state_digest,
                "manifest_hash": state.manifest_hash,
            },
        )
        return state

    def _store_audio(self, data: bytes) -> str:
        content_hash = audio_content_hash(data)
        path = self.audio_dir / f"{content_hash}.wav"
        if path.exists():
            if sha256_file(path) != content_hash:
                raise WorkbenchError(
                    "E_PARTITION_HASH",
                    "content-addressed audio file is corrupt",
                    {"audio_sha256": content_hash},
                )
        else:
            atomic_write_bytes(path, data)
        return content_hash

    def _load_audio(self, content_hash: str) -> tuple[bytes, Any]:
        path = self.audio_dir / f"{content_hash}.wav"
        if not path.exists():
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "content-addressed audio file is missing",
                {"audio_sha256": content_hash},
            )
        data = path.read_bytes()
        if audio_content_hash(data) != content_hash:
            raise WorkbenchError(
                "E_PARTITION_HASH",
                "content-addressed audio hash mismatch",
                {"audio_sha256": content_hash},
            )
        samples, _ = validate_audio_bytes(data)
        return data, samples

    def _store_fingerprint(self, result: FingerprintResult) -> None:
        path = self.fingerprints_dir / f"{result.content_hash}.json"
        data = canonical_bytes(result.document) + b"\n"
        if path.exists():
            if path.read_bytes() != data:
                raise WorkbenchError(
                    "E_PARTITION_HASH",
                    "fingerprint content-addressed path contains different bytes",
                    {"content_hash": result.content_hash},
                )
        else:
            atomic_write_bytes(path, data)

    def load_fingerprint(self, content_hash: str) -> dict[str, Any]:
        path = self.fingerprints_dir / f"{content_hash}.json"
        if not path.exists():
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "fingerprint file is missing",
                {"content_hash": content_hash},
            )
        document = validate_fingerprint_document(read_json(path))
        if document.get("content_hash") != content_hash:
            raise WorkbenchError(
                "E_PARTITION_HASH",
                "fingerprint file name/content mismatch",
                {"content_hash": content_hash},
            )
        return document

    def _extract_for_audio(self, audio_hash: str, config: FingerprintConfig) -> FingerprintResult:
        _, samples = self._load_audio(audio_hash)
        return extract_fingerprint(samples, config, audio_sha256=audio_hash)

    def _records_for_fingerprint(self, track_id: str, document: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"track_id": track_id, **record}
            for record in document["fingerprints"]
        ]

    def _group_records_by_shard(
        self, records: Iterable[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            shard = shard_id_for_hash(record["hash"])
            grouped.setdefault(shard, []).append(record)
        return grouped

    def _manifest(self, manifest_hash: str) -> dict[str, Any]:
        return self.index.load_manifest(manifest_hash)

    def _install_state(self, state: LibraryState, operation: dict[str, Any]) -> LibraryState:
        if state.version != self.current.version + 1:
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "new state version is not the next version",
                {"current": self.current.version, "new": state.version},
            )
        record = Operation(
            version=state.version,
            parent_version=self.current.version,
            operation=operation,
            state=state,
        )
        atomic_write_json(self.ops_dir / _version_filename(state.version), record.to_document())
        atomic_write_json(
            self.current_path,
            {
                "schema": CURRENT_SCHEMA,
                "version": state.version,
                "state_digest": state.state_digest,
                "manifest_hash": state.manifest_hash,
            },
        )
        self.current = state
        return state

    def import_track(
        self,
        track_id: str,
        source_path: str | Path,
        *,
        display_name: str | None = None,
    ) -> LibraryState:
        if not track_id or "/" in track_id or "\\" in track_id:
            raise WorkbenchError("E_CONFIG_RANGE", "track_id must be a non-empty simple name")
        if track_id in self.current.member_map:
            raise WorkbenchError(
                "E_STATE_CONFLICT",
                "track id already exists in the current state",
                {"track_id": track_id, "version": self.current.version},
            )
        data, samples = read_wav_bytes(source_path)
        audio_hash = self._store_audio(data)
        fingerprint = self._extract_for_audio(audio_hash, self.current.fingerprint_config)
        self._store_fingerprint(fingerprint)
        document = fingerprint.document
        records = self._records_for_fingerprint(track_id, document)
        grouped = self._group_records_by_shard(records)
        new_version = self.current.version + 1
        old_manifest = self._manifest(self.current.manifest_hash)
        shards = dict(old_manifest.get("shards", {}))
        self.index.begin_commit()
        for shard in sorted(grouped):
            old_reference = shards.get(shard)
            if old_reference is None:
                mapping: dict[str, list[dict[str, Any]]] = {}
            else:
                mapping = entries_to_mapping(self.index.load_shard(old_reference, shard)["entries"])
            for record in grouped[shard]:
                mapping.setdefault(record["hash"], []).append(
                    {
                        "track_id": record["track_id"],
                        "kind": record["kind"],
                        "anchor_frame": record["anchor_frame"],
                        "target_frame": record["target_frame"],
                        "anchor_bin": record["anchor_bin"],
                        "target_bin": record["target_bin"],
                    }
                )
            payload = make_shard_payload(
                shard,
                new_version,
                mapping_to_records(shard, mapping),
            )
            shards[shard] = self.index.save_shard(payload, crashable=True)
        fingerprints = self.current.fingerprint_map
        fingerprints[track_id] = fingerprint.content_hash
        members = list(self.current.members)
        members.append(
            Member(
                track_id=track_id,
                audio_sha256=audio_hash,
                duration_samples=int(len(samples)),
                display_name=display_name or Path(source_path).name,
            )
        )
        provisional = LibraryState.create(
            version=new_version,
            members=members,
            fingerprint_config=self.current.fingerprint_config,
            similarity_threshold=self.current.similarity_threshold,
            fingerprints=fingerprints,
            manifest_hash="",
        )
        manifest_payload = make_manifest_payload(
            version=new_version,
            state_digest=provisional.state_digest,
            fingerprint_config_digest=self.current.fingerprint_config.digest,
            fingerprints=fingerprints,
            member_count=len(members),
            shards=shards,
        )
        _, manifest_hash = self.index.save_manifest(manifest_payload)
        state = LibraryState.create(
            version=new_version,
            members=members,
            fingerprint_config=self.current.fingerprint_config,
            similarity_threshold=self.current.similarity_threshold,
            fingerprints=fingerprints,
            manifest_hash=manifest_hash,
        )
        return self._install_state(
            state,
            {
                "type": "import",
                "track_id": track_id,
                "audio_sha256": audio_hash,
                "fingerprint_hash": fingerprint.content_hash,
                "affected_shards": sorted(grouped),
            },
        )

    def delete_track(self, track_id: str) -> LibraryState:
        if track_id not in self.current.member_map:
            raise WorkbenchError("E_NOT_FOUND", "track is not in the current state", {"track_id": track_id})
        fingerprint_hash = self.current.fingerprint_map[track_id]
        document = self.load_fingerprint(fingerprint_hash)
        records = self._records_for_fingerprint(track_id, document)
        grouped = self._group_records_by_shard(records)
        new_version = self.current.version + 1
        old_manifest = self._manifest(self.current.manifest_hash)
        shards = dict(old_manifest.get("shards", {}))
        self.index.begin_commit()
        for shard in sorted(grouped):
            reference = shards.get(shard)
            if reference is None:
                continue
            mapping = entries_to_mapping(self.index.load_shard(reference, shard)["entries"])
            for fingerprint_key in list(mapping):
                occurrences = [
                    occurrence
                    for occurrence in mapping[fingerprint_key]
                    if occurrence["track_id"] != track_id
                ]
                if occurrences:
                    mapping[fingerprint_key] = occurrences
                else:
                    del mapping[fingerprint_key]
            if not mapping:
                shards.pop(shard, None)
                continue
            payload = make_shard_payload(
                shard,
                new_version,
                mapping_to_records(shard, mapping),
            )
            shards[shard] = self.index.save_shard(payload, crashable=True)
        fingerprints = self.current.fingerprint_map
        fingerprints.pop(track_id)
        members = [member for member in self.current.members if member.track_id != track_id]
        provisional = LibraryState.create(
            version=new_version,
            members=members,
            fingerprint_config=self.current.fingerprint_config,
            similarity_threshold=self.current.similarity_threshold,
            fingerprints=fingerprints,
            manifest_hash="",
        )
        manifest_payload = make_manifest_payload(
            version=new_version,
            state_digest=provisional.state_digest,
            fingerprint_config_digest=self.current.fingerprint_config.digest,
            fingerprints=fingerprints,
            member_count=len(members),
            shards=shards,
        )
        _, manifest_hash = self.index.save_manifest(manifest_payload)
        state = LibraryState.create(
            version=new_version,
            members=members,
            fingerprint_config=self.current.fingerprint_config,
            similarity_threshold=self.current.similarity_threshold,
            fingerprints=fingerprints,
            manifest_hash=manifest_hash,
        )
        return self._install_state(
            state,
            {
                "type": "delete",
                "track_id": track_id,
                "affected_shards": sorted(grouped),
            },
        )

    def update_config(
        self,
        *,
        fingerprint_config: FingerprintConfig | None = None,
        similarity_threshold: float | None = None,
    ) -> LibraryState:
        config = fingerprint_config or self.current.fingerprint_config
        threshold = (
            self.current.similarity_threshold
            if similarity_threshold is None
            else round(float(similarity_threshold), 6)
        )
        if not 0.0 <= threshold <= 1.0:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "similarity_threshold must be between 0 and 1",
                {"similarity_threshold": threshold},
            )
        new_version = self.current.version + 1
        old_manifest = self._manifest(self.current.manifest_hash)
        fingerprints = self.current.fingerprint_map
        affected_shards: set[str] = set()
        if config.digest != self.current.fingerprint_config.digest:
            records: list[dict[str, Any]] = []
            for member in self.current.members:
                result = self._extract_for_audio(member.audio_sha256, config)
                self._store_fingerprint(result)
                fingerprints[member.track_id] = result.content_hash
                records.extend(self._records_for_fingerprint(member.track_id, result.document))
            grouped = self._group_records_by_shard(records)
            shards: dict[str, dict[str, Any]] = {}
            affected_shards.update(old_manifest.get("shards", {}))
            affected_shards.update(grouped)
            self.index.begin_commit()
            for shard in sorted(grouped):
                payload = make_shard_payload(
                    shard,
                    new_version,
                    grouped[shard],
                )
                shards[shard] = self.index.save_shard(payload, crashable=True)
        else:
            shards = dict(old_manifest.get("shards", {}))

        provisional = LibraryState.create(
            version=new_version,
            members=self.current.members,
            fingerprint_config=config,
            similarity_threshold=threshold,
            fingerprints=fingerprints,
            manifest_hash="",
        )
        manifest_payload = make_manifest_payload(
            version=new_version,
            state_digest=provisional.state_digest,
            fingerprint_config_digest=config.digest,
            fingerprints=fingerprints,
            member_count=len(self.current.members),
            shards=shards,
        )
        _, manifest_hash = self.index.save_manifest(manifest_payload)
        state = LibraryState.create(
            version=new_version,
            members=self.current.members,
            fingerprint_config=config,
            similarity_threshold=threshold,
            fingerprints=fingerprints,
            manifest_hash=manifest_hash,
        )
        return self._install_state(
            state,
            {
                "type": "config",
                "fingerprint_config_digest": config.digest,
                "similarity_threshold": threshold,
                "affected_shards": sorted(affected_shards),
                "fingerprint_reextracted": config.digest != self.current.fingerprint_config.digest,
            },
        )

    def revert(self, target_version: int, *, operation_type: str = "revert") -> LibraryState:
        target = self.state_at(target_version)
        if not self.manifest_exists(target.manifest_hash):
            raise WorkbenchError(
                "E_SNAPSHOT_STALE",
                "target version manifest is not available",
                {"target_version": target_version},
            )
        new_version = self.current.version + 1
        state = LibraryState.create(
            version=new_version,
            members=target.members,
            fingerprint_config=target.fingerprint_config,
            similarity_threshold=target.similarity_threshold,
            fingerprints=target.fingerprint_map,
            manifest_hash=target.manifest_hash,
        )
        if state.state_digest != target.state_digest:
            raise WorkbenchError(
                "E_QUERY_DRIFT",
                "revert did not reproduce the target state digest",
                {"target_version": target_version},
            )
        return self._install_state(
            state,
            {
                "type": operation_type,
                "target_version": target_version,
                "reused_manifest_hash": target.manifest_hash,
                "shards_rewritten": 0,
            },
        )

    def undo(self) -> LibraryState:
        operation = self._load_operation(self.current.version)
        target = operation.parent_version
        if target < 0:
            raise WorkbenchError("E_STATE_CONFLICT", "initial state cannot be undone")
        return self.revert(target, operation_type="undo")

    def state_at(self, version: int) -> LibraryState:
        return self._load_operation(version).state

    def manifest_at(self, version: int) -> dict[str, Any]:
        state = self.state_at(version)
        return self.index.load_manifest(state.manifest_hash)

    def manifest_exists(self, manifest_hash: str) -> bool:
        return (self.root / "indexes" / "manifests" / f"{manifest_hash}.json").exists()

    def version_chain(self) -> list[dict[str, Any]]:
        chain = self._committed_chain()
        result = []
        for version in sorted(chain):
            operation = chain[version]
            result.append(
                {
                    "version": version,
                    "parent_version": operation.parent_version,
                    "operation": operation.operation,
                    "state_digest": operation.state.state_digest,
                    "manifest_hash": operation.state.manifest_hash,
                }
            )
        return result

    def shard_file_hashes(self, version: int | None = None) -> dict[str, str]:
        state = self.current if version is None else self.state_at(version)
        manifest = self.index.load_manifest(state.manifest_hash)
        return {
            shard: reference["content_hash"]
            for shard, reference in sorted(manifest.get("shards", {}).items())
        }

    def reconstruct_version(self, version: int) -> dict[str, Any]:
        """Rebuild historical shard/manifest hashes in memory and compare them.

        This is an audit path only. State rollback never relies on it: rollback points
        CURRENT at the historical immutable manifest and reuses its shards verbatim.
        """

        state = self.state_at(version)
        manifest = self.index.load_manifest(state.manifest_hash)
        records: list[dict[str, Any]] = []
        for member in state.members:
            fingerprint_hash = state.fingerprint_map[member.track_id]
            fingerprint = self.load_fingerprint(fingerprint_hash)
            records.extend(self._records_for_fingerprint(member.track_id, fingerprint))
        grouped = self._group_records_by_shard(records)
        expected_shards = manifest.get("shards", {})
        shard_results: list[dict[str, Any]] = []
        all_shards = sorted(set(expected_shards) | set(grouped))
        for shard in all_shards:
            reference = expected_shards.get(shard)
            if reference is None:
                reconstructed_hash = None
                match = False
                version_stamp = None
            else:
                version_stamp = int(reference["version_stamp"])
                payload = make_shard_payload(shard, version_stamp, grouped.get(shard, []))
                reconstructed_hash = digest_object(payload)
                match = reconstructed_hash == reference["content_hash"]
            shard_results.append(
                {
                    "shard": shard,
                    "version_stamp": version_stamp,
                    "expected_hash": reference["content_hash"] if reference else None,
                    "reconstructed_hash": reconstructed_hash,
                    "match": match,
                }
            )
        reconstructed_manifest_payload = make_manifest_payload(
            version=int(manifest["version"]),
            state_digest=state.state_digest,
            fingerprint_config_digest=state.fingerprint_config.digest,
            fingerprints=state.fingerprint_map,
            member_count=len(state.members),
            shards=expected_shards,
        )
        reconstructed_manifest_hash = digest_object(reconstructed_manifest_payload)
        return {
            "version": version,
            "manifest_version": int(manifest["version"]),
            "expected_index_hash": state.manifest_hash,
            "reconstructed_index_hash": reconstructed_manifest_hash,
            "index_hash_match": reconstructed_manifest_hash == state.manifest_hash,
            "all_shards_match": all(item["match"] for item in shard_results),
            "shards": shard_results,
            "write_isolation": "audit_only_no_files_written",
        }
