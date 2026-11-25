"""Stage snapshots with temp-directory commit and atomic pointer switch."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .codes import Code
from .engine import IncrementalEngine
from .io_utils import canonical_dumps


STAGE_FILES = {
    "validation": "01_validation.json",
    "features": "02_features.json",
    "emission": "03_emission.json",
    "viterbi": "04_viterbi.json",
    "merge": "05_merge.json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class SnapshotStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.snapshots_dir = self.workspace / "snapshots"
        self.current_pointer = self.workspace / "current.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def save(self, engine: IncrementalEngine, phase: str = "stable") -> dict[str, Any]:
        engine.snapshot_version += 1
        commit_suffix = uuid.uuid4().hex[:8]
        snapshot_id = f"snapshot-{engine.snapshot_version:08d}-cursor-{engine.command_cursor:08d}-{commit_suffix}"
        staging = self.workspace / f".staging-{snapshot_id}-{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True, exist_ok=False)

        try:
            validation = {
                "corpus_failures": engine.corpus_failures,
                "lexicon_failures": engine.lexicon_failures,
                "command_failures": engine.command_failures,
                "business_counts": engine.business_counts(),
            }
            self._write_stage(staging, STAGE_FILES["validation"], validation)
            for stage in ("features", "emission", "viterbi", "merge"):
                envelopes = [envelope for envelope in engine.cache.values() if envelope.get("stage") == stage]
                envelopes.sort(key=lambda item: (item.get("sample_id", ""), item.get("cache_key", "")))
                self._write_stage(staging, STAGE_FILES[stage], {"stage": stage, "envelopes": envelopes})

            state = engine.to_persistent_dict()
            state["phase"] = phase
            self._write_stage(staging, "state.json", state)

            file_hashes = {
                name: _sha256_file(staging / filename)
                for name, filename in [*STAGE_FILES.items(), ("state", "state.json")]
            }
            manifest = {
                "manifest_schema": "ner-snapshot-manifest-v1",
                "snapshot_id": snapshot_id,
                "snapshot_format_version": engine.config.snapshot_format_version if hasattr(engine.config, "snapshot_format_version") else 1,
                "snapshot_version": engine.snapshot_version,
                "command_cursor": engine.command_cursor,
                "generation": engine.generation,
                "phase": phase,
                "model_version": engine.package.get("model_version"),
                "config_hash": engine.config_hash(),
                "file_hashes": file_hashes,
                "canonical_hash": hashlib.sha256(canonical_dumps(file_hashes).encode("utf-8")).hexdigest(),
            }
            self._write_stage(staging, "manifest.json", manifest)
            manifest_hash = _sha256_file(staging / "manifest.json")

            final_dir = self.snapshots_dir / snapshot_id
            os.replace(staging, final_dir)
            pointer = {
                "snapshot_id": snapshot_id,
                "snapshot_dir": str(Path("snapshots") / snapshot_id),
                "manifest_hash": manifest_hash,
                "canonical_hash": manifest["canonical_hash"],
                "snapshot_format_version": manifest["snapshot_format_version"],
                "model_version": manifest["model_version"],
                "config_hash": manifest["config_hash"],
            }
            pointer_tmp = self.current_pointer.with_name(f".current.json.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
            _write_json(pointer_tmp, pointer)
            os.replace(pointer_tmp, self.current_pointer)
            return {"ok": True, "action": "committed", "snapshot_id": snapshot_id, "manifest": manifest, "pointer": pointer}
        except Exception:
            raise

    def _write_stage(self, staging: Path, filename: str, value: Any) -> None:
        _write_json(staging / filename, value)

    def load(
        self,
        expected_model_version: str | None = None,
        expected_config_hash: str | None = None,
        strict_config_hash: bool = False,
    ) -> dict[str, Any]:
        if not self.current_pointer.exists():
            return {"ok": False, "engine": None, "reason_code": Code.SNAPSHOT_REBUILD, "reason": "current.json 不存在"}
        try:
            pointer = json.loads(self.current_pointer.read_text(encoding="utf-8"))
            snapshot_dir = self.workspace / pointer["snapshot_dir"]
            manifest_path = snapshot_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _sha256_file(manifest_path) != pointer.get("manifest_hash"):
                return {"ok": False, "engine": None, "reason_code": Code.SNAPSHOT_CORRUPT, "reason": "manifest 指针哈希不匹配"}
            if manifest.get("canonical_hash") != pointer.get("canonical_hash"):
                return {"ok": False, "engine": None, "reason_code": Code.SNAPSHOT_CORRUPT, "reason": "manifest 规范哈希不匹配"}
            for name, expected_hash in manifest["file_hashes"].items():
                filename = "state.json" if name == "state" else STAGE_FILES[name]
                path = snapshot_dir / filename
                if not path.exists() or _sha256_file(path) != expected_hash:
                    return {"ok": False, "engine": None, "reason_code": Code.SNAPSHOT_CORRUPT, "reason": f"阶段文件哈希不匹配：{filename}"}
            if manifest.get("snapshot_format_version") != 1:
                return {"ok": False, "engine": None, "reason_code": Code.SNAPSHOT_VERSION_MISMATCH, "reason": "snapshot_format_version 不兼容", "manifest": manifest}
            if expected_model_version is not None and manifest.get("model_version") != expected_model_version:
                return {"ok": False, "engine": None, "reason_code": Code.SNAPSHOT_VERSION_MISMATCH, "reason": "model_version 已变化", "manifest": manifest}
            if strict_config_hash and expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
                return {"ok": False, "engine": None, "reason_code": Code.SNAPSHOT_VERSION_MISMATCH, "reason": "config_hash 与启动配置不一致", "manifest": manifest}

            state = json.loads((snapshot_dir / "state.json").read_text(encoding="utf-8"))
            cache_by_stage: dict[str, list[dict[str, Any]]] = {}
            for stage, filename in STAGE_FILES.items():
                if stage == "validation":
                    continue
                data = json.loads((snapshot_dir / filename).read_text(encoding="utf-8"))
                cache_by_stage[stage] = data.get("envelopes", [])
            engine = IncrementalEngine.from_persistent_dict(state, cache_by_stage)
            return {"ok": True, "engine": engine, "manifest": manifest, "pointer": pointer, "action": "resume"}
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return {"ok": False, "engine": None, "reason_code": Code.SNAPSHOT_CORRUPT, "reason": f"快照无法读取：{type(exc).__name__}: {exc}"}

    def recover_or_create(
        self,
        package: dict[str, Any],
        config: Any,
        samples: list[dict[str, Any]],
        lexicon: list[dict[str, str]],
        corpus_failures: list[dict[str, Any]] | None = None,
        lexicon_failures: list[dict[str, Any]] | None = None,
    ) -> tuple[IncrementalEngine, dict[str, Any]]:
        loaded = self.load(expected_model_version=package.get("model_version"))
        if loaded["ok"]:
            return loaded["engine"], {"recovery_action": "resume", "code": Code.OK, **{k: loaded[k] for k in ("manifest", "pointer")}}
        engine = IncrementalEngine(
            package=package,
            config=config,
            samples=samples,
            lexicon=lexicon,
            corpus_failures=corpus_failures,
            lexicon_failures=lexicon_failures,
        )
        saved = self.save(engine, phase="fresh")
        return engine, {"recovery_action": "rebuild", "code": loaded["reason_code"], "reason": loaded["reason"], "commit": saved}
