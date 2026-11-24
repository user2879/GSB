from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .canonical import canonical_dumps, sha256_text


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_pointer = self.root / "CURRENT"

    def _write_json(self, path: Path, value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
        return sha256_text(text)

    def load_current(self) -> Optional[Dict[str, Any]]:
        candidate_dir: Optional[Path] = None
        fallback_from_corrupt_pointer = False
        if self.current_pointer.exists():
            name = self.current_pointer.read_text(encoding="utf-8").strip()
            pointer_dir = self.root / name
            if (pointer_dir / "manifest.json").exists():
                candidate_dir = pointer_dir
                try:
                    json.loads((pointer_dir / "manifest.json").read_text(encoding="utf-8"))
                except Exception:
                    candidate_dir = None
                    fallback_from_corrupt_pointer = True

        if candidate_dir is None:
            manifests = sorted(self.root.glob("snapshot-v*/manifest.json"))
            valid = []
            for manifest_path in manifests:
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    int(manifest["global_version"])
                    valid.append((int(manifest["global_version"]), manifest_path.parent))
                except Exception:
                    continue
            if not valid:
                return None
            candidate_dir = max(valid, key=lambda item: item[0])[1]

        try:
            manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
        except Exception:
            return None
        artifacts: Dict[str, Any] = {}
        for name in ("blocks", "buckets", "ledger", "recall", "cache", "view"):
            path = candidate_dir / f"{name}.json"
            if path.exists():
                artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
        return {"snapshot_dir": str(candidate_dir), "manifest": manifest, "artifacts": artifacts}

    def create_staging_fragment(self, name: str = "manifest.json", text: str = "{truncated") -> Path:
        stage = self.root / f".staging-interrupted-{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        (stage / name).write_text(text, encoding="utf-8", newline="")
        return stage

    def commit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        previous = self.load_current()
        global_version = int(previous["manifest"]["global_version"]) + 1 if previous else 1
        stage = self.root / f".staging-{uuid.uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        artifact_versions: Dict[str, Any] = {}
        try:
            for name in ("blocks", "buckets", "ledger", "recall", "cache", "view"):
                if name not in payload:
                    continue
                text = json.dumps(payload[name], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                (stage / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")
                artifact_versions[name] = {
                    "version": global_version,
                    "sha256": sha256_text(text),
                }
            manifest = {
                "global_version": global_version,
                "config_hash": payload["config_hash"],
                "param_fingerprint": payload["param_fingerprint"],
                "quant_level": payload["quant_level"],
                "completed_blocks": payload.get("completed_blocks", []),
                "pending_blocks": payload.get("pending_blocks", []),
                "artifacts": artifact_versions,
            }
            manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            (stage / "manifest.json").write_text(manifest_text, encoding="utf-8", newline="\n")
            final_dir = self.root / f"snapshot-v{global_version:06d}"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            stage.rename(final_dir)
            temporary_pointer = self.current_pointer.with_name("CURRENT.tmp")
            temporary_pointer.write_text(final_dir.name, encoding="utf-8", newline="\n")
            temporary_pointer.replace(self.current_pointer)
            manifest["snapshot_dir"] = str(final_dir)
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    def truncate_current_manifest(self, length: int = 40) -> str:
        current = self.load_current()
        if not current:
            raise FileNotFoundError("no committed snapshot to truncate")
        path = Path(current["snapshot_dir"]) / "manifest.json"
        text = path.read_text(encoding="utf-8")
        path.write_text(text[: max(1, length)], encoding="utf-8", newline="")
        return str(path)
