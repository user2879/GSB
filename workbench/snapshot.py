from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .hashing import canonical_dumps, fingerprint, sha256_hex
from .report import phase_products, render_markdown


class SnapshotStore:
    def __init__(self, output_dir: str | Path, *, config_hash: str | None = None, fresh: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.snapshot_root = self.output_dir / "snapshots"
        self.current_pointer = self.output_dir / "current.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.existing_pointer = self._read_pointer()
        self.version_change: dict[str, Any] | None = None

        old_version = int((self.existing_pointer or {}).get("global_version", 0))
        old_hash = (self.existing_pointer or {}).get("config_hash")
        if fresh:
            self.global_version = old_version + 1
            self.version_change = {
                "resume": False,
                "reason": "fresh_requested",
                "previous_global_version": old_version or None,
                "global_version": self.global_version,
            }
        elif self.existing_pointer and config_hash and old_hash == config_hash:
            self.global_version = old_version
        elif self.existing_pointer:
            self.global_version = old_version + 1
            self.version_change = {
                "resume": False,
                "reason": "config_hash_changed",
                "previous_global_version": old_version,
                "global_version": self.global_version,
                "old_config_hash": old_hash,
                "new_config_hash": config_hash,
            }
        else:
            self.global_version = 1

    def _read_pointer(self) -> dict[str, Any] | None:
        if not self.current_pointer.exists():
            return None
        try:
            return json.loads(self.current_pointer.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def recovery_plan(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.existing_pointer:
            return {"resume": False, "reason": "no_committed_snapshot", "global_version": self.global_version}
        if self.version_change:
            return self.version_change
        cursor = int(self.existing_pointer.get("event_cursor", 0))
        old_prefix_fingerprint = self.existing_pointer.get("event_prefix_fingerprint")
        prefix = [event for event in events if int(event["seq"]) <= cursor]
        prefix_fingerprint = fingerprint(prefix)
        if old_prefix_fingerprint and old_prefix_fingerprint != prefix_fingerprint:
            self.global_version = int(self.existing_pointer.get("global_version", 0)) + 1
            self.version_change = {
                "resume": False,
                "reason": "event_prefix_changed",
                "previous_global_version": self.existing_pointer.get("global_version"),
                "global_version": self.global_version,
                "event_cursor": cursor,
            }
            return self.version_change
        return {
            "resume": True,
            "reason": "matching_global_version_and_config_hash",
            "global_version": self.global_version,
            "event_cursor": cursor,
            "replay_event_count": len(prefix),
            "event_prefix_fingerprint": prefix_fingerprint,
        }

    def commit(self, workbench: Any, *, final: bool = False) -> dict[str, Any]:
        state = workbench.state
        products = phase_products(workbench)
        cursor = int(state["cursor"])
        staging = Path(tempfile.mkdtemp(prefix=f".g{self.global_version:04d}-c{cursor:06d}-", dir=self.output_dir))
        try:
            phase_versions: dict[str, Any] = {}
            file_names = {
                "01_load_validation": "01_load_validation.json",
                "02_capacity_watermarks": "02_capacity_watermarks.json",
                "03_leases_evictions": "03_leases_evictions.json",
                "04_ledgers": "04_ledgers.json",
                "resident_snapshot": "resident_snapshot.json",
                "report": "report.json",
            }
            for key, file_name in file_names.items():
                product = products[key]
                text = json.dumps(product, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
                (staging / file_name).write_text(text, encoding="utf-8")
                phase_versions[key] = {
                    "version": phase_version_label(product.get("schema_version") if isinstance(product, dict) else None),
                    "schema_version": product.get("schema_version") if isinstance(product, dict) else None,
                    "file": file_name,
                    "sha256": sha256_hex(text),
                }

            markdown = render_markdown(products["report"])
            (staging / "report.md").write_text(markdown, encoding="utf-8")

            quota_versions = {
                node_id: int(values["quota_version"])
                for node_id, values in sorted(state["node_versions"].items())
            }
            prefix_events = [
                event for event in workbench.validation.get("events", [])
                if int(event["seq"]) <= cursor
            ]
            manifest = {
                "schema_version": "workbench.snapshot_manifest.v1",
                "global_version": self.global_version,
                "config_hash": state["config_hash"],
                "quota_version": max(quota_versions.values()) if quota_versions else 1,
                "quota_versions_by_node": quota_versions,
                "quota_version_fingerprint": fingerprint(quota_versions),
                "completed_event_cursor": cursor,
                "event_cursor": cursor,
                "processed_event_count": state["processed_event_count"],
                "event_prefix_fingerprint": fingerprint(prefix_events),
                "final": final,
                "watermark_rule_metadata": state["rules"],
                "phase_versions": phase_versions,
            }
            manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            (staging / "snapshot_manifest.json").write_text(manifest_text, encoding="utf-8")
            manifest_hash = sha256_hex(manifest_text)

            final_dir = self.snapshot_root / f"g{self.global_version:04d}-c{cursor:06d}-{manifest_hash[:12]}"
            if final_dir.exists():
                shutil.rmtree(staging, ignore_errors=True)
            else:
                import os
                os.replace(staging, final_dir)

            pointer = {
                "schema_version": "workbench.current_pointer.v1",
                "global_version": self.global_version,
                "config_hash": state["config_hash"],
                "event_cursor": cursor,
                "snapshot_dir": str(final_dir.relative_to(self.output_dir)),
                "manifest_sha256": manifest_hash,
                "committed_final": final,
            }
            pointer_text = json.dumps(pointer, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            pointer_tmp = self.current_pointer.with_suffix(".json.tmp")
            pointer_tmp.write_text(pointer_text, encoding="utf-8")
            import os
            os.replace(pointer_tmp, self.current_pointer)
            self.existing_pointer = pointer
            return {"manifest": manifest, "pointer": pointer, "snapshot_dir": final_dir}
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def read_current_report(self) -> dict[str, Any]:
        pointer = self._read_pointer()
        if not pointer:
            raise FileNotFoundError(self.current_pointer)
        report_path = self.output_dir / pointer["snapshot_dir"] / "report.json"
        return json.loads(report_path.read_text(encoding="utf-8"))


def phase_version_label(value: Any) -> str:
    return str(value or "unversioned")
