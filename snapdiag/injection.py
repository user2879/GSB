"""Deterministic anomaly injection and recovery helpers."""

from __future__ import annotations

import json
from typing import Any

from .errors import DiagnosticError, E_FORMAT, E_INTEGRITY, E_STATE, E_VERSION
from .geometry import hex_preview
from .store import Session
from .util import atomic_write_bytes, atomic_write_json, ensure_writable, sha256_bytes


INJECTION_KINDS = ("block-corruption", "manifest-truncation", "version-conflict")


class InjectionManager:
    def __init__(self, session: Session):
        self.session = session

    def choose_snapshot_with_refs(self, target: str | None = None) -> tuple[str, int]:
        if target is not None:
            self.session.snapshot(target)
            refs = self.session.snapshot_blocks(target)
            if not refs:
                raise DiagnosticError(
                    E_STATE,
                    "target snapshot has no data references to corrupt",
                    snapshot_id=target,
                    field="block_refs",
                )
            return target, min(refs)
        for row in reversed(self.session.snapshot_rows()):
            snapshot_id = str(row["snapshot_id"])
            refs = self.session.snapshot_blocks(snapshot_id)
            if refs:
                return snapshot_id, min(refs)
        raise DiagnosticError(
            E_STATE,
            "no snapshot has data references to corrupt",
            field="snapshot_id",
        )

    def corrupt_block_data(
        self,
        *,
        target: str | None = None,
        block_index: int | None = None,
    ) -> dict[str, Any]:
        target_snapshot, chosen_block = self.choose_snapshot_with_refs(target)
        if block_index is not None:
            refs = self.session.snapshot_blocks(target_snapshot)
            if block_index not in refs:
                raise DiagnosticError(
                    E_STATE,
                    "requested block is not referenced by the target snapshot",
                    snapshot_id=target_snapshot,
                    field="block_index",
                    details={"block_index": block_index},
                )
            chosen_block = block_index
        item = self.session.snapshot_blocks(target_snapshot)[chosen_block]
        ref = item["ref"]
        object_path = self.session._object_path(ref)
        if not object_path.exists():
            raise DiagnosticError(
                E_INTEGRITY,
                "injection target object is already missing",
                snapshot_id=target_snapshot,
                field="ref",
                details={"block_index": chosen_block, "expected_digest": ref},
            )
        ensure_writable(object_path)
        data = bytearray(object_path.read_bytes())
        if not data:
            data = bytearray(b"\x00")
        position = (chosen_block * 17) % len(data)
        data[position] ^= 0xA5
        atomic_write_bytes(object_path, bytes(data))
        actual = sha256_bytes(bytes(data))
        result = {
            "ok": True,
            "operation": "inject",
            "kind": "block-corruption",
            "snapshot_id": target_snapshot,
            "block_index": chosen_block,
            "ref": f"sha256:{ref}",
            "expected_digest": f"sha256:{ref}",
            "actual_digest": f"sha256:{actual}",
            "preview": hex_preview(bytes(data)),
            "expected_business_code": E_INTEGRITY,
        }
        self.session.logger.emit(
            "anomaly_injected",
            code=E_INTEGRITY,
            operation="inject",
            snapshot_id=target_snapshot,
            block_index=chosen_block,
            expected=f"sha256:{ref}",
            actual=f"sha256:{actual}",
            kind="block-corruption",
        )
        return result

    def truncate_manifest(self, *, keep_ratio: float = 0.5) -> dict[str, Any]:
        if not self.session.manifest_path.exists():
            raise DiagnosticError(
                E_FORMAT,
                "manifest is missing before truncation injection",
                snapshot_id="<manifest>",
                field="file",
            )
        self.session._backup_manifest("before-manifest-truncation")
        data = self.session.manifest_path.read_bytes()
        keep = max(1, min(len(data) - 1, int(len(data) * keep_ratio)))
        marker = data.find(b'"snapshot_id"')
        if marker >= 0:
            # Retain the first identifier so E_FORMAT can name the affected
            # snapshot even though the JSON document is deliberately cut.
            keep = max(keep, min(len(data) - 1, marker + 64))
        truncated = data[:keep]
        ensure_writable(self.session.manifest_path)
        atomic_write_bytes(self.session.manifest_path, truncated)
        result = {
            "ok": True,
            "operation": "inject",
            "kind": "manifest-truncation",
            "manifest_path": str(self.session.manifest_path),
            "original_size": len(data),
            "truncated_size": len(truncated),
            "expected_business_code": E_FORMAT,
        }
        self.session.logger.emit(
            "anomaly_injected",
            code=E_FORMAT,
            operation="inject",
            field="json",
            kind="manifest-truncation",
            original_size=len(data),
            truncated_size=len(truncated),
        )
        return result

    def inject_version_conflict(self, *, conflicting_version: str = "m2") -> dict[str, Any]:
        try:
            document = json.loads(self.session.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DiagnosticError(
                E_FORMAT,
                "cannot inject a version conflict into a malformed manifest",
                snapshot_id="<manifest>",
                field="json",
                details={"reason": str(exc)},
            )
        self.session._backup_manifest("before-version-conflict")
        original = document.get("format_version")
        document["format_version"] = conflicting_version
        atomic_write_json(self.session.manifest_path, document)
        ensure_writable(self.session.manifest_path)
        result = {
            "ok": True,
            "operation": "inject",
            "kind": "version-conflict",
            "expected_version": self.session.meta().get("manifest_format", "m3"),
            "conflicting_version": conflicting_version,
            "original_version": original,
            "expected_business_code": E_VERSION,
        }
        self.session.logger.emit(
            "anomaly_injected",
            code=E_VERSION,
            operation="inject",
            field="format_version",
            kind="version-conflict",
            expected=self.session.meta().get("manifest_format", "m3"),
            actual=conflicting_version,
        )
        return result

    def inject(self, kind: str, *, target: str | None = None, block_index: int | None = None) -> dict[str, Any]:
        if kind == "block-corruption":
            return self.corrupt_block_data(target=target, block_index=block_index)
        if kind == "manifest-truncation":
            return self.truncate_manifest()
        if kind == "version-conflict":
            return self.inject_version_conflict()
        raise DiagnosticError(
            E_FORMAT,
            "unknown injection kind",
            field="inject",
            details={"actual": kind, "supported": list(INJECTION_KINDS)},
        )

    def diagnose(self, kind: str, *, target: str | None = None) -> dict[str, Any]:
        observed_code: str
        details: dict[str, Any] = {}
        try:
            if kind == "block-corruption":
                if target is None:
                    target, _ = self.choose_snapshot_with_refs(None)
                self.session.recover_session()
                self.session.rollback(target, path="auto", retry=True)
                observed_code = "NONE"
                details = {"unexpected": "corruption was not detected"}
            elif kind == "manifest-truncation":
                self.session.read_manifest(strict_version=True)
                observed_code = "NONE"
                details = {"unexpected": "truncation was not detected"}
            elif kind == "version-conflict":
                self.session.read_manifest(strict_version=True)
                observed_code = "NONE"
                details = {"unexpected": "version conflict was not detected"}
            else:
                raise DiagnosticError(E_FORMAT, "unknown injection kind", field="inject")
        except DiagnosticError as error:
            observed_code = error.code
            details = error.to_dict()
            self.session.logger.emit_error(error, operation=f"diagnose-injection:{kind}")
        result = {
            "ok": True,
            "operation": "diagnose-injection",
            "kind": kind,
            "observed_code": observed_code,
            "details": details,
        }
        return result

    def restore_manifest(self) -> dict[str, Any]:
        report = self.session.reconcile_manifest(reason="injection-repair")
        self.session.logger.emit(
            "manifest_recovered",
            operation="repair",
            backup=report.get("backup"),
            manifest_digest=report.get("new_manifest_digest"),
        )
        return report

    def repair_and_verify(self, *, target: str | None = None) -> dict[str, Any]:
        objects = self.session.repair_objects()
        manifest = self.restore_manifest()
        verification: dict[str, Any] = {"ok": True, "checked_blocks": 0, "issues": []}
        rollback: dict[str, Any] | None = None
        if target is not None or self.session.snapshot_rows():
            if target is None:
                target = str(self.session.snapshot_rows()[-1]["snapshot_id"])
            result = self.session.rollback(target, path="auto", retry=True)
            rollback = result.as_dict()
            verification = dict(result.verification)
        report = {
            "ok": bool(objects["ok"] and verification["ok"]),
            "operation": "repair-and-verify",
            "objects": objects,
            "manifest": manifest,
            "rollback": rollback,
            "block_verification": verification,
            "recovered": bool(objects["ok"] and verification["ok"]),
        }
        self.session.logger.emit(
            "recovery_verified" if report["ok"] else "recovery_failed",
            code=None if report["ok"] else E_INTEGRITY,
            operation="repair",
            recovered=report["recovered"],
            checked_blocks=verification.get("checked_blocks", 0),
        )
        return report
