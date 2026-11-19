"""Interactive view model, staleness tagging and authoritative consistency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config.compiler import CompiledVersion
from ..config.manifest import ManifestVersion
from ..errors import E_VIEW, ErrorDetail, PlatformError
from ..runtime.converge import ConvergenceReport
from ..util.canonical import canonical_bytes, digest


@dataclass
class ViewRow:
    key: str
    version: int
    derived_summary: str          # canonical text of the fixed value
    verdict: str                  # converged | divergent | stale
    stale: bool
    source_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "version": self.version,
            "derived_summary": self.derived_summary,
            "verdict": self.verdict,
            "stale": self.stale,
            "source_version": self.source_version,
        }


def build_view(
    manifest_version: ManifestVersion,
    compiled: CompiledVersion,
    convergence: ConvergenceReport,
) -> list[ViewRow]:
    divergent_nodes = {node.node_id for node in convergence.nodes if not node.match}
    rows: list[ViewRow] = []
    for key in manifest_version.order:
        item = manifest_version.items[key]
        fixed = compiled.values[key]
        verdict = "converged"
        if divergent_nodes & set(item.targets):
            verdict = "divergent"
        rows.append(
            ViewRow(
                key=key,
                version=manifest_version.version,
                derived_summary=fixed.decimal_text(),
                verdict=verdict,
                stale=False,
                source_version=manifest_version.version,
            )
        )
    return rows


def view_payload(rows: list[ViewRow]) -> dict[str, Any]:
    payload = {"rows": [row.to_dict() for row in rows]}
    payload["view_digest"] = digest(payload)
    return payload


def mark_stale(rows: list[ViewRow], current_version: int) -> list[ViewRow]:
    """Any row citing an older parameter version is stale and leaves the report."""
    fresh: list[ViewRow] = []
    for row in rows:
        if row.source_version != current_version:
            row = ViewRow(
                key=row.key,
                version=row.version,
                derived_summary=row.derived_summary,
                verdict="stale",
                stale=True,
                source_version=row.source_version,
            )
        fresh.append(row)
    return fresh


def verify_view(
    displayed: list[ViewRow],
    authoritative: list[ViewRow],
    current_version: int,
) -> None:
    """Byte-level comparison; stale rows never participate in the current report."""
    details: list[ErrorDetail] = []
    displayed_fresh = [row for row in displayed if not row.stale and row.source_version == current_version]
    auth_by_key = {row.key: row for row in authoritative}

    for row in displayed_fresh:
        expected = auth_by_key.get(row.key)
        if expected is None:
            details.append(
                ErrorDetail(reason="view row has no authoritative counterpart", key=row.key, source="view")
            )
            continue
        if canonical_bytes(row.to_dict()) != canonical_bytes(expected.to_dict()):
            details.append(
                ErrorDetail(
                    reason="displayed view differs byte-for-byte from authoritative replay",
                    key=row.key,
                    source="view",
                )
            )
    PlatformError.raise_if(E_VIEW, "interactive view is inconsistent with authoritative replay", details)
