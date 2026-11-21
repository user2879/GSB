from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .util import digest_json, digest_file


@dataclass(frozen=True)
class PipelineConfig:
    declaration_version: int
    format_version: int
    default_silence_window_ms: int
    seed: int
    batch_size: int
    global_concurrency: int
    per_service_concurrency: int
    services_path: Path
    rules_path: Path
    alerts_path: Path
    database_path: Path
    log_path: Path
    output_dir: Path

    def with_runtime_paths(
        self,
        *,
        database_path: str | Path | None = None,
        log_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> "PipelineConfig":
        return replace(
            self,
            database_path=Path(database_path) if database_path else self.database_path,
            log_path=Path(log_path) if log_path else self.log_path,
            output_dir=Path(output_dir) if output_dir else self.output_dir,
        )

    def semantic_payload(self, dataset: Any) -> dict[str, Any]:
        return {
            "pipeline_schema": "alert-pipeline/config/v1",
            "format_version": self.format_version,
            "default_silence_window_ms": self.default_silence_window_ms,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "services_digest": digest_file(self.services_path),
            "rules_digest": digest_file(self.rules_path),
            "alerts_digest": digest_file(self.alerts_path),
            "graph_version": dataset.graph.version,
            "rules_version": dataset.rules.version,
        }

    def semantic_digest(self, dataset: Any) -> str:
        return digest_json(self.semantic_payload(dataset))

    def public_dict(self, dataset: Any) -> dict[str, Any]:
        return {
            "declaration_version": self.declaration_version,
            "format_version": self.format_version,
            "default_silence_window_ms": self.default_silence_window_ms,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "services_path": str(self.services_path.resolve()),
            "rules_path": str(self.rules_path.resolve()),
            "alerts_path": str(self.alerts_path.resolve()),
            "graph_version": dataset.graph.version,
            "rules_version": dataset.rules.version,
            "semantic_digest": self.semantic_digest(dataset),
        }
