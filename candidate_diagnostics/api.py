from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config, load_config
from .engine import EngineRun, IncrementalEngine
from .formats import result_to_records, serialize_v1
from .loader import load_requests_jsonl
from .models import Request


class CandidateDiagnostics:
    """Top-level integrable component for truncation, sampling and ledgers."""

    def __init__(self, config: Config | dict[str, Any] | str | Path):
        if isinstance(config, Config):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = Config(config)
        self.engine = IncrementalEngine(self.config)

    @classmethod
    def from_files(
        cls, config_path: str | Path, candidates_path: str | Path
    ) -> tuple["CandidateDiagnostics", dict[str, Request], list[Any]]:
        component = cls(load_config(config_path))
        requests, failures = load_requests_jsonl(candidates_path, component.config)
        return component, requests, failures

    def process(self, requests: dict[str, Request] | list[Request]) -> EngineRun:
        return self.engine.run(requests)

    def process_file(self, candidates_path: str | Path) -> tuple[EngineRun, list[Any]]:
        requests, failures = load_requests_jsonl(candidates_path, self.config)
        return self.process(requests), failures

    def v1_lines_for(self, run: EngineRun, request_id: str) -> list[str]:
        return list(run.v1_lines[request_id])

    def v2_records_for(self, run: EngineRun, request_id: str) -> list[dict[str, Any]]:
        return list(run.records[request_id])

    def all_v1_lines(self, run: EngineRun) -> list[str]:
        return [
            line
            for request_id in sorted(run.v1_lines)
            for line in run.v1_lines[request_id]
        ]

    def all_v2_records(self, run: EngineRun) -> list[dict[str, Any]]:
        return [
            record
            for request_id in sorted(run.records)
            for record in run.records[request_id]
        ]


def analyze_jsonl_batch(
    config_path: str | Path, candidates_path: str | Path
) -> dict[str, Any]:
    component, requests, failures = CandidateDiagnostics.from_files(
        config_path, candidates_path
    )
    run = component.process(requests)
    return {
        "run": run,
        "failures": failures,
        "v1_lines": component.all_v1_lines(run),
        "v2_records": component.all_v2_records(run),
    }


__all__ = [
    "CandidateDiagnostics",
    "analyze_jsonl_batch",
    "result_to_records",
    "serialize_v1",
]
