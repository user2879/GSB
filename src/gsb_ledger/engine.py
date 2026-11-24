"""End-to-end deterministic ledger engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from .attribution import attribute_receipts
from .config import config_hash, load_config, load_requests, requests_hash
from .conversions import Sidecar, convert_receipt
from .errors import E_CONVERSION_LOSS, Issue, issue_from_dict
from .errors import E_RESPONSE_BUDGET_EXCEEDED
from .faults import FaultInjector, load_fault_injector
from .ledger import LedgerBuild, assemble_ledger
from .model import AppConfig, Receipt, RequestSpec
from .receipts import ReceiptLoadResult, load_receipts, receipt_from_dict
from .resources import (
    ResourceMetrics,
    compute_resource,
    resource_cache_key,
)
from .snapshot import SnapshotStore
from .util import digest_value, sha256_file
from .views import compare_views, render_view


PARSER_SCHEMA_VERSION = 1


@dataclass
class EngineResult:
    ledger: dict[str, Any]
    ledger_version: str
    build_info: dict[str, Any]

    @property
    def entries(self) -> list[dict[str, Any]]:
        return self.ledger["entries"]


class LedgerEngine:
    def __init__(
        self,
        project_root: str | Path,
        *,
        config_path: str = "config.json",
        requests_path: str = "requests.ndjson",
        state_dir: str = ".state",
        fault_config: str | None = None,
        command_cursor: int = 0,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config_path = self._resolve(config_path)
        self.requests_path = self._resolve(requests_path)
        self.state_dir = self._resolve(state_dir)
        self.fault_config_override = fault_config
        self.command_cursor = int(command_cursor)
        self.store = SnapshotStore(self.state_dir)
        self.config: AppConfig | None = None
        self.requests: tuple[RequestSpec, ...] | None = None

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def load(self) -> tuple[AppConfig, tuple[RequestSpec, ...]]:
        config = load_config(self.config_path)
        requests = load_requests(self.requests_path, batch_size=config.batch_size)
        self.config = config
        self.requests = requests
        return config, requests

    def _fault_path(self, config: AppConfig) -> Path | None:
        value = self.fault_config_override or config.fault_config
        if not value:
            return None
        return self._resolve(value)

    def _fault_hash(self, path: Path | None) -> str:
        return "missing" if path is None or not path.exists() else sha256_file(path)

    def _receipt_fingerprints(self, config: AppConfig) -> dict[str, str]:
        result: dict[str, str] = {}
        for source in config.receipt_sources:
            path = self._resolve(source.path)
            result[source.path] = sha256_file(path) if path.exists() else "missing"
        return dict(sorted(result.items()))

    def _global_version(
        self,
        *,
        config: AppConfig,
        requests: tuple[RequestSpec, ...],
        receipt_fingerprints: dict[str, str],
        fault_hash: str,
        render_version: str | None,
    ) -> str:
        return digest_value(
            {
                "parser_schema_version": PARSER_SCHEMA_VERSION,
                "config_hash": config_hash(config),
                "requests_hash": requests_hash(requests),
                "receipt_fingerprints": receipt_fingerprints,
                "fault_config_hash": fault_hash,
                "render_version": render_version,
            }
        )

    def status(self, *, render_version: str | None = None) -> dict[str, Any]:
        config, requests = self.load()
        fault_path = self._fault_path(config)
        fault_hash = self._fault_hash(fault_path)
        receipts_fp = self._receipt_fingerprints(config)
        global_version = self._global_version(
            config=config,
            requests=requests,
            receipt_fingerprints=receipts_fp,
            fault_hash=fault_hash,
            render_version=render_version,
        )
        result = self.store.resume_status(
            global_version=global_version,
            config_hash=config_hash(config),
            command_cursor=self.command_cursor,
        )
        result.update(
            {
                "global_version": global_version,
                "config_hash": config_hash(config),
                "requests_hash": requests_hash(requests),
                "receipt_fingerprints": receipts_fp,
                "fault_config_hash": fault_hash,
                "state_dir": str(self.state_dir),
            }
        )
        return result

    def _parse_cache_key(
        self,
        *,
        config: AppConfig,
        receipt_fingerprints: dict[str, str],
        fault_hash: str,
    ) -> str:
        return digest_value(
            {
                "parser_schema_version": PARSER_SCHEMA_VERSION,
                "sources": [
                    {
                        "path": source.path,
                        "default_worker_id": source.default_worker_id,
                        "default_param_version": source.default_param_version,
                        "format": source.format,
                    }
                    for source in config.receipt_sources
                ],
                "fingerprints": receipt_fingerprints,
                "fault_config_hash": fault_hash,
                "known_worker_versions": {
                    worker.worker_id: list(
                        worker.known_param_versions or (worker.param_version,)
                    )
                    for worker in config.workers
                },
            }
        )

    def _load_parse_cached(
        self,
        key: str,
        *,
        injector: FaultInjector | None,
    ) -> ReceiptLoadResult | None:
        payload = self.store.load_cache("parse", key)
        if payload is None:
            return None
        try:
            receipts = tuple(receipt_from_dict(item) for item in payload.get("receipts", []))
            issues = tuple(issue_from_dict(item) for item in payload.get("issues", []))
            fingerprints = {
                str(name): str(value)
                for name, value in payload.get("file_fingerprints", {}).items()
            }
            files = tuple(str(item) for item in payload.get("files", []))
            result = ReceiptLoadResult(
                receipts=receipts,
                issues=list(issues),
                file_fingerprints=fingerprints,
                files=files,
            )
            return result
        except (KeyError, TypeError, ValueError):
            return None

    def _save_parse_cache(self, key: str, result: ReceiptLoadResult) -> None:
        self.store.save_cache(
            "parse",
            key,
            {
                "schema_version": PARSER_SCHEMA_VERSION,
                "receipts": [receipt.to_dict() for receipt in result.receipts],
                "issues": [issue.to_dict() for issue in result.issues],
                "file_fingerprints": result.file_fingerprints,
                "files": list(result.files),
            },
        )

    def _apply_render_version(
        self,
        receipts: tuple[Receipt, ...],
        render_version: str | None,
    ) -> tuple[tuple[Receipt, ...], Sidecar | None, tuple[dict[str, Any], ...], list[Issue]]:
        if render_version is None:
            return receipts, None, (), []
        order = {"r1": 1, "r2": 2, "r3": 3}
        if render_version not in order:
            raise ValueError("render_version must be r1, r2 or r3")
        sidecar = Sidecar()
        converted: list[Receipt] = []
        degradations: list[dict[str, Any]] = []
        issues: list[Issue] = []
        for receipt in receipts:
            if order[render_version] >= order[receipt.version]:
                converted.append(receipt)
                continue
            down = convert_receipt(receipt, render_version, sidecar=sidecar)
            up = convert_receipt(
                down.receipt,
                receipt.version,
                sidecar=sidecar,
                source_version=render_version,
            )
            converted.append(up.receipt)
            if down.downgrade is not None:
                degradations.append(down.downgrade.to_dict())
            if up.lost_fields:
                issues.append(
                    Issue(
                        code=E_CONVERSION_LOSS,
                        message=(
                            "down/up conversion could not restore: "
                            + ", ".join(up.lost_fields)
                        ),
                        path=receipt.source_path,
                        line=receipt.source_line,
                        item=receipt.key,
                        context={"lost_fields": list(up.lost_fields)},
                    )
                )
        return tuple(converted), sidecar, tuple(degradations), issues

    def _compute_resources(
        self,
        batches,
        config: AppConfig,
    ) -> tuple[tuple[ResourceMetrics, ...], dict[str, Any]]:
        metrics: list[ResourceMetrics] = []
        hits: list[str] = []
        misses: list[str] = []
        for batch in sorted(batches, key=lambda item: (item.batch_id, item.worker_id)):
            cache_key = resource_cache_key(batch, config)
            cached = None if batch.invalidated else self.store.load_cache("resources", cache_key)
            if cached is not None:
                try:
                    metric = ResourceMetrics.from_dict(cached)
                    if (
                        metric.cache_key != cache_key
                        or metric.fixed_point_bits != config.fixed_point_bits
                        or metric.batch_id != batch.batch_id
                        or metric.worker_id != batch.worker_id
                    ):
                        raise ValueError("resource cache key or metadata mismatch")
                    metrics.append(metric)
                    hits.append(f"{batch.batch_id}/{batch.worker_id}")
                    continue
                except (KeyError, TypeError, ValueError):
                    pass
            metric = compute_resource(batch, config)
            self.store.save_cache("resources", cache_key, metric.to_dict())
            metrics.append(metric)
            misses.append(f"{batch.batch_id}/{batch.worker_id}")
        return tuple(metrics), {
            "resource_cache": {
                "hits": sorted(hits),
                "misses": sorted(misses),
                "hit_count": len(hits),
                "miss_count": len(misses),
            }
        }

    def build(
        self,
        *,
        resume: bool = True,
        save_snapshot: bool = True,
        render_version: str | None = None,
    ) -> EngineResult:
        config, requests = self.load()
        fault_path = self._fault_path(config)
        fault_hash = self._fault_hash(fault_path)
        receipts_fp = self._receipt_fingerprints(config)
        global_version = self._global_version(
            config=config,
            requests=requests,
            receipt_fingerprints=receipts_fp,
            fault_hash=fault_hash,
            render_version=render_version,
        )
        resume_status = self.store.resume_status(
            global_version=global_version,
            config_hash=config_hash(config),
            command_cursor=self.command_cursor,
        )
        if resume:
            resumed = self.store.load_resumed_ledger(
                global_version=global_version,
                config_hash=config_hash(config),
                command_cursor=self.command_cursor,
            )
            if resumed is not None:
                ledger, manifest = resumed
                return EngineResult(
                    ledger=ledger,
                    ledger_version=ledger["ledger_version"],
                    build_info={
                        "resumed": True,
                        "config_changed": False,
                        "global_version": global_version,
                        "stage_versions": manifest.get("stage_versions", {}),
                        "command_cursor": manifest.get("command_cursor", self.command_cursor),
                        "resume_status": resume_status,
                    },
                )

        injector = (
            load_fault_injector(fault_path, project_root=self.project_root)
            if fault_path is not None
            else None
        )
        parse_key = self._parse_cache_key(
            config=config,
            receipt_fingerprints=receipts_fp,
            fault_hash=fault_hash,
        )
        receipt_load = self._load_parse_cached(parse_key, injector=injector)
        parse_hit = receipt_load is not None
        if receipt_load is None:
            receipt_load = load_receipts(
                config,
                project_root=self.project_root,
                injector=injector,
            )
            self._save_parse_cache(parse_key, receipt_load)

        rendered_receipts, sidecar, degradations, conversion_issues = self._apply_render_version(
            receipt_load.receipts, render_version
        )
        if rendered_receipts is not receipt_load.receipts:
            receipt_load = ReceiptLoadResult(
                receipts=rendered_receipts,
                issues=list(receipt_load.issues),
                file_fingerprints=receipt_load.file_fingerprints,
                files=receipt_load.files,
            )
        attribution = attribute_receipts(receipt_load.receipts, requests, config)
        resources, resource_info = self._compute_resources(attribution.batches, config)
        all_issue_objects = (
            list(receipt_load.issues)
            + list(attribution.issues)
            + list(conversion_issues)
        )
        diagnostics: list[dict[str, Any]] = []
        if injector is not None:
            diagnostics = injector.reports_for(all_issue_objects)
        ledger_build: LedgerBuild = assemble_ledger(
            config=config,
            requests=requests,
            receipt_load=receipt_load,
            attribution=attribution,
            resources=resources,
            input_fingerprint=global_version,
            extra_issues=tuple(conversion_issues),
            diagnostics=tuple(diagnostics),
            degradations=degradations,
        )
        stage_payloads = {
            "parse": {
                "receipts": [receipt.to_dict() for receipt in receipt_load.receipts],
                "issues": [issue.to_dict() for issue in receipt_load.issues],
                "file_fingerprints": receipt_load.file_fingerprints,
                "files": list(receipt_load.files),
            },
            "attribution": {
                "batches": [batch.to_dict() for batch in attribution.batches],
                "decisions": [decision.to_dict() for decision in attribution.decisions],
                "issues": [issue.to_dict() for issue in attribution.issues],
                "invalid_receipts": [
                    receipt.to_dict() for receipt in attribution.invalid_receipts
                ],
                "uncovered_request_ids": list(attribution.uncovered_request_ids),
            },
            "resources": {
                "metrics": [metric.to_dict() for metric in resources],
                **resource_info,
            },
            "ledger": ledger_build.ledger,
            "conversion_sidecar": (
                sidecar.to_dict()
                if sidecar is not None
                else {"schema_version": 1, "records": {}}
            ),
        }
        committed = None
        if save_snapshot:
            committed = self.store.commit(
                global_version=global_version,
                config_hash=config_hash(config),
                requests_hash=requests_hash(requests),
                receipt_fingerprints=receipts_fp,
                fault_config_hash=fault_hash,
                command_cursor=self.command_cursor,
                stages=stage_payloads,
                extra_manifest={
                    "render_version": render_version,
                    "conversion_sidecar_version": (
                        "sha256:" + digest_value(stage_payloads["conversion_sidecar"]).split(":", 1)[1]
                    ),
                },
            )
        build_info = {
            "resumed": False,
            "config_changed": bool(resume_status.get("config_changed")),
            "global_version": global_version,
            "stage_versions": (
                committed.manifest.get("stage_versions", {}) if committed is not None else {}
            ),
            "command_cursor": self.command_cursor,
            "resume_status": resume_status,
            "parse_cache": "hit" if parse_hit else "miss",
            **resource_info,
        }
        return EngineResult(
            ledger=ledger_build.ledger,
            ledger_version=ledger_build.ledger_version,
            build_info=build_info,
        )

    def query(
        self,
        *,
        mode: str = "fast",
        budget_ms: int | None = None,
        result: EngineResult | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        started: float | None = None
        if result is None and budget_ms is not None:
            started = time.perf_counter()
            config, requests = self.load()
            fault_path = self._fault_path(config)
            fault_hash = self._fault_hash(fault_path)
            receipts_fp = self._receipt_fingerprints(config)
            global_version = self._global_version(
                config=config,
                requests=requests,
                receipt_fingerprints=receipts_fp,
                fault_hash=fault_hash,
                render_version=None,
            )
            if resume:
                resumed = self.store.load_resumed_ledger(
                    global_version=global_version,
                    config_hash=config_hash(config),
                    command_cursor=self.command_cursor,
                )
                if resumed is not None:
                    remaining = max(
                        0,
                        int(budget_ms - (time.perf_counter() - started) * 1000),
                    )
                    return render_view(
                        resumed[0],
                        mode=mode,
                        budget_ms=remaining,
                    )
            if time.perf_counter() >= started + max(0, budget_ms) / 1000.0:
                summary = {
                    "workers": len(config.workers),
                    "requests_total": len(requests),
                    "entries_authoritative": 0,
                    "entries_uncovered": 0,
                    "invalidated_requests": 0,
                    "reused_requests": 0,
                    "failures": 0,
                    "diagnostic_hits": 0,
                    "degradations": 0,
                    "cold_start": True,
                }
                return {
                    "mode": mode,
                    "status": "partial",
                    "error_codes": [E_RESPONSE_BUDGET_EXCEEDED],
                    "entries": [],
                    "progress": {
                        "completed": 0,
                        "total": len(requests),
                        "fraction": f"0/{len(requests)}",
                        "complete": False,
                        "invalidated_total": 0,
                        "invalidated_completed": 0,
                    },
                    "budget_ms": budget_ms,
                    "summary": summary,
                }
        built = result or self.build(resume=resume)
        if budget_ms is None and mode == "fast":
            budget_ms = self.config.interaction_budget_ms if self.config else None
        elif started is not None:
            budget_ms = max(
                0,
                int(budget_ms - (time.perf_counter() - started) * 1000),
            )
        return render_view(built.ledger, mode=mode, budget_ms=budget_ms)

    def verify_views(self, *, budget_ms: int | None = None) -> dict[str, Any]:
        result = self.build(resume=False)
        fast = render_view(result.ledger, mode="fast", budget_ms=budget_ms)
        authority = render_view(result.ledger, mode="authoritative", budget_ms=None)
        issues = compare_views(fast, authority)
        return {
            "consistent": not issues,
            "fast": fast,
            "authoritative": authority,
            "issues": [issue.to_dict() for issue in issues],
        }
