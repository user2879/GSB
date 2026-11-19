from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from .artifact import encode_index, load_meta, write_index
from .compat import CompatTable
from .config import Config
from .errors import PipelineError, ValidationError
from .loaders import (
    format_document_line,
    load_corpus,
    replace_document_line,
)
from .postings import build_logical_index
from .search import execute_search
from .snapshot import SnapshotManager, SnapshotState, stage_digest
from .util import atomic_write_bytes, atomic_write_text, deep_merge, read_json, sha256_bytes, sha256_json


class BudgetExceeded(Exception):
    def __init__(self, stage: str, elapsed_ms: float):
        super().__init__(f"interactive budget exceeded during {stage}")
        self.stage = stage
        self.elapsed_ms = elapsed_ms


class BudgetGuard:
    def __init__(self, budget_ms: float | None, started_ns: int | None = None):
        self.started_ns = started_ns if started_ns is not None else time.perf_counter_ns()
        self.budget_ms = budget_ms
        self.deadline_ns = None if budget_ms is None else self.started_ns + int(budget_ms * 1_000_000)
        self.last_stage = "command_received"
        self.completed_stages: list[str] = []

    @property
    def enabled(self) -> bool:
        return self.deadline_ns is not None

    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self.started_ns) / 1_000_000.0

    def check(self, stage: str) -> None:
        self.last_stage = stage
        if self.deadline_ns is not None and time.perf_counter_ns() > self.deadline_ns:
            raise BudgetExceeded(stage, self.elapsed_ms())
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)


@dataclass(frozen=True)
class PipelineOutcome:
    status: str
    result: dict
    snapshot: SnapshotState | None


class RetrievalEngine:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.manager = SnapshotManager(self.state_dir)

    def build(
        self,
        docs_path: str | Path,
        queries_path: str | Path,
        config_path: str | Path,
        compat_path: str | Path,
        output_dir: str | Path | None = None,
        fresh: bool = False,
    ) -> PipelineOutcome:
        current = self.manager.load_current()
        if current is not None and not fresh:
            raise PipelineError(
                "E_CONFIG_MISMATCH",
                "state already has a committed snapshot; use edit/resume or choose a fresh state directory",
            )
        docs_bytes = Path(docs_path).read_bytes()
        queries_bytes = Path(queries_path).read_bytes()
        config_bytes = Path(config_path).read_bytes()
        compat_bytes = Path(compat_path).read_bytes()
        try:
            config_raw = json.loads(config_bytes)
            compat_raw = json.loads(compat_bytes)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise PipelineError("E_INDEX_INVALID", f"invalid config or compatibility JSON: {exc}") from exc
        self.manager.cleanup_work()
        outcome = self._execute(
            previous=None if fresh else current,
            docs_bytes=docs_bytes,
            queries_bytes=queries_bytes,
            config_raw=config_raw,
            compat_raw=compat_raw,
            command_id="build",
            applied_commands=[],
            budget_ms=None,
            budget_started_ns=time.perf_counter_ns(),
            enforce_budget=False,
        )
        if output_dir is not None and outcome.snapshot is not None:
            self.publish(outcome.snapshot, output_dir)
        return outcome

    def resume(
        self,
        expected_config_path: str | Path | None = None,
        expected_version: int | None = None,
    ) -> SnapshotState:
        expected_hash = None
        if expected_config_path is not None:
            try:
                raw = json.loads(Path(expected_config_path).read_bytes())
            except (OSError, json.JSONDecodeError, UnicodeError) as exc:
                raise PipelineError("E_INDEX_INVALID", f"cannot load expected config: {exc}") from exc
            expected_hash = sha256_json(raw)
        current = self.manager.check_resume(expected_hash, expected_version)
        removed = self.manager.cleanup_work()
        removed_snapshots = self.manager.cleanup_orphan_snapshots(current)
        current.manifest["recovery"] = {
            "resumed_from_version": current.version,
            "removed_staging": removed,
            "removed_orphan_snapshots": removed_snapshots,
        }
        return current

    def apply_edit(
        self,
        command: dict,
        output_dir: str | Path | None = None,
        budget_ms: float | None = None,
    ) -> PipelineOutcome:
        started_ns = time.perf_counter_ns()
        current = self.manager.load_current()
        if current is None:
            raise PipelineError("E_INDEX_INVALID", "no committed snapshot; build before editing")
        if not isinstance(command, dict):
            raise PipelineError("E_EDIT_INVALID", "edit command must be an object")
        command_id = command.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise PipelineError("E_EDIT_INVALID", "command_id must be a non-empty string")
        applied = list(current.manifest.get("applied_commands", []))
        if command_id in applied:
            result = self._result_for_existing(current, command_id, status="OK_IDEMPOTENT")
            if output_dir is not None:
                self.publish(current, output_dir)
            return PipelineOutcome("OK_IDEMPOTENT", result, current)

        docs_bytes = current.input_path("docs.jsonl").read_bytes()
        queries_bytes = current.input_path("queries.jsonl").read_bytes()
        config_raw = read_json(current.input_path("config.json"))
        compat_raw = read_json(current.input_path("compat.json"))
        try:
            new_docs, new_config = self._apply_command(
                command,
                docs_bytes,
                config_raw,
                current,
            )
        except ValidationError as exc:
            raise PipelineError(exc.code, exc.message) from exc
        outcome = self._execute(
            previous=current,
            docs_bytes=new_docs,
            queries_bytes=queries_bytes,
            config_raw=new_config,
            compat_raw=compat_raw,
            command_id=command_id,
            applied_commands=applied + [command_id],
            budget_ms=budget_ms,
            budget_started_ns=started_ns,
            enforce_budget=True,
        )
        if outcome.snapshot is not None and output_dir is not None:
            self.publish(outcome.snapshot, output_dir)
        return outcome

    def apply_edit_file(
        self,
        path: str | Path,
        output_dir: str | Path | None = None,
        budget_ms: float | None = None,
    ) -> list[PipelineOutcome]:
        outcomes = []
        for line_no, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
            if not text.strip():
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError as exc:
                raise PipelineError("E_EDIT_INVALID", f"edit command line {line_no} is invalid JSON: {exc.msg}") from exc
            outcome = self.apply_edit(command, output_dir=output_dir, budget_ms=budget_ms)
            outcomes.append(outcome)
            if outcome.status == "E_INTERACT_BUDGET_PENDING":
                break
        return outcomes

    def _apply_command(
        self,
        command: dict,
        docs_bytes: bytes,
        config_raw: dict,
        current: SnapshotState,
    ) -> tuple[bytes, dict]:
        op = command.get("op")
        if op == "add_doc":
            document = command.get("doc")
            if not isinstance(document, dict):
                raise PipelineError("E_EDIT_INVALID", "add_doc requires a doc object")
            doc_id = document.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise PipelineError("E_EDIT_INVALID", "add_doc doc_id must be a non-empty string")
            lines = docs_bytes.decode("utf-8", errors="surrogateescape").split("\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(existing, dict) and existing.get("doc_id") == doc_id:
                    raise PipelineError("E_EDIT_INVALID", f"document already exists: {doc_id}")
            if lines and lines[-1] == "":
                lines.insert(len(lines) - 1, format_document_line(document))
            else:
                lines.append(format_document_line(document))
            return "\n".join(lines).encode("utf-8", errors="surrogateescape"), config_raw
        if op == "fix_doc":
            document = command.get("doc")
            if not isinstance(document, dict):
                raise PipelineError("E_EDIT_INVALID", "fix_doc requires a doc object")
            doc_id = document.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise PipelineError("E_EDIT_INVALID", "fix_doc doc_id must be a non-empty string")
            lines = docs_bytes.decode("utf-8", errors="surrogateescape").split("\n")
            replaced = replace_document_line(lines, doc_id, document)
            return "\n".join(replaced).encode("utf-8", errors="surrogateescape"), config_raw
        if op == "set_config":
            patch = command.get("patch", command.get("config_patch"))
            if not isinstance(patch, dict):
                raise PipelineError("E_EDIT_INVALID", "set_config requires a patch object")
            patched = deep_merge(config_raw, patch)
            try:
                Config.from_mapping(patched, self._compat_from_snapshot(current))
            except ValidationError as exc:
                raise PipelineError(exc.code, exc.message) from exc
            return docs_bytes, patched
        raise PipelineError("E_EDIT_INVALID", f"unsupported edit op: {op}")

    def _compat_from_snapshot(self, current: SnapshotState) -> CompatTable:
        raw = read_json(current.input_path("compat.json"))
        return CompatTable.from_mapping(raw, file_hash=sha256_json(raw))

    def _execute(
        self,
        previous: SnapshotState | None,
        docs_bytes: bytes,
        queries_bytes: bytes,
        config_raw: dict,
        compat_raw: dict,
        command_id: str,
        applied_commands: list[str],
        budget_ms: float | None,
        budget_started_ns: int,
        enforce_budget: bool,
    ) -> PipelineOutcome:
        compat = CompatTable.from_mapping(compat_raw, file_hash=sha256_json(compat_raw))
        try:
            config = Config.from_mapping(config_raw, compat)
        except ValidationError as exc:
            raise PipelineError(exc.code, exc.message) from exc
        effective_budget = (config.interaction_budget_ms if budget_ms is None else budget_ms) if enforce_budget else None
        guard = BudgetGuard(effective_budget, budget_started_ns)
        version = self.manager.next_version(previous)
        staging = self.manager.begin_staging(version)
        try:
            guard.check("command_received")
            inputs_tmp = self.manager.stage_temp(staging, "inputs")
            inputs_tmp.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(inputs_tmp / "docs.jsonl", docs_bytes)
            atomic_write_bytes(inputs_tmp / "queries.jsonl", queries_bytes)
            atomic_write_text(inputs_tmp / "config.json", json.dumps(config_raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            atomic_write_text(inputs_tmp / "compat.json", json.dumps(compat_raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            self.manager.commit_stage(staging, "inputs")
            guard.check("inputs_committed")

            corpus = load_corpus(staging / "inputs" / "docs.jsonl", staging / "inputs" / "queries.jsonl", config)
            guard.check("input_validation")

            normalized_tmp = self.manager.stage_temp(staging, "normalized")
            normalized_tmp.mkdir(parents=True, exist_ok=True)
            normalized_payload = corpus.to_dict()
            atomic_write_text(
                normalized_tmp / "corpus.json",
                json.dumps(normalized_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            atomic_write_text(
                normalized_tmp / "failures.json",
                json.dumps([failure.to_dict() for failure in corpus.failures], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            self.manager.commit_stage(staging, "normalized")
            guard.check("normalization")

            logical = build_logical_index(
                corpus.documents,
                config,
                checkpoint=lambda: guard.check("logical_index_build"),
            )
            previous_meta = None
            previous_compressed = None
            if previous is not None:
                candidate_dir = previous.stage_path("compressed")
                if (candidate_dir / "index.json").is_file():
                    previous_compressed = candidate_dir
                    previous_meta = load_meta(candidate_dir)
            encoded = encode_index(
                logical,
                config,
                previous_root=previous_compressed,
                previous_meta=previous_meta,
                checkpoint=lambda: guard.check("compression_encoding"),
            )
            guard.check("logical_index")
            compressed_tmp = self.manager.stage_temp(staging, "compressed")
            compressed_tmp.mkdir(parents=True, exist_ok=True)
            write_index(
                encoded,
                compressed_tmp,
                checkpoint=lambda: guard.check("compression_write"),
            )
            self.manager.commit_stage(staging, "compressed")
            guard.check("compression_written")

            previous_results = None
            if previous is not None:
                prior_results_path = previous.stage_path("fusion") / "results.json"
                if prior_results_path.is_file():
                    previous_results = read_json(prior_results_path)
            search = execute_search(
                corpus.documents,
                corpus.queries,
                logical,
                config,
                previous_results=previous_results,
                checkpoint=lambda: guard.check("retrieval_fusion_query"),
            )
            guard.check("retrieval_fusion")
            fusion_tmp = self.manager.stage_temp(staging, "fusion")
            fusion_tmp.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                fusion_tmp / "results.json",
                json.dumps(list(search.results), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            atomic_write_text(
                fusion_tmp / "diagnostics.json",
                json.dumps(search.diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            invalidation = self._build_invalidation(previous, previous_meta, encoded, config, search, version)
            atomic_write_text(
                fusion_tmp / "invalidation.json",
                json.dumps(invalidation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            self.manager.commit_stage(staging, "fusion")
            guard.check("fusion_written")

            completed_batches = math.ceil(len(corpus.documents) / config.posting_block_size) if corpus.documents else 0
            checkpoint_batches = list(range(config.snapshot_interval_batches, completed_batches + 1, config.snapshot_interval_batches))
            manifest = {
                "schema_version": 1,
                "global_version": version,
                "status": "complete",
                "command_id": command_id,
                "applied_commands": applied_commands,
                "config_hash": config.config_hash,
                "compat_hash": sha256_json(compat_raw),
                "index_config_hash": config.index_hash(),
                "scoring_hash": config.scoring_hash(),
                "source_hashes": {
                    "docs": sha256_bytes(docs_bytes),
                    "queries": sha256_bytes(queries_bytes),
                    "config": sha256_json(config_raw),
                    "compat": sha256_json(compat_raw),
                },
                "completed_batches": completed_batches,
                "checkpoint_batches": checkpoint_batches,
                "snapshot_interval_batches": config.snapshot_interval_batches,
                "stages": {},
            }
            for stage in ("inputs", "normalized", "compressed", "fusion"):
                manifest["stages"][stage] = {
                    "version": stage_digest(staging / stage),
                    "complete": True,
                }
            atomic_write_text(
                staging / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            guard.check("manifest_written")
            committed = self.manager.commit(staging, version)
            result = self._success_result(committed, encoded, search, command_id, invalidation)
            return PipelineOutcome("OK", result, committed)
        except BudgetExceeded as exc:
            self.manager.discard_staging(staging)
            summary = previous.summary() if previous is not None else {
                "completed_queries": 0,
                "result_count": 0,
                "progress": 0.0,
            }
            elapsed = guard.elapsed_ms()
            event = {
                "schema_version": 1,
                "command_id": command_id,
                "status": "E_INTERACT_BUDGET_PENDING",
                "budget_ms": effective_budget,
                "elapsed_ms": elapsed,
                "completed_stage": exc.stage,
                "completed_stages": list(guard.completed_stages),
                "prior_snapshot_version": previous.version if previous is not None else None,
            }
            self.manager.record_budget_event(event)
            prior_fingerprint = None
            if previous is not None:
                prior_meta = previous.stage_path("compressed") / "index.json"
                if prior_meta.is_file():
                    prior_fingerprint = read_json(prior_meta).get("index_fingerprint")
            result = {
                "status": "E_INTERACT_BUDGET_PENDING",
                "command_id": command_id,
                "budget_ms": effective_budget,
                "elapsed_ms": elapsed,
                "completed_stage": exc.stage,
                "completed_stages": list(guard.completed_stages),
                "summary": summary,
                "result": summary,
                "index_fingerprint": prior_fingerprint,
                "message": "latest committed retrieval summary returned without waiting for the new parameters",
            }
            return PipelineOutcome("E_INTERACT_BUDGET_PENDING", result, previous)
        except BaseException:
            self.manager.discard_staging(staging)
            raise

    def _build_invalidation(self, previous, previous_meta, encoded, config: Config, search, new_version: int) -> dict:
        reasons: list[str] = []
        if previous is None or previous_meta is None:
            reasons.append("full_rebuild")
        else:
            if previous_meta.get("format_version") != encoded.meta.get("format_version"):
                reasons.append("format_version_changed")
            if previous_meta.get("compression_tier") != encoded.meta.get("compression_tier"):
                reasons.append("compression_tier_changed")
            if bool(previous_meta.get("keep_positions")) != bool(encoded.meta.get("keep_positions")):
                reasons.append("keep_positions_changed")
            if previous_meta.get("block_size") != encoded.meta.get("block_size"):
                reasons.append("posting_block_size_changed")
            if previous.manifest.get("index_config_hash") != config.index_hash() and not reasons:
                reasons.append("index_configuration_changed")
            if previous_meta.get("logical_digest") != encoded.logical_digest:
                reasons.append("posting_content_changed")
            if previous_meta.get("sidecar_digest") != encoded.meta.get("sidecar_digest"):
                reasons.append("sidecar_metadata_changed")
            if not reasons and encoded.rebuilt_keys:
                reasons.append("block_representation_changed")
        if not reasons and encoded.reused_keys:
            reasons.append("no_index_invalidation")
        query_actions = search.diagnostics.get("query_invalidation", {})
        query_reasons: dict[str, int] = {}
        for item in query_actions.values():
            key = item["reason"]
            query_reasons[key] = query_reasons.get(key, 0) + 1
        return {
            "previous_snapshot_version": previous.version if previous is not None else None,
            "new_snapshot_version": new_version,
            "index_reasons": sorted(set(reasons)),
            "tier_semantic_mapping": {
                "previous": previous_meta.get("tier_semantic") if previous_meta else None,
                "current": encoded.meta.get("tier_semantic"),
            },
            "scoring_reason": (
                "scoring_config_changed"
                if previous is not None and previous.manifest.get("scoring_hash") != config.scoring_hash()
                else "scoring_config_unchanged"
            ),
            "reused_blocks": list(encoded.reused_keys),
            "rebuilt_blocks": list(encoded.rebuilt_keys),
            "reused_queries": list(search.reused_query_ids),
            "rebuilt_queries": list(search.rebuilt_query_ids),
            "query_reasons": query_reasons,
            "degradation_losses": encoded.meta.get("degradation_losses", []),
        }

    def _success_result(self, snapshot: SnapshotState, encoded, search, command_id: str, invalidation: dict) -> dict:
        diagnostics = search.diagnostics
        return {
            "status": "OK",
            "command_id": command_id,
            "global_version": snapshot.version,
            "snapshot": str(snapshot.snapshot_root),
            "index_fingerprint": encoded.fingerprint,
            "index": {
                "format_version": encoded.meta["format_version"],
                "compression_tier": encoded.meta["compression_tier"],
                "tier_semantic": encoded.meta["tier_semantic"],
                "keep_positions": encoded.meta["keep_positions"],
                "block_size": encoded.meta["block_size"],
                "logical_digest": encoded.logical_digest,
                "block_count": encoded.meta["block_count"],
                "document_count": encoded.meta["document_count"],
                "reuse_report": encoded.meta["reuse_report"],
                "degradation_losses": encoded.meta["degradation_losses"],
            },
            "summary": diagnostics["summary"],
            "query_reuse": diagnostics["query_reuse"],
            "invalidation": invalidation,
            "failures": len(read_json(snapshot.stage_path("normalized") / "failures.json")),
            "stages": snapshot.manifest["stages"],
        }

    def _result_for_existing(self, snapshot: SnapshotState, command_id: str, status: str) -> dict:
        index_meta = read_json(snapshot.stage_path("compressed") / "index.json")
        diagnostics = read_json(snapshot.stage_path("fusion") / "diagnostics.json")
        failures = read_json(snapshot.stage_path("normalized") / "failures.json")
        return {
            "status": status,
            "command_id": command_id,
            "global_version": snapshot.version,
            "snapshot": str(snapshot.snapshot_root),
            "index_fingerprint": index_meta["index_fingerprint"],
            "index": {
                "format_version": index_meta["format_version"],
                "compression_tier": index_meta["compression_tier"],
                "tier_semantic": index_meta["tier_semantic"],
                "keep_positions": index_meta["keep_positions"],
                "block_size": index_meta["block_size"],
                "logical_digest": index_meta["logical_digest"],
                "block_count": index_meta["block_count"],
                "document_count": index_meta["document_count"],
                "reuse_report": index_meta.get("reuse_report", {}),
                "degradation_losses": index_meta.get("degradation_losses", []),
            },
            "summary": diagnostics.get("summary", {}),
            "query_reuse": diagnostics.get("query_reuse", {}),
            "invalidation": read_json(snapshot.stage_path("fusion") / "invalidation.json")
            if (snapshot.stage_path("fusion") / "invalidation.json").is_file()
            else {},
            "failures": len(failures),
            "stages": snapshot.manifest["stages"],
        }

    def publish(self, snapshot: SnapshotState, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        index_meta = read_json(snapshot.stage_path("compressed") / "index.json")
        diagnostics = read_json(snapshot.stage_path("fusion") / "diagnostics.json")
        results = read_json(snapshot.stage_path("fusion") / "results.json")
        failures = read_json(snapshot.stage_path("normalized") / "failures.json")
        index_version = {
            "global_version": snapshot.version,
            "index_fingerprint": index_meta["index_fingerprint"],
            "logical_digest": index_meta["logical_digest"],
            "format_version": index_meta["format_version"],
            "compression_tier": index_meta["compression_tier"],
            "tier_semantic": index_meta["tier_semantic"],
            "keep_positions": index_meta["keep_positions"],
            "block_count": index_meta["block_count"],
            "degradation_losses": index_meta.get("degradation_losses", []),
        }
        invalidation_path = snapshot.stage_path("fusion") / "invalidation.json"
        if invalidation_path.is_file():
            index_version["invalidation"] = read_json(invalidation_path)
        atomic_write_text(out / "index_version.json", json.dumps(index_version, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        atomic_write_text(out / "retrieval_results.json", json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        atomic_write_text(out / "diagnostics.json", json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        atomic_write_text(out / "failures.json", json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        atomic_write_text(out / "manifest.json", json.dumps(snapshot.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if invalidation_path.is_file():
            atomic_write_text(out / "invalidation.json", json.dumps(read_json(invalidation_path), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
