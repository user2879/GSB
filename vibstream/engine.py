"""Command-driven monitor engine backed by the canonical staged materializer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

from .blocks import SignalBlock, parse_block
from .builder import materialize
from .cache import CacheRegistry
from .canonical import sha256_value
from .config import AppConfig, parse_config
from .errors import (
    E_BLOCK_NOT_FOUND,
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_SNAPSHOT_CORRUPT,
    SnapshotError,
    ValidationError,
)
from .ledger import LedgerWriter
from .snapshot import SnapshotStore


def _repository(config: AppConfig, blocks: Iterable[SignalBlock]) -> dict[str, SignalBlock]:
    repository = {block.block_id: block for block in blocks}
    return repository


def _strict_blocks(config: AppConfig, repository: dict[str, SignalBlock]) -> list[SignalBlock]:
    ordered = sorted(repository.values(), key=lambda item: (item.arrival_seq, item.block_id))
    seen_sequences: dict[int, str] = {}
    for block in ordered:
        if block.arrival_seq in seen_sequences:
            raise ValidationError(
                E_CONFIG_INVALID,
                f"arrival_seq duplicates {seen_sequences[block.arrival_seq]}",
                f"blocks.{block.block_id}.arrival_seq",
            )
        seen_sequences[block.arrival_seq] = block.block_id
    return ordered


def _failure(code: str, message: str, path: str = "$", **extra: Any) -> dict[str, Any]:
    payload = {"code": code, "path": path, "message": message}
    payload.update(extra)
    return payload


def _strip_runtime(state: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(state)
    result.pop("_runtime", None)
    return result


def _deep_config_with(config: AppConfig, section: str, value: Any) -> AppConfig:
    data = copy.deepcopy(config.to_dict())
    data[section] = value
    return parse_config(data)


class MonitorEngine:
    def __init__(
        self,
        config: AppConfig,
        blocks: Iterable[SignalBlock] | None = None,
        *,
        snapshot_root: str | Path | None = None,
        ledger_path: str | Path | None = None,
        auto_snapshot: bool = True,
    ) -> None:
        self.config = config
        self.blocks = _repository(config, blocks or [])
        self.cache = CacheRegistry()
        self.version = 0
        self.high_watermark = max((block.arrival_seq for block in self.blocks.values()), default=-1)
        self.rejections: list[dict[str, Any]] = []
        self.snapshot_store = SnapshotStore(snapshot_root) if snapshot_root else None
        self.auto_snapshot = auto_snapshot
        self.ledger = LedgerWriter(ledger_path) if ledger_path else None
        _strict_blocks(self.config, self.blocks)
        self.state = self._materialize()

    def _materialize(self) -> dict[str, Any]:
        return materialize(blocks=list(self.blocks.values()), config=self.config, cache=self.cache)

    def canonical_state(self) -> dict[str, Any]:
        return _strip_runtime(self.state)

    def full_replay(self) -> dict[str, Any]:
        fresh_cache = CacheRegistry()
        fresh_state = materialize(blocks=list(self.blocks.values()), config=self.config, cache=fresh_cache)
        return _strip_runtime(fresh_state)

    @staticmethod
    def materialize_fresh(config: AppConfig, blocks: Iterable[SignalBlock]) -> dict[str, Any]:
        fresh_cache = CacheRegistry()
        fresh_state = materialize(blocks=list(blocks), config=config, cache=fresh_cache)
        return _strip_runtime(fresh_state)

    def _snapshot_stages(self) -> dict[str, Any]:
        state = self.state
        phases = state["phases"]
        canonical = self.canonical_state()
        return {
            "windows": phases["window"],
            "observations": phases["observation"],
            "clusters": phases["cluster"],
            "segments": phases["segment"],
            "scheduler": phases["scheduler"],
            "cache": state["cache"],
            "dataset": {
                "config": self.config.to_dict(),
                "blocks": [block.to_dict(include_samples=True) for block in _strict_blocks(self.config, self.blocks)],
                "state_meta": {
                    "schema_version": state["schema_version"],
                    "config_hash": state["config_hash"],
                    "failures": state["failures"],
                    "dataset": state["dataset"],
                    "canonical_digest": state["canonical_digest"],
                },
            },
        }

    def commit_snapshot(self, *, force: bool = False) -> dict[str, Any] | None:
        if self.snapshot_store is None:
            return None
        stages = self._snapshot_stages()
        result = self.snapshot_store.commit(
            version=self.version,
            config_hash=self.config.config_hash,
            stages=stages,
            force=force,
        )
        if self.ledger is not None:
            self.ledger.append_event(
                {
                    "event": "snapshot_commit",
                    "version": self.version,
                    "config_hash": self.config.config_hash,
                    "committed": result["committed"],
                    "generation": result.get("generation"),
                }
            )
            self.ledger.write_state(self.canonical_state())
        return result

    def _config_invalidation_tags(self, old: AppConfig, new: AppConfig) -> set[str]:
        tags: set[str] = set()
        if old.window != new.window:
            tags.update(("window", "analysis", "clustering", "segment"))
        if old.bands != new.bands:
            tags.update(("analysis", "bands", "window", "clustering", "segment"))
        if old.clustering != new.clustering:
            tags.update(("clustering", "segment"))
        if old.fixed_point != new.fixed_point:
            tags.update(("fixed-point", "window", "analysis", "bands", "clustering", "segment"))
        if old.segment != new.segment:
            tags.add("segment")
        return tags

    def _apply_config_section(self, section: str, value: Any) -> AppConfig:
        return _deep_config_with(self.config, section, value)

    def apply_command(self, command: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(command, dict) or not isinstance(command.get("type"), str):
            return self._reject_command(E_COMMAND_INVALID, "command must be an object with a type", command)
        command_type = command["type"]
        version_before = self.version
        old_config = self.config
        old_blocks = dict(self.blocks)
        old_high_watermark = self.high_watermark
        new_config = self.config
        new_blocks = dict(self.blocks)
        removed_or_replaced: set[str] = set()
        command_failures: list[dict[str, Any]] = []
        applied = False

        try:
            if command_type == "append":
                raw_blocks = command.get("blocks")
                if not isinstance(raw_blocks, list):
                    raise ValidationError(E_CONFIG_INVALID, "append.blocks must be an array", "append.blocks")
                for index, raw_block in enumerate(raw_blocks):
                    try:
                        block = parse_block(raw_block, new_config, f"append.blocks[{index}]")
                        if block.block_id in new_blocks:
                            raise ValidationError(
                                E_CONFIG_INVALID,
                                "duplicate block_id",
                                f"append.blocks[{index}].block_id",
                            )
                        if block.arrival_seq <= old_high_watermark:
                            raise ValidationError(
                                E_CONFIG_INVALID,
                                "append arrival_seq must exceed the stream high-watermark",
                                f"append.blocks[{index}].arrival_seq",
                            )
                        new_blocks[block.block_id] = block
                        self.high_watermark = max(self.high_watermark, block.arrival_seq)
                        applied = True
                    except ValidationError as exc:
                        failure = exc.as_dict()
                        failure["command_index"] = index
                        command_failures.append(failure)
                        self.rejections.append(failure)
                _strict_blocks(new_config, new_blocks)

            elif command_type == "replace":
                raw_block = command.get("block")
                if not isinstance(raw_block, dict):
                    raise ValidationError(E_CONFIG_INVALID, "replace.block must be an object", "replace.block")
                block = parse_block(raw_block, new_config, "replace.block")
                if block.block_id not in new_blocks:
                    raise ValidationError(E_BLOCK_NOT_FOUND, "block not found", "replace.block.block_id")
                old_block = new_blocks[block.block_id]
                new_blocks[block.block_id] = block
                removed_or_replaced.add(block.block_id)
                self.high_watermark = max(self.high_watermark, block.arrival_seq)
                applied = True
                _strict_blocks(new_config, new_blocks)

            elif command_type == "delete":
                block_id = command.get("block_id")
                if not isinstance(block_id, str) or block_id not in new_blocks:
                    raise ValidationError(E_BLOCK_NOT_FOUND, "block not found", "delete.block_id")
                del new_blocks[block_id]
                removed_or_replaced.add(block_id)
                applied = True

            elif command_type == "set_window":
                new_config = self._apply_config_section("window", command.get("window"))
            elif command_type == "set_clusters":
                new_config = self._apply_config_section("clustering", command.get("clustering"))
            elif command_type == "set_fixed_point":
                new_config = self._apply_config_section("fixed_point", command.get("fixed_point"))
            elif command_type == "set_bands":
                new_config = self._apply_config_section("bands", command.get("bands"))
            elif command_type == "set_scheduler":
                new_config = self._apply_config_section("scheduler", command.get("scheduler"))
            elif command_type == "set_segment":
                new_config = self._apply_config_section("segment", command.get("segment"))
            elif command_type in {"noop", "checkpoint"}:
                applied = True
            else:
                raise ValidationError(E_COMMAND_INVALID, f"unknown command type: {command_type}", "command.type")

            _strict_blocks(new_config, new_blocks)
            if new_config != old_config:
                applied = True

            if not applied:
                report = self._report(
                    command=command_type,
                    status="REJECTED",
                    version_before=version_before,
                    version_after=version_before,
                    invalidated=[],
                    reused=[],
                    failures=command_failures,
                )
                self._write_ledger(report)
                return report

            invalidated: list[str] = []
            if removed_or_replaced:
                invalidated.extend(self.cache.invalidate_blocks(removed_or_replaced))
            tags = self._config_invalidation_tags(old_config, new_config)
            if tags:
                invalidated.extend(self.cache.invalidate_config_tags(tags))
            invalidated = sorted(set(invalidated))

            # Commit repository/config only after all serializable validation succeeds.
            if new_config.scheduler.batch_point_budget != old_config.scheduler.batch_point_budget or new_config.scheduler.cluster_point_quota != old_config.scheduler.cluster_point_quota:
                pass
            self.config = new_config
            self.blocks = new_blocks
            self.high_watermark = old_high_watermark if not applied else self.high_watermark
            self.version += 1
            self.state = self._materialize()
            reused = list(self.state.get("_runtime", {}).get("reused_cache_keys", []))
            if self.auto_snapshot:
                self.commit_snapshot()
            report = self._report(
                command=command_type,
                status="APPLIED",
                version_before=version_before,
                version_after=self.version,
                invalidated=invalidated,
                reused=reused,
                failures=command_failures,
            )
            self._write_ledger(report)
            return report
        except ValidationError as exc:
            self.config = old_config
            self.blocks = old_blocks
            self.high_watermark = old_high_watermark
            command_failures.append(exc.as_dict())
            self.rejections.extend(command_failures)
            report = self._report(
                command=command_type,
                status="REJECTED",
                version_before=version_before,
                version_after=version_before,
                invalidated=[],
                reused=[],
                failures=command_failures,
            )
            self._write_ledger(report)
            return report

    def _report(
        self,
        *,
        command: str,
        status: str,
        version_before: int,
        version_after: int,
        invalidated: list[str],
        reused: list[str],
        failures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "event": "transaction",
            "command": command,
            "status": status,
            "version_before": version_before,
            "version_after": version_after,
            "config_hash": self.config.config_hash,
            "invalidated_cache_keys": sorted(invalidated),
            "reused_cache_keys": sorted(set(reused)),
            "failures": failures,
            "canonical_digest": self.state["canonical_digest"] if status == "APPLIED" else None,
        }

    def _reject_command(self, code: str, message: str, command: Any) -> dict[str, Any]:
        failure = _failure(code, message, "command")
        report = {
            "event": "transaction",
            "command": command.get("type") if isinstance(command, dict) else None,
            "status": "REJECTED",
            "version_before": self.version,
            "version_after": self.version,
            "config_hash": self.config.config_hash,
            "invalidated_cache_keys": [],
            "reused_cache_keys": [],
            "failures": [failure],
            "canonical_digest": None,
        }
        self._write_ledger(report)
        return report

    def _write_ledger(self, report: dict[str, Any]) -> None:
        if self.ledger is not None:
            self.ledger.append_event(report)

    def apply_commands(self, commands: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        reports = [self.apply_command(command) for command in commands]
        return reports

    @classmethod
    def restore(
        cls,
        snapshot_root: str | Path,
        *,
        ledger_path: str | Path | None = None,
        auto_snapshot: bool = True,
    ) -> "MonitorEngine":
        store = SnapshotStore(snapshot_root)
        pointer, loaded = store.load_current()
        dataset = loaded["stages"]["dataset"]
        config = parse_config(dataset["config"])
        blocks = [parse_block(item, config, f"snapshot.blocks[{index}]") for index, item in enumerate(dataset["blocks"])]
        engine = cls(
            config,
            blocks,
            snapshot_root=snapshot_root,
            ledger_path=ledger_path,
            auto_snapshot=auto_snapshot,
        )
        engine.version = int(pointer["version"])
        engine.snapshot_store = store
        phases = {
            "window": loaded["stages"]["windows"],
            "observation": loaded["stages"]["observations"],
            "cluster": loaded["stages"]["clusters"],
            "segment": loaded["stages"]["segments"],
            "scheduler": loaded["stages"]["scheduler"],
        }
        meta = dataset["state_meta"]
        restored_state = {
            "schema_version": meta["schema_version"],
            "config_hash": meta["config_hash"],
            "config": config.to_dict(),
            "phases": phases,
            "dataset": meta["dataset"],
            "cache": loaded["stages"]["cache"],
            "failures": meta["failures"],
            "canonical_digest": meta["canonical_digest"],
        }
        digest_payload = {
            key: value for key, value in restored_state.items() if key != "canonical_digest"
        }
        digest = sha256_value(digest_payload)
        if digest != meta["canonical_digest"]:
            raise SnapshotError(E_SNAPSHOT_CORRUPT, "restored canonical digest does not match")
        if restored_state["config_hash"] != pointer["config_hash"]:
            raise SnapshotError(E_SNAPSHOT_CORRUPT, "restored config hash does not match CURRENT")
        engine.state = restored_state
        return engine

    def verify_full_replay(self) -> dict[str, Any]:
        canonical = self.canonical_state()
        fresh = self.full_replay()
        equal = canonical == fresh
        differing: list[str] = []
        if not equal:
            for key in sorted(set(canonical) | set(fresh)):
                if canonical.get(key) != fresh.get(key):
                    differing.append(key)
        return {
            "equal": equal,
            "incremental_digest": canonical.get("canonical_digest"),
            "full_replay_digest": fresh.get("canonical_digest"),
            "differing_top_level_fields": differing,
        }
