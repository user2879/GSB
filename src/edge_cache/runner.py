"""End-to-end run orchestration.

A run applies the validated command stream between batches, keeps real node
subprocesses in sync with node add/remove commands, performs precise
invalidation between batches and stages every phase through atomic snapshots.
The same orchestration backs both incremental runs and clean-slate full
rebuilds, which is how the suite proves field-level equivalence.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from typing import Any

from .clock import DeterministicClock
from .config import (
    ALLOWED_FIXED_POINTS,
    ALLOWED_QUANT_LEVELS,
    BatchConfig,
    NodeConfig,
    ShardConfig,
    load_config_dir,
)
from .errors import E_CONFIG_INVALID, EdgeCacheError, Failure
from .invalidation import (
    InvalidationPlan,
    apply_plan,
    empty_plan,
    invalidate_format,
    invalidate_node_profile,
    invalidate_node_quota,
    invalidate_shard,
)
from .ledger import Ledger
from .node_pool import NodePool
from .receipts import C1, C2, C3, FORMATS
from .snapshot import SnapshotStore, build_diagnostic_report, write_exit_record
from .scheduler import DeliveryEngine


def _clock_schedule(config: BatchConfig, work_dir: str) -> tuple[DeterministicClock, str | None]:
    # Generous, strictly increasing millisecond schedule shared with nodes so
    # timestamps stay deterministic while remaining monotonic.
    steps = max(256, len(config.shards) * len(config.nodes) * 16)
    schedule = list(range(1, steps + 1))
    clock = DeterministicClock(schedule)
    path = os.path.join(work_dir, "clock-schedule.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schedule": schedule}, handle)
    return clock, path


class RunOrchestrator:
    def __init__(self, config_dir: str, work_dir: str,
                 exchange_format: str = C3) -> None:
        self.config_dir = config_dir
        self.work_dir = work_dir
        self.exchange_format = exchange_format
        os.makedirs(work_dir, exist_ok=True)
        self.ledger_dir = os.path.join(work_dir, "ledger")
        self.snapshot_dir = os.path.join(work_dir, "snapshots")
        self.config: BatchConfig = load_config_dir(config_dir)
        self.ledger = Ledger(self.ledger_dir)
        self.snapshots = SnapshotStore(self.snapshot_dir)
        self.snapshots.cleanup_temp()
        self.global_version = 1
        self.completed_shards: list[str] = []
        self.format_counts = {C1: 0, C2: 0, C3: 0}
        self.node_advances: dict[str, int] = {}

    # -- mutation helpers --------------------------------------------------
    def _apply_command(self, name: str, args: dict[str, Any]) -> InvalidationPlan:
        plan = empty_plan()
        if name == "replace_shard":
            shard_id = str(args["shard_id"])
            current = self.config.shard(shard_id)
            updated = replace(
                current,
                seed=int(args.get("seed", current.seed)),
                size_units=int(args.get("size_units", current.size_units)),
                sample_size=int(args.get("sample_size", current.sample_size)),
            )
            self.config = self._replace_shard(updated)
            invalidate_shard(self.config, shard_id, plan)
        elif name == "set_quota_version":
            node_id = str(args["node_id"])
            version = str(args["quota_version"])
            if version not in {quota.version for quota in self.config.quotas}:
                raise EdgeCacheError(Failure(E_CONFIG_INVALID,
                                             "unknown quota version in command",
                                             {"version": version}))
            node = replace(self.config.node(node_id), quota_version=version)
            self.config = self._replace_node(node)
            invalidate_node_quota(self.config, node_id, plan)
        elif name == "set_fixed_point":
            node_id = str(args["node_id"])
            fixed_point = int(args["fixed_point"])
            if fixed_point not in ALLOWED_FIXED_POINTS:
                raise EdgeCacheError(Failure(E_CONFIG_INVALID,
                                             "fixed_point outside enum",
                                             {"fixed_point": fixed_point}))
            node = replace(self.config.node(node_id), fixed_point=fixed_point)
            self.config = self._replace_node(node)
            invalidate_node_profile(self.config, node_id, plan)
        elif name == "format_roundtrip":
            target = str(args.get("format", self.exchange_format))
            if target not in FORMATS:
                raise EdgeCacheError(Failure(E_CONFIG_INVALID,
                                             "unknown exchange format",
                                             {"format": target}))
            self.exchange_format = target
            invalidate_format(self.config, plan)
        elif name == "add_node":
            self._add_node(args)
            invalidate_node_quota(self.config, str(args["node_id"]),
                                  empty_plan())  # new node: nothing to remove
        elif name == "remove_node":
            self._remove_node(str(args["node_id"]))
        return plan

    def _replace_shard(self, updated: ShardConfig) -> BatchConfig:
        shards = tuple(
            updated if shard.shard_id == updated.shard_id else shard
            for shard in self.config.shards
        )
        return replace(self.config, shards=shards)

    def _replace_node(self, updated: NodeConfig) -> BatchConfig:
        nodes = tuple(
            updated if node.node_id == updated.node_id else node
            for node in self.config.nodes
        )
        return replace(self.config, nodes=nodes)

    def _add_node(self, args: dict[str, Any]) -> None:
        node = NodeConfig(
            node_id=str(args["node_id"]),
            port=int(args["port"]),
            quota_version=str(args["quota_version"]),
            fixed_point=int(args.get("fixed_point", 2)),
            quant_level=int(args.get("quant_level", 1)),
            lease_limit=int(args["lease_limit"]),
            capacity_slots=int(args["capacity_slots"]),
            soft_watermark_permille=int(args.get("soft_watermark_permille", 700)),
            hard_watermark_permille=int(args.get("hard_watermark_permille", 900)),
        )
        if any(existing.node_id == node.node_id for existing in self.config.nodes):
            raise EdgeCacheError(Failure(E_CONFIG_INVALID, "duplicate node id",
                                         {"node_id": node.node_id}))
        if any(existing.port == node.port for existing in self.config.nodes):
            raise EdgeCacheError(Failure(E_CONFIG_INVALID, "duplicate port",
                                         {"port": node.port}))
        self.config = replace(self.config, nodes=self.config.nodes + (node,))

    def _remove_node(self, node_id: str) -> None:
        nodes = tuple(node for node in self.config.nodes if node.node_id != node_id)
        if not nodes:
            raise EdgeCacheError(Failure(E_CONFIG_INVALID,
                                         "cannot remove the last node"))
        self.config = replace(self.config, nodes=nodes)

    # -- main run ----------------------------------------------------------
    def run(self) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        config_failures: list[Failure] = []
        # The schedule file is rewritten per batch so each batch starts from
        # the same logical timeline; node processes read it on every clock
        # call only when the file changes, and restore resets their algebra.
        _, schedule_path = _clock_schedule(self.config, self.work_dir)
        with NodePool(self.config, os.path.join(self.work_dir, "node-specs"),
                      clock_schedule_path=schedule_path) as pool:
            self._sync_pool(pool)
            reports.append(self._run_one_batch(pool))
            for command in self.config.commands:
                try:
                    plan = self._apply_command(command.name, command.args)
                except EdgeCacheError as exc:
                    config_failures.append(exc.failure)
                    continue
                apply_plan(self.ledger, plan)
                self._sync_pool(pool)
                self.global_version += 1
                reports.append(self._run_one_batch(pool))
        self.ledger.add_failures(config_failures)
        self.ledger.flush_failures()
        report = build_diagnostic_report(
            recovery={"status": "clean-run", "missing_stages": []},
            ledger=self.ledger,
            format_counts=self.format_counts,
            node_advances=self.node_advances,
            budget_consumed=reports[-1]["budget_consumed"] if reports else 0,
        )
        report["batch_reports"] = reports
        return report

    def _sync_pool(self, pool: NodePool) -> None:
        running = set(pool._nodes)  # noqa: SLF001 - supervised lifecycle
        wanted = {node.node_id for node in self.config.nodes}
        for node_id in sorted(running - wanted):
            pool.shutdown_node(node_id)
        for node in self.config.nodes:
            if not pool.is_running(node.node_id):
                pool.start_node(node)

    def _run_one_batch(self, pool: NodePool) -> dict[str, Any]:
        # Fresh deterministic timeline and clean node algebra per batch, so a
        # clean-slate rebuild and an incremental run observe identical state.
        clock, _ = _clock_schedule(self.config, self.work_dir)
        pool.restore_all({node_id: {"leases": [], "algebra_log": [], "reset_clock": True}
                          for node_id in pool._nodes})
        handle = self.snapshots.begin(f"v{self.global_version}")
        self.snapshots.write_stage(handle, "config", {
            "global_version": self.global_version,
            "config_hash": self.config.source_hash,
            "nodes": [node.__dict__ for node in self.config.nodes],
            "shards": [shard.__dict__ for shard in self.config.shards],
            "quotas": [quota.__dict__ for quota in self.config.quotas],
        })
        engine = DeliveryEngine(self.config, self.ledger, pool, clock=clock)
        result = engine.run_batch(exchange_format=self.exchange_format)
        for fmt in FORMATS:
            self.format_counts[fmt] = self.format_counts.get(fmt, 0) + result["format_counts"].get(fmt, 0)
        self.snapshots.write_stage(handle, "leases", engine.lease_manager.snapshot())
        self.snapshots.write_stage(handle, "receipts", {
            shard_id: arbitration.authority.to_dict()
            for shard_id, arbitration in engine.arb_results.items()
            if arbitration.authority is not None
        })
        self.snapshots.write_stage(handle, "conclusions", {
            shard_id: conclusion.to_dict()
            for shard_id, conclusion in engine.conclusions.items()
        })
        self.snapshots.write_stage(handle, "ledger", {
            "business": self.ledger.business_rows(),
            "failures": self.ledger.failure_rows(),
            "algebra": self.ledger.algebra_rows(),
        })
        completed = [
            shard.shard_id for shard in sorted(self.config.shards, key=lambda item: item.index)
            if engine.conclusions[shard.shard_id].available
        ]
        self.completed_shards = completed
        unfinished = [
            shard.shard_id for shard in self.config.shards if shard.shard_id not in completed
        ]
        self.snapshots.commit(
            handle,
            global_version=self.global_version,
            config_hash=self.config.source_hash,
            cursor={
                "completed_shards": completed,
                "unfinished_shards": unfinished,
                "budget_consumed": engine.budget_consumed,
            },
        )
        self.node_advances = self._count_advances()
        return result

    def _count_advances(self) -> dict[str, int]:
        # Read back the flushed stream so counts survive multi-batch runs.
        rows = Ledger._read_jsonl(os.path.join(self.ledger.root, "algebra.jsonl"))
        advances: dict[str, int] = {}
        for event in rows:
            if event["action"] in ("renew", "release", "expiry-release"):
                node_id = event.get("node_id")
                advances[node_id] = advances.get(node_id, 0) + 1
        return dict(sorted(advances.items()))


def full_rebuild(config_dir: str, work_dir: str,
                 exchange_format: str = C3) -> dict[str, Any]:
    """Run the same orchestration from a blank ledger/snapshot workspace."""

    if os.path.isdir(work_dir):
        for name in os.listdir(work_dir):
            path = os.path.join(work_dir, name)
            if os.path.isdir(path):
                import shutil

                shutil.rmtree(path)
            else:
                os.remove(path)
    orchestrator = RunOrchestrator(config_dir, work_dir, exchange_format)
    return orchestrator.run()
