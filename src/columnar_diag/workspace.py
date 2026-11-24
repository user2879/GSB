from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .codec import encode_block
from .errors import AppError
from .ledger import Ledger
from .models import (
    ENCODINGS,
    SEGMENT_FORMAT_VERSION,
    BlockSpec,
    BudgetConfig,
    ScanTask,
    SegmentSpec,
)
from .planner import (
    PlanBundle,
    attach_task_memory,
    build_plans,
    plan_to_dict,
)
from .scan import (
    compute_query_results,
    execute_tasks,
    query_overall_digest,
)
from .scheduler import validate_budget
from .segment import SegmentImage, encode_spec_blocks, read_segment, spec_digest, write_segment
from .spec import load_spec, write_spec
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_json,
)


CONFIG_FILE = "workspace.json"
SPEC_FILE = "spec.json"
SEGMENT_FILE = "segment.seg"
LEDGER_FILE = "ledger.sqlite3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _default_config(budget: BudgetConfig) -> dict[str, Any]:
    return {
        "app_format_version": 1,
        "segment_file": SEGMENT_FILE,
        "spec_file": SPEC_FILE,
        "ledger_file": LEDGER_FILE,
        "versions": {
            "segment_format_version": SEGMENT_FORMAT_VERSION,
            "encoding_selection_version": 1,
            "budget_params_version": 1,
        },
        "budget": budget.to_dict(),
    }


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.config_path = self.root / CONFIG_FILE
        self.spec_path = self.root / SPEC_FILE
        self.segment_path = self.root / SEGMENT_FILE
        self.ledger_path = self.root / LEDGER_FILE
        self.config = read_json(self.config_path)
        self.spec = load_spec(self.spec_path)
        self.image = read_segment(self.segment_path)
        self.ledger = Ledger(self.ledger_path)
        try:
            self._validate_loaded_state()
            generation = self.ledger.current_generation()
            if generation is not None:
                if int(generation["budget_version"]) != int(
                    self.config["versions"]["budget_params_version"]
                ):
                    raise AppError(
                        "E_LEDGER",
                        "活动账本预算版本与工作区配置不一致",
                        field="budget_params_version",
                    )
                if int(generation["encoding_version"]) != int(
                    self.config["versions"]["encoding_selection_version"]
                ):
                    raise AppError(
                        "E_LEDGER",
                        "活动账本编码版本与工作区配置不一致",
                        field="encoding_selection_version",
                    )
                if generation["segment_digest"] != self.image.file_digest:
                    raise AppError(
                        "E_LEDGER",
                        "活动账本段摘要与当前段文件不一致",
                        field="segment_digest",
                    )
        except Exception:
            self.ledger.close()
            raise
        self.budget = BudgetConfig.from_dict(self.config["budget"])
        self.bundle = self._build_bundle()

    @classmethod
    def create(
        cls,
        root: str | Path,
        spec: SegmentSpec,
        *,
        budget: BudgetConfig | None = None,
        overwrite: bool = False,
    ) -> "Workspace":
        target = Path(root)
        if target.exists() and any(target.iterdir()):
            if not overwrite:
                raise AppError(
                    "E_WORKSPACE",
                    "工作目录非空；使用 overwrite=True 才能整体重建",
                    field="root",
                )
            protected = {CONFIG_FILE, SPEC_FILE, SEGMENT_FILE, LEDGER_FILE}
            for child in target.iterdir():
                if child.name in protected or child.name.startswith("ledger.sqlite3-"):
                    child.unlink(missing_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        config = _default_config(budget or BudgetConfig())
        write_spec(spec, target / SPEC_FILE)
        atomic_write_json(target / CONFIG_FILE, config)
        encoded = encode_spec_blocks(spec)
        image = write_segment(
            target / SEGMENT_FILE,
            spec,
            [encoded[block.id] for block in spec.blocks],
            encoding_selection_version=config["versions"]["encoding_selection_version"],
        )
        ledger = Ledger(target / LEDGER_FILE)
        try:
            bundle = attach_task_memory(build_plans(spec), image.blocks)
            ledger.begin_generation(
                reason="initial-create",
                budget=BudgetConfig.from_dict(config["budget"]),
                budget_version=config["versions"]["budget_params_version"],
                encoding_version=config["versions"]["encoding_selection_version"],
                segment_digest=image.file_digest,
                spec_digest=image.spec_digest,
                plan_digest=bundle.digest,
                tasks_total=len(bundle.tasks),
                cursor=0,
            )
        finally:
            ledger.close()
        return cls(target)

    def close(self) -> None:
        self.ledger.close()

    def _build_bundle(self) -> PlanBundle:
        return attach_task_memory(build_plans(self.spec), self.image.blocks)

    def _validate_loaded_state(self) -> None:
        if self.spec.format_version != SEGMENT_FORMAT_VERSION:
            raise AppError(
                "E_SPEC",
                "规范格式版本与段格式版本不一致",
                block_id="<spec>",
                field="format_version",
            )
        if self.spec.segment_id != self.image.header.get("segment_id"):
            raise AppError(
                "E_SEGMENT",
                "规范 segment_id 与段头不一致",
                field="segment_id",
            )
        expected_spec_digest = spec_digest(self.spec)
        if expected_spec_digest != self.image.spec_digest:
            raise AppError(
                "E_SEGMENT",
                "规范内容摘要与段头不一致",
                field="spec_digest",
                details={
                    "expected": expected_spec_digest,
                    "actual": self.image.spec_digest,
                },
            )
        if tuple(block.id for block in self.spec.blocks) != self.image.block_order:
            raise AppError(
                "E_SEGMENT", "规范块顺序与段文件不一致", field="block_order"
            )
        for block in self.spec.blocks:
            encoded = self.image.blocks[block.id]
            if (
                encoded.column != block.column
                or encoded.row_group != block.row_group
                or encoded.encoding != block.encoding
                or encoded.compressed != block.compressed
            ):
                raise AppError(
                    "E_SEGMENT",
                    "块元数据与规范不一致",
                    block_id=block.id,
                    field="metadata",
                )
            decoded = self.image.values(block.id)
            if decoded != block.values:
                raise AppError(
                    "E_SEGMENT",
                    "块解码值与规范样本值不一致",
                    block_id=block.id,
                    field="values",
                )
        config_encoding_version = int(
            self.config["versions"]["encoding_selection_version"]
        )
        config_segment_version = int(
            self.config["versions"]["segment_format_version"]
        )
        if config_segment_version != self.spec.format_version:
            raise AppError(
                "E_SEGMENT",
                "段格式版本与工作区配置不一致",
                field="segment_format_version",
            )
        if config_encoding_version != self.image.encoding_selection_version:
            raise AppError(
                "E_SEGMENT",
                "编码选择版本与段头不一致",
                field="encoding_selection_version",
                details={
                    "config": config_encoding_version,
                    "segment": self.image.encoding_selection_version,
                },
            )
        self.budget = BudgetConfig.from_dict(self.config["budget"])
        validate_budget(self.budget)

    def segment_set_summary(self) -> dict[str, Any]:
        return {
            "digest": sha256_json(
                [
                    {
                        "segment_id": self.spec.segment_id,
                        "file_digest": self.image.file_digest,
                        "spec_digest": self.image.spec_digest,
                        "format_version": self.spec.format_version,
                        "encoding_selection_version": self.image.encoding_selection_version,
                    }
                ]
            ),
            "segments": [
                {
                    "segment_id": self.spec.segment_id,
                    "file_digest": self.image.file_digest,
                    "spec_digest": self.image.spec_digest,
                    "format_version": self.spec.format_version,
                    "encoding_selection_version": self.image.encoding_selection_version,
                }
            ],
        }

    @property
    def versions(self) -> dict[str, int]:
        return {
            "segment_format_version": SEGMENT_FORMAT_VERSION,
            "encoding_selection_version": int(
                self.config["versions"]["encoding_selection_version"]
            ),
            "budget_params_version": int(
                self.config["versions"]["budget_params_version"]
            ),
        }

    def _save_config(self) -> None:
        self.config["budget"] = self.budget.to_dict()
        atomic_write_json(self.config_path, self.config)

    def plan_payloads(self) -> dict[str, dict[str, Any]]:
        return {
            plan.query_id: plan_to_dict(plan, self.bundle.query_asts[plan.query_id])
            for plan in self.bundle.plans
        }

    def query_view(self) -> list[dict[str, Any]]:
        qresults = self.ledger.current_query_results()
        by_query: dict[str, list[dict[str, Any]]] = {}
        for item in qresults:
            by_query.setdefault(item["query_id"], []).append(item)
        output: list[dict[str, Any]] = []
        for plan in self.bundle.plans:
            items = by_query.get(plan.query_id, [])
            output.append(
                {
                    "query_id": plan.query_id,
                    "select": list(plan.select),
                    "plan_digest": plan.digest,
                    "candidate_blocks": list(plan.candidate_blocks),
                    "pruned_columns": list(plan.pruned_columns),
                    "basis": list(plan.basis),
                    "row_groups": [
                        {
                            **item,
                            "rows": item["rows"],
                        }
                        for item in items
                    ],
                    "overall_digest": (
                        query_overall_digest(plan.query_id, items) if items else None
                    ),
                }
            )
        return output

    def scan_pending(
        self,
        *,
        budget: BudgetConfig | None = None,
        batch_size: int = 4,
        max_tasks: int | None = None,
        delay_ms: float = 0.0,
        commit_partial_tail: bool = True,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise AppError("E_BUDGET", "batch_size 必须为正整数", field="batch_size")
        active_budget = budget or self.budget
        validate_budget(active_budget, self.bundle.tasks)
        generation = self.ledger.current_generation()
        if generation is None:
            raise AppError("E_LEDGER", "没有活动生成", field="generation")
        cursor = int(generation["cursor"])
        sequence = list(self.bundle.tasks)
        if cursor > len(sequence):
            raise AppError(
                "E_REPLAY",
                "账本游标超过当前扫描序列长度",
                block_ordinal=sequence[-1].block_ordinal if sequence else 1,
                field="cursor",
            )
        remaining = sequence[cursor:]
        selected = remaining if max_tasks is None else remaining[: max(0, max_tasks)]
        if not selected:
            return {
                "status": "no_pending_tasks",
                "cursor_before": cursor,
                "cursor_after": cursor,
                "tasks_total": len(sequence),
                "committed_tasks": 0,
                "batches": [],
                "metrics": {},
            }
        chunks: list[list[ScanTask]] = [
            selected[index : index + batch_size]
            for index in range(0, len(selected), batch_size)
        ]
        if not commit_partial_tail and len(chunks[-1]) < batch_size:
            chunks.pop()
        if not chunks:
            return {
                "status": "no_complete_batch",
                "cursor_before": cursor,
                "cursor_after": cursor,
                "tasks_total": len(sequence),
                "committed_tasks": 0,
                "batches": [],
                "metrics": {},
            }
        tasks_to_run = [task for chunk in chunks for task in chunk]
        report = execute_tasks(
            tasks_to_run,
            bundle=self.bundle,
            image=self.image,
            budget=active_budget,
            delay_ms=delay_ms,
        )
        results_by_task = {
            result.task.task_id: result for result in report.task_results
        }
        current_payloads = {
            row["task_id"]: row["result_json"]
            for row in self.ledger.current_task_rows()
        }
        current_qkeys = {
            (row["query_id"], row["row_group_id"])
            for row in self.ledger.current_query_results()
        }
        existing_plans = {row["query_id"] for row in self.ledger.current_plans()}
        committed_batches: list[int] = []
        position = cursor
        for chunk_index, chunk in enumerate(chunks):
            chunk_results = [results_by_task[task.task_id] for task in chunk]
            for result in chunk_results:
                current_payloads[result.task.task_id] = result.result_json
            position += len(chunk)
            computed = compute_query_results(
                self.spec, self.bundle, current_payloads.values()
            )
            new_qresults = {
                key: value
                for key, value in computed.items()
                if key not in current_qkeys
            }
            plans_to_insert = (
                self.plan_payloads()
                if chunk_index == 0 and not existing_plans
                else {}
            )
            batch_id = self.ledger.commit_batch(
                generation_id=int(generation["generation_id"]),
                reason="scan-pending",
                task_results=chunk_results,
                query_results=new_qresults,
                plans=plans_to_insert,
                cursor_before=position - len(chunk),
                cursor_after=position,
                tasks_total=len(sequence),
                budget_version=self.versions["budget_params_version"],
                encoding_version=self.versions["encoding_selection_version"],
                segment_digest=self.image.file_digest,
                plan_digest=self.bundle.digest,
            )
            committed_batches.append(batch_id)
            current_qkeys.update(new_qresults.keys())
        return {
            "status": "committed",
            "cursor_before": cursor,
            "cursor_after": position,
            "tasks_total": len(sequence),
            "committed_tasks": position - cursor,
            "batches": committed_batches,
            "metrics": report.schedule.metrics,
        }

    def run_full_scan(
        self,
        *,
        budget: BudgetConfig | None = None,
        batch_size: int = 4,
        delay_ms: float = 0.0,
    ) -> dict[str, Any]:
        return self.scan_pending(
            budget=budget,
            batch_size=batch_size,
            delay_ms=delay_ms,
        )

    def ledger_statistics(self) -> dict[str, Any]:
        stats = self.ledger.statistics()
        stats["budget"] = self.budget.to_dict()
        stats["versions"] = self.versions
        stats["segment_digest"] = self.image.file_digest
        stats["generation"] = self.ledger.current_generation()
        return stats

    def _normalize_changes(self, changes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen_blocks: set[str] = set()
        block_by_location = {
            (block.column, block.row_group): block for block in self.spec.blocks
        }
        for index, change in enumerate(changes):
            if not isinstance(change, dict):
                raise AppError(
                    "E_SPEC", "变更项必须为对象", block_id=f"<change:{index}>", field="changes"
                )
            block_id = change.get("block_id")
            if block_id is not None:
                if not isinstance(block_id, str) or block_id not in self.image.blocks:
                    raise AppError(
                        "E_SPEC",
                        f"变更引用了不存在的块 {block_id!r}",
                        block_id=str(block_id),
                        field="block_id",
                    )
                block = next(item for item in self.spec.blocks if item.id == block_id)
            else:
                column = change.get("column")
                row_group = change.get("row_group")
                block = block_by_location.get((column, row_group))
                if block is None:
                    raise AppError(
                        "E_SPEC",
                        f"变更引用了不存在的列/行组 {column!r}/{row_group!r}",
                        block_id=f"{column}@{row_group}",
                        field="column",
                    )
            if block.id in seen_blocks:
                raise AppError(
                    "E_SPEC",
                    "同一变更集重复修改同一块",
                    block_id=block.id,
                    field="changes",
                )
            seen_blocks.add(block.id)
            values: tuple[int, ...] | None = None
            if "values" in change:
                raw_values = change["values"]
                if not isinstance(raw_values, list):
                    raise AppError(
                        "E_SPEC", "values 必须为数组", block_id=block.id, field="values"
                    )
                row_group = self.spec.row_group_map[block.row_group]
                if len(raw_values) != row_group.row_count:
                    raise AppError(
                        "E_SPEC",
                        "更新样本值行数与行组不一致",
                        block_id=block.id,
                        field="values",
                    )
                for value in raw_values:
                    if (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < -(2**63)
                        or value > 2**63 - 1
                    ):
                        raise AppError(
                            "E_SPEC",
                            "更新样本值必须为有符号 64 位整数",
                            block_id=block.id,
                            field="values",
                        )
                values = tuple(raw_values)
            encoding = change.get("encoding")
            if encoding is not None and encoding not in ENCODINGS:
                raise AppError(
                    "E_SPEC",
                    f"encoding 必须属于 {', '.join(ENCODINGS)}",
                    block_id=block.id,
                    field="encoding",
                )
            compressed = change.get("compressed")
            if compressed is not None and not isinstance(compressed, bool):
                raise AppError(
                    "E_SPEC",
                    "compressed 必须为布尔值",
                    block_id=block.id,
                    field="compressed",
                )
            if values is None and encoding is None and compressed is None:
                raise AppError(
                    "E_SPEC",
                    "变更项未包含 values、encoding 或 compressed",
                    block_id=block.id,
                    field="changes",
                )
            normalized.append(
                {
                    "block": block,
                    "values": values,
                    "encoding": encoding,
                    "compressed": compressed,
                }
            )
        return normalized

    def _write_current_state_files(
        self,
        *,
        spec: SegmentSpec,
        encoded_blocks: dict[str, Any],
        encoding_version: int,
    ) -> SegmentImage:
        write_spec(spec, self.spec_path)
        return write_segment(
            self.segment_path,
            spec,
            [encoded_blocks[block.id] for block in spec.blocks],
            encoding_selection_version=encoding_version,
        )

    def _restore_state_files(
        self,
        *,
        old_spec: SegmentSpec,
        old_image: SegmentImage,
        old_config: dict[str, Any],
    ) -> None:
        write_spec(old_spec, self.spec_path)
        atomic_write_bytes(self.segment_path, old_image.file_bytes)
        atomic_write_json(self.config_path, old_config)

    @staticmethod
    def _task_id(query_id: str, row_group_id: str, block_id: str) -> str:
        return f"q:{query_id}|rg:{row_group_id}|block:{block_id}"

    def _query_ids_referencing(self, column: str) -> list[str]:
        result: list[str] = []
        for plan in self.bundle.plans:
            if column in plan.select or column in plan.predicate_columns:
                result.append(plan.query_id)
        return result

    def _current_conclusions(self) -> dict[str, str]:
        conclusions: dict[str, str] = {}
        for row in self.ledger.current_task_rows():
            conclusions[f"task:{row['task_id']}"] = row["result_digest"]
        for row in self.ledger.current_query_results():
            conclusions[f"query:{row['query_id']}:{row['row_group_id']}"] = row[
                "result_digest"
            ]
        return conclusions

    @staticmethod
    def conclusion_diff(
        before: dict[str, str], after: dict[str, str]
    ) -> dict[str, Any]:
        before_keys = set(before)
        after_keys = set(after)
        added = sorted(after_keys - before_keys)
        deleted = sorted(before_keys - after_keys)
        changed = sorted(
            key for key in before_keys & after_keys if before[key] != after[key]
        )
        unchanged = sorted(
            key for key in before_keys & after_keys if before[key] == after[key]
        )
        return {
            "added": added,
            "deleted": deleted,
            "changed": changed,
            "unchanged": unchanged,
            "counts": {
                "added": len(added),
                "deleted": len(deleted),
                "changed": len(changed),
                "unchanged": len(unchanged),
            },
        }

    def apply_changeset(
        self,
        changes: Iterable[dict[str, Any]],
        *,
        budget: BudgetConfig | None = None,
        delay_ms: float = 0.0,
        reason: str = "incremental-update",
    ) -> dict[str, Any]:
        change_list = list(changes)
        if not change_list:
            raise AppError("E_SPEC", "变更集不能为空", block_id="<changeset>", field="changes")
        normalized = self._normalize_changes(change_list)
        active_budget = budget or self.budget
        validate_budget(active_budget, self.bundle.tasks)
        old_spec = self.spec
        old_image = self.image
        old_bundle = self.bundle
        old_config = json.loads(canonical_json(self.config))
        old_state = self.ledger.export_active_state()
        old_conclusions = self._current_conclusions()
        old_task_rows = {row["task_id"]: row for row in old_state["tasks"]}
        old_generation = old_state["generation"]
        old_cursor = int(old_generation["cursor"])

        changed_blocks: list[BlockSpec] = []
        replacement_encoded: dict[str, Any] = {}
        for item in normalized:
            source: BlockSpec = item["block"]
            updated = source.with_update(
                values=item["values"],
                encoding=item["encoding"],
                compressed=item["compressed"],
            )
            encoded = encode_block(
                block_id=updated.id,
                column=updated.column,
                row_group=updated.row_group,
                encoding=updated.encoding,
                compressed=updated.compressed,
                values=updated.values,
            )
            updated = BlockSpec(
                id=updated.id,
                column=updated.column,
                row_group=updated.row_group,
                encoding=updated.encoding,
                compressed=updated.compressed,
                values=updated.values,
                declared_pre_bytes=encoded.pre_bytes,
                declared_post_bytes=encoded.post_bytes,
            )
            replacement_encoded[updated.id] = encoded
            changed_blocks.append(updated)
        changed_by_id = {block.id: block for block in changed_blocks}
        changed_block_ids = [block.id for block in old_spec.blocks if block.id in changed_by_id]
        changed_columns = sorted(
            {changed_by_id[block_id].column for block_id in changed_block_ids},
            key=self.spec.column_names.index,
        )
        changed_row_groups = sorted(
            {changed_by_id[block_id].row_group for block_id in changed_block_ids},
            key=[group.id for group in self.spec.row_groups].index,
        )
        encoding_changed = any(
            changed_by_id[block_id].encoding != old_spec.block_map[block_id].encoding
            for block_id in changed_block_ids
        )
        new_encoding_version = self.versions["encoding_selection_version"] + (
            1 if encoding_changed else 0
        )
        new_spec = SegmentSpec(
            segment_id=old_spec.segment_id,
            columns=old_spec.columns,
            row_groups=old_spec.row_groups,
            blocks=tuple(changed_by_id.get(block.id, block) for block in old_spec.blocks),
            queries=old_spec.queries,
            format_version=old_spec.format_version,
        )
        existing_encoded = dict(old_image.blocks)
        existing_encoded.update(replacement_encoded)
        try:
            if encoding_changed:
                self.config["versions"]["encoding_selection_version"] = new_encoding_version
                self._save_config()
            new_image = self._write_current_state_files(
                spec=new_spec,
                encoded_blocks=existing_encoded,
                encoding_version=new_encoding_version,
            )
            self.spec = new_spec
            self.image = new_image
            self.bundle = self._build_bundle()
            affected_queries = sorted(
                {
                    query_id
                    for column in changed_columns
                    for query_id in self._query_ids_referencing(column)
                },
                key=[plan.query_id for plan in self.bundle.plans].index,
            )
            old_plan_map = old_bundle.plan_map
            new_plan_map = self.bundle.plan_map
            old_tasks_replace: set[str] = set()
            new_tasks_scan: set[str] = set()
            affected_qkeys: set[tuple[str, str]] = set()
            transition_details: list[dict[str, Any]] = []
            new_task_map = {task.task_id: task for task in self.bundle.tasks}
            old_task_map = {task.task_id: task for task in old_bundle.tasks}
            for query_id in affected_queries:
                old_plan = old_plan_map[query_id]
                new_plan = new_plan_map[query_id]
                old_groups = {
                    item["row_group_id"]: item for item in old_plan.row_groups
                }
                new_groups = {
                    item["row_group_id"]: item for item in new_plan.row_groups
                }
                for row_group_id in changed_row_groups:
                    old_candidates = set(old_groups[row_group_id]["candidate_blocks"])
                    new_candidates = set(new_groups[row_group_id]["candidate_blocks"])
                    if old_candidates or new_candidates:
                        affected_qkeys.add((query_id, row_group_id))
                    if old_candidates and new_candidates:
                        for block_id in changed_block_ids:
                            if changed_by_id[block_id].row_group != row_group_id:
                                continue
                            task_id = self._task_id(query_id, row_group_id, block_id)
                            old_tasks_replace.add(task_id)
                            new_tasks_scan.add(task_id)
                        transition = "candidate-retained"
                    elif new_candidates and not old_candidates:
                        for block_id in new_candidates:
                            task_id = self._task_id(query_id, row_group_id, block_id)
                            new_tasks_scan.add(task_id)
                        transition = "candidate-added"
                    elif old_candidates and not new_candidates:
                        for block_id in old_candidates:
                            task_id = self._task_id(query_id, row_group_id, block_id)
                            old_tasks_replace.add(task_id)
                        transition = "candidate-removed"
                    else:
                        transition = "remained-pruned"
                    transition_details.append(
                        {
                            "query_id": query_id,
                            "row_group_id": row_group_id,
                            "transition": transition,
                            "old_candidate_blocks": sorted(old_candidates),
                            "new_candidate_blocks": sorted(new_candidates),
                        }
                    )
            tasks_to_scan = sorted(
                (new_task_map[task_id] for task_id in new_tasks_scan),
                key=lambda task: task.block_ordinal,
            )
            scan_report = execute_tasks(
                tasks_to_scan,
                bundle=self.bundle,
                image=self.image,
                budget=active_budget,
                delay_ms=delay_ms,
            ) if tasks_to_scan else None
            new_results = (
                list(scan_report.task_results) if scan_report is not None else []
            )
            payloads = {
                row["task_id"]: row["result_json"]
                for row in old_state["tasks"]
                if row["task_id"] not in old_tasks_replace
            }
            for result in new_results:
                payloads[result.task.task_id] = result.result_json
            computed = compute_query_results(
                self.spec,
                self.bundle,
                payloads.values(),
                query_ids=set(affected_queries),
                row_group_ids=set(changed_row_groups),
            )
            qresult_updates = {
                key: value for key, value in computed.items() if key in affected_qkeys
            }
            affected_plans = {
                query_id: plan_to_dict(
                    self.bundle.plan_map[query_id],
                    self.bundle.query_asts[query_id],
                )
                for query_id in affected_queries
            }
            if encoding_changed:
                generation_id = self.ledger.clone_current_generation(
                    reason=f"{reason}:encoding-version",
                    budget=active_budget,
                    budget_version=self.versions["budget_params_version"],
                    encoding_version=new_encoding_version,
                    segment_digest=new_image.file_digest,
                    spec_digest=new_image.spec_digest,
                    plan_digest=self.bundle.digest,
                    tasks_total=len(self.bundle.tasks),
                    cursor=old_cursor,
                )
            else:
                generation_id = int(old_generation["generation_id"])
            if affected_queries:
                self.ledger.commit_batch(
                    generation_id=generation_id,
                    reason=reason,
                    task_results=new_results,
                    query_results=qresult_updates,
                    plans=affected_plans,
                    cursor_before=old_cursor,
                    cursor_after=len(self.bundle.tasks),
                    tasks_total=len(self.bundle.tasks),
                    budget_version=self.versions["budget_params_version"],
                    encoding_version=new_encoding_version,
                    segment_digest=new_image.file_digest,
                    plan_digest=self.bundle.digest,
                    replace_task_ids=sorted(old_tasks_replace),
                    replace_query_result_keys=sorted(affected_qkeys),
                    replace_query_ids=affected_queries,
                )
            else:
                self.ledger.update_generation_progress(
                    generation_id=generation_id,
                    cursor=len(self.bundle.tasks),
                    tasks_total=len(self.bundle.tasks),
                    plan_digest=self.bundle.digest,
                    segment_digest=new_image.file_digest,
                )
            self.config["versions"]["encoding_selection_version"] = new_encoding_version
            self._save_config()
        except Exception:
            self.spec = old_spec
            self.image = old_image
            self.bundle = old_bundle
            self.config = old_config
            try:
                self.ledger.import_as_current(
                    old_state, reason="rollback-after-failed-incremental-update"
                )
            except Exception:
                pass
            try:
                self._restore_state_files(
                    old_spec=old_spec, old_image=old_image, old_config=old_config
                )
            except Exception:
                pass
            raise

        after_conclusions = self._current_conclusions()
        new_state = self.ledger.export_active_state()
        new_task_rows = {row["task_id"]: row for row in new_state["tasks"]}
        preserved_blocks: list[dict[str, Any]] = []
        for block in old_spec.blocks:
            if block.id in changed_by_id:
                continue
            before = old_image.blocks[block.id]
            after = self.image.blocks[block.id]
            preserved_blocks.append(
                {
                    "block_id": block.id,
                    "block_digest_before": before.block_digest,
                    "block_digest_after": after.block_digest,
                    "payload_sha256_before": sha256_bytes(before.payload),
                    "payload_sha256_after": sha256_bytes(after.payload),
                    "record_sha256_before": sha256_bytes(before.record_bytes),
                    "record_sha256_after": sha256_bytes(after.record_bytes),
                    "unchanged": (
                        before.block_digest == after.block_digest
                        and before.payload == after.payload
                        and before.record_bytes == after.record_bytes
                    ),
                }
            )
        preserved_tasks: list[dict[str, Any]] = []
        for task_id, before_row in sorted(
            old_task_rows.items(), key=lambda item: item[1]["block_ordinal"]
        ):
            if task_id in old_tasks_replace:
                continue
            after_row = new_task_rows.get(task_id)
            preserved_tasks.append(
                {
                    "task_id": task_id,
                    "result_digest_before": before_row["result_digest"],
                    "result_digest_after": after_row["result_digest"] if after_row else None,
                    "block_digest_before": before_row["block_digest"],
                    "block_digest_after": after_row["block_digest"] if after_row else None,
                    "unchanged": bool(
                        after_row
                        and before_row["result_digest"] == after_row["result_digest"]
                        and before_row["block_digest"] == after_row["block_digest"]
                        and before_row["result_json"] == after_row["result_json"]
                    ),
                }
            )
        preserved_query_results: list[dict[str, Any]] = []
        old_qresult_map = {
            (row["query_id"], row["row_group_id"]): row
            for row in old_state["query_results"]
        }
        new_qresult_map = {
            (row["query_id"], row["row_group_id"]): row
            for row in new_state["query_results"]
        }
        for key, before_row in sorted(old_qresult_map.items()):
            if key in affected_qkeys:
                continue
            after_row = new_qresult_map.get(key)
            preserved_query_results.append(
                {
                    "query_id": key[0],
                    "row_group_id": key[1],
                    "result_digest_before": before_row["result_digest"],
                    "result_digest_after": after_row["result_digest"] if after_row else None,
                    "rows_before_sha256": sha256_bytes(
                        canonical_json(before_row["rows"]).encode("utf-8")
                    ),
                    "rows_after_sha256": (
                        sha256_bytes(canonical_json(after_row["rows"]).encode("utf-8"))
                        if after_row
                        else None
                    ),
                    "unchanged": bool(
                        after_row
                        and before_row["result_digest"] == after_row["result_digest"]
                        and before_row["rows"] == after_row["rows"]
                    ),
                }
            )
        block_order = [block.id for block in old_spec.blocks]
        indirect_block_ids = sorted(
            {
                new_task_map[task_id].block_id
                for task_id in new_tasks_scan
                if new_task_map[task_id].block_id not in changed_by_id
            }
            | {
                old_task_map[task_id].block_id
                for task_id in old_tasks_replace
                if old_task_map[task_id].block_id not in changed_by_id
            },
            key=block_order.index,
        )
        propagation_steps: list[dict[str, Any]] = [
            {
                "step": 1,
                "from": "changeset",
                "to": changed_block_ids,
                "basis": "直接修改了样本值、编码方式或压缩标记",
            },
            {
                "step": 2,
                "from": changed_columns,
                "to": affected_queries,
                "basis": "查询在 select 或谓词中引用变更列",
            },
            {
                "step": 3,
                "from": [f"query:{query_id}" for query_id in affected_queries],
                "to": sorted(
                    old_tasks_replace | new_tasks_scan,
                    key=lambda task_id: (
                        new_task_map.get(task_id) or old_task_map[task_id]
                    ).block_ordinal,
                ),
                "basis": "沿查询×行组×块引用关系传播；仅变更块及其查询结论失效",
            },
            {
                "step": 4,
                "from": sorted(f"query:{q}:{rg}" for q, rg in affected_qkeys),
                "to": ["row-merge", "result-summary"],
                "basis": "块级新旧结果合并，未失效块扫描结果直接复用",
            },
        ]
        generation = self.ledger.current_generation()
        assert generation is not None
        return {
            "reason": reason,
            "changed_blocks": changed_block_ids,
            "invalid_blocks": changed_block_ids,
            "indirectly_affected_blocks": indirect_block_ids,
            "affected_block_set": sorted(
                set(changed_block_ids) | set(indirect_block_ids), key=block_order.index
            ),
            "affected_columns": changed_columns,
            "affected_row_groups": changed_row_groups,
            "affected_queries": affected_queries,
            "affected_tasks": sorted(
                old_tasks_replace | new_tasks_scan,
                key=lambda task_id: (
                    new_task_map.get(task_id) or old_task_map[task_id]
                ).block_ordinal,
            ),
            "recomputed_encoding_blocks": changed_block_ids,
            "recomputed_scan_tasks": [result.task.task_id for result in new_results],
            "propagation_steps": propagation_steps,
            "candidate_transitions": transition_details,
            "preserved_blocks": preserved_blocks,
            "preserved_task_results": preserved_tasks,
            "preserved_query_results": preserved_query_results,
            "all_preserved_blocks_byte_identical": all(
                item["unchanged"] for item in preserved_blocks
            ),
            "all_preserved_tasks_identical": all(
                item["unchanged"] for item in preserved_tasks
            ),
            "all_preserved_query_results_identical": all(
                item["unchanged"] for item in preserved_query_results
            ),
            "metrics": scan_report.schedule.metrics if scan_report else {},
            "versions": self.versions,
            "encoding_version_bumped": encoding_changed,
            "generation_id": generation["generation_id"],
            "conclusion_diff": self.conclusion_diff(old_conclusions, after_conclusions),
        }

    def change_budget(
        self,
        budget: BudgetConfig,
        *,
        batch_size: int = 4,
        delay_ms: float = 0.0,
    ) -> dict[str, Any]:
        validate_budget(budget, self.bundle.tasks)
        if budget == self.budget:
            return {
                "version_changed": False,
                "budget": budget.to_dict(),
                "conclusion_diff": {"added": [], "deleted": [], "changed": [], "unchanged": [], "counts": {"added": 0, "deleted": 0, "changed": 0, "unchanged": 0}},
            }
        before = self._current_conclusions()
        before_state = self.ledger.export_active_state()
        old_budget = self.budget
        old_config = json.loads(canonical_json(self.config))
        self.budget = budget
        self.config["versions"]["budget_params_version"] = (
            int(self.config["versions"]["budget_params_version"]) + 1
        )
        self._save_config()
        self.bundle = self._build_bundle()
        try:
            generation_id = self.ledger.begin_generation(
                reason="budget-version-change",
                budget=self.budget,
                budget_version=self.versions["budget_params_version"],
                encoding_version=self.versions["encoding_selection_version"],
                segment_digest=self.image.file_digest,
                spec_digest=self.image.spec_digest,
                plan_digest=self.bundle.digest,
                tasks_total=len(self.bundle.tasks),
                cursor=0,
            )
            scan = self.scan_pending(
                budget=self.budget,
                batch_size=batch_size,
                delay_ms=delay_ms,
            )
        except Exception:
            self.budget = old_budget
            self.config = old_config
            self._save_config()
            self.bundle = self._build_bundle()
            try:
                self.ledger.import_as_current(
                    before_state, reason="rollback-after-failed-budget-change"
                )
            except Exception:
                pass
            raise
        after = self._current_conclusions()
        return {
            "version_changed": True,
            "generation_id": generation_id,
            "budget": self.budget.to_dict(),
            "budget_params_version": self.versions["budget_params_version"],
            "scan": scan,
            "conclusion_diff": self.conclusion_diff(before, after),
            "stale_generations": [
                row
                for row in self.ledger.generation_rows()
                if not int(row["is_current"])
            ],
        }

    def change_encoding_policy(
        self,
        *,
        column: str,
        encoding: str,
        compressed: bool,
        budget: BudgetConfig | None = None,
        delay_ms: float = 0.0,
    ) -> dict[str, Any]:
        if column not in self.spec.column_names:
            raise AppError(
                "E_SPEC",
                f"列不存在: {column}",
                block_id=f"column:{column}",
                field="column",
            )
        if encoding not in ENCODINGS:
            raise AppError(
                "E_SPEC",
                f"encoding 必须属于 {', '.join(ENCODINGS)}",
                block_id=f"column:{column}",
                field="encoding",
            )
        changes = [
            {
                "block_id": block.id,
                "encoding": encoding,
                "compressed": compressed,
            }
            for block in self.spec.blocks
            if block.column == column
            and (block.encoding != encoding or block.compressed != compressed)
        ]
        if not changes:
            return {
                "version_changed": False,
                "column": column,
                "encoding": encoding,
                "compressed": compressed,
                "conclusion_diff": {"added": [], "deleted": [], "changed": [], "unchanged": [], "counts": {"added": 0, "deleted": 0, "changed": 0, "unchanged": 0}},
            }
        result = self.apply_changeset(
            changes,
            budget=budget,
            delay_ms=delay_ms,
            reason="encoding-policy-change",
        )
        result["version_changed"] = result.get("encoding_version_bumped", False)
        result["encoding_selection_version"] = self.versions[
            "encoding_selection_version"
        ]
        return result
