"""Deterministic editing-command application."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .config import WorkbenchConfig
from .errors import CommandInvalid, ConfigInvalid, DocumentInvalid
from .loader import normalize_document
from .util import deep_copy_json


class CommandEngine:
    def __init__(self, config: WorkbenchConfig, documents: dict[str, dict[str, Any]],
                 overrides: dict[str, Any] | None = None,
                 batch_history: list[dict[str, Any]] | None = None) -> None:
        self.base_config = config
        self.overrides = copy.deepcopy(overrides or {})
        self.effective_config = self.base_config.with_overrides(self.overrides)
        self.documents = copy.deepcopy(documents)
        self.batch_history = copy.deepcopy(batch_history or [])
        self.failures: list[dict[str, Any]] = []

    def apply_records(self, records: list[Any], start_index: int) -> dict[str, Any]:
        """Apply top-level records whose record index is >= start_index."""

        applied_count = start_index
        for record in records:
            if record.index < start_index:
                continue
            applied_count = record.index + 1
            if record.error is not None:
                self._record_failure(
                    CommandInvalid(record.error, details={"record_index": record.index}),
                    command_index=record.index,
                )
                continue
            self._apply_top_level(record.value, record.index)
        return {
            "applied_command_count": applied_count,
            "documents": self.documents,
            "overrides": self.overrides,
            "effective_config": self.effective_config,
            "batch_history": self.batch_history,
            "failures": self.failures,
        }

    def _apply_top_level(self, command: Any, command_index: int) -> None:
        if not isinstance(command, dict):
            self._record_failure(CommandInvalid("命令必须是 JSON 对象"), command_index)
            return
        command_type = _command_type(command)
        if command_type in {"append_batch", "batch", "append_edits"}:
            self._apply_batch(command, command_index)
            return
        try:
            self._apply_single(command, command_index, command_type)
        except CommandInvalid as exc:
            self._record_failure(exc, command_index, command_type)
        except ConfigInvalid as exc:
            self._record_failure(
                CommandInvalid(f"命令导致配置非法: {exc.message}", document_id=exc.document_id,
                               details={**exc.details, "source_code": exc.code}),
                command_index,
                command_type,
            )
        except DocumentInvalid as exc:
            self._record_failure(
                CommandInvalid(f"命令导致文档非法: {exc.message}", document_id=exc.document_id,
                               details=exc.details),
                command_index,
                command_type,
            )
        except Exception as exc:  # Keep one bad command from terminating the stream.
            self._record_failure(
                CommandInvalid(f"命令执行失败: {exc}",
                               details={"exception_type": type(exc).__name__}),
                command_index,
                command_type,
            )

    def _apply_single(self, command: dict[str, Any], command_index: int,
                      command_type: str) -> None:
        if command_type in {"replace_document", "replace_doc"}:
            raw_document = command.get("document", command.get("payload"))
            if raw_document is None:
                raise CommandInvalid("replace_document 缺少 document")
            document = normalize_document(raw_document, stage="command")
            self.documents[document["document_id"]] = document
            return

        if command_type in {"add_paragraph", "insert_paragraph"}:
            document = self._target_document(command)
            paragraph = _first_present(command, "paragraph", "text", "content")
            if not isinstance(paragraph, str) or not paragraph.strip():
                raise CommandInvalid("add_paragraph.paragraph 必须是非空字符串",
                                     document_id=document["document_id"])
            index = command.get("index")
            if index is None:
                index = len(document["paragraphs"])
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= len(document["paragraphs"]):
                raise CommandInvalid("add_paragraph.index 越界", document_id=document["document_id"],
                                     details={"index": index})
            updated = copy.deepcopy(document)
            updated["paragraphs"].insert(index, paragraph)
            self.documents[updated["document_id"]] = normalize_document(updated, stage="command")
            return

        if command_type in {"remove_paragraph", "delete_paragraph"}:
            document = self._target_document(command)
            index = command.get("index", command.get("paragraph_index"))
            if isinstance(index, bool) or not isinstance(index, int):
                raise CommandInvalid("remove_paragraph.index 必须是整数", document_id=document["document_id"])
            if not 0 <= index < len(document["paragraphs"]):
                raise CommandInvalid("remove_paragraph.index 越界", document_id=document["document_id"],
                                     details={"index": index})
            if len(document["paragraphs"]) <= 1:
                raise CommandInvalid("不得删除文档的唯一段落", document_id=document["document_id"])
            updated = copy.deepcopy(document)
            updated["paragraphs"].pop(index)
            self.documents[updated["document_id"]] = normalize_document(updated, stage="command")
            return

        if command_type in {"set_max_summary_sentences", "set_summary_length"}:
            value = _first_present(command, "max_summary_sentences", "value", "length")
            self._set_overrides({"max_summary_sentences": value})
            return

        if command_type in {"set_style_template", "set_rewrite_style", "set_rewrite_style_template"}:
            if command_type == "set_rewrite_style" or isinstance(command.get("rewrite"), dict):
                rewrite = copy.deepcopy(command.get("rewrite", {}))
                for key in ("style_template_id", "template_id"):
                    if key in command:
                        rewrite["style_template_id"] = command[key]
                for key in ("synonym_table_version", "table_version"):
                    if key in command:
                        rewrite["synonym_table_version"] = command[key]
                if "max_chars" in command:
                    rewrite["max_chars"] = command["max_chars"]
                self._set_overrides({"rewrite": rewrite})
                return
            value = _first_present(command, "style_template_id", "template_id", "value", "id")
            self._set_overrides({"rewrite": {"style_template_id": value}})
            return

        if command_type in {"set_synonym_table_version", "set_replacement_table_version"}:
            value = _first_present(command, "synonym_table_version", "table_version", "value", "version")
            self._set_overrides({"rewrite": {"synonym_table_version": value}})
            return

        if command_type in {"set_scoring", "set_scoring_weights"}:
            scoring = copy.deepcopy(command.get("scoring", command.get("weights", {})))
            if not isinstance(scoring, dict):
                raise CommandInvalid("set_scoring.scoring 必须是映射")
            allowed = {"word_frequency_weight", "entity_coverage_weight", "position_decay_weight",
                       "entity_coverage_min", "position_decay_base"}
            unknown = set(scoring) - allowed
            if unknown:
                raise CommandInvalid(f"未知打分字段: {sorted(unknown)}")
            self._set_overrides({"scoring": scoring})
            return

        if command_type in {"set_fixed_decimal_places", "set_precision"}:
            value = _first_present(command, "fixed_decimal_places", "precision", "value")
            self._set_overrides({"fixed_decimal_places": value})
            return

        if command_type in {"rollback_batch", "rollback_last_batch", "rollback"}:
            self._rollback_batch(command)
            return

        raise CommandInvalid(f"未知命令类型: {command_type or '<empty>'}",
                             details={"command_type": command_type})

    def _apply_batch(self, command: dict[str, Any], command_index: int) -> None:
        nested = command.get("commands", command.get("edits"))
        if not isinstance(nested, list):
            self._record_failure(CommandInvalid("编辑批次 commands 必须是数组"),
                                 command_index, "append_batch")
            return
        batch_id = command.get("batch_id", f"batch-{command_index}")
        if not isinstance(batch_id, str) or not batch_id.strip():
            self._record_failure(CommandInvalid("batch_id 必须是非空字符串"),
                                 command_index, "append_batch")
            return
        self.batch_history.append({
            "batch_id": batch_id.strip(),
            "command_index": command_index,
            "documents": deep_copy_json(self.documents),
            "overrides": deep_copy_json(self.overrides),
        })
        for nested_index, nested_command in enumerate(nested):
            if not isinstance(nested_command, dict):
                self._record_failure(CommandInvalid("批次内命令必须是对象", details={
                    "batch_id": batch_id, "nested_index": nested_index,
                }), command_index, "append_batch")
                continue
            nested_type = _command_type(nested_command)
            try:
                self._apply_single(nested_command, command_index, nested_type)
            except CommandInvalid as exc:
                self._record_failure(exc, command_index, nested_type, batch_id=batch_id,
                                     nested_index=nested_index)
            except ConfigInvalid as exc:
                self._record_failure(CommandInvalid(f"批次命令导致配置非法: {exc.message}"),
                                     command_index, nested_type, batch_id=batch_id,
                                     nested_index=nested_index)
            except DocumentInvalid as exc:
                self._record_failure(CommandInvalid(f"批次命令导致文档非法: {exc.message}"),
                                     command_index, nested_type, batch_id=batch_id,
                                     nested_index=nested_index)

    def _rollback_batch(self, command: dict[str, Any]) -> None:
        if not self.batch_history:
            raise CommandInvalid("没有可回滚的编辑批次")
        requested = command.get("batch_id")
        latest = self.batch_history[-1]
        if requested is not None and requested != latest["batch_id"]:
            raise CommandInvalid(f"只能回滚最新批次 {latest['batch_id']}",
                                 details={"requested_batch_id": requested})
        self.documents = copy.deepcopy(latest["documents"])
        self.overrides = copy.deepcopy(latest["overrides"])
        self.effective_config = self.base_config.with_overrides(self.overrides)
        self.batch_history.pop()

    def _set_overrides(self, updates: dict[str, Any]) -> None:
        candidate = copy.deepcopy(self.overrides)
        merge_overrides(candidate, updates)
        config = self.base_config.with_overrides(candidate)
        self.overrides = candidate
        self.effective_config = config

    def _target_document(self, command: dict[str, Any]) -> dict[str, Any]:
        document_id = _first_present(command, "document_id", "doc_id", "id")
        if not isinstance(document_id, str) or not document_id:
            raise CommandInvalid("命令缺少 document_id")
        document = self.documents.get(document_id)
        if document is None:
            raise CommandInvalid(f"文档不存在: {document_id}", document_id=document_id)
        return document

    def _record_failure(self, error: CommandInvalid, command_index: int,
                        command_type: str | None = None, **extra: Any) -> None:
        failure = error.to_failure(
            command_index=command_index,
            command_type=command_type or "",
            **extra,
        )
        self.failures.append(failure)


def merge_overrides(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_overrides(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _command_type(command: dict[str, Any]) -> str:
    value = _first_present(command, "type", "command", "op", "action")
    return value.strip() if isinstance(value, str) else ""


def _first_present(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None
