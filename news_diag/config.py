"""Configuration parsing and validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigInvalid
from .util import content_fingerprint, sha256_file


def _as_decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ConfigInvalid(f"{name} 必须是数值", details={"field": name}) from exc
    if not result.is_finite():
        raise ConfigInvalid(f"{name} 必须是有限数值", details={"field": name})
    return result


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigInvalid(f"{name} 必须是正整数", details={"field": name, "value": value})
    return value


def _as_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigInvalid(f"{name} 必须是非空字符串", details={"field": name, "value": value})
    return value


@dataclass(frozen=True)
class WorkbenchConfig:
    raw: dict[str, Any]
    max_summary_sentences: int
    word_frequency_weight: Decimal
    entity_coverage_weight: Decimal
    position_decay_weight: Decimal
    entity_coverage_min: Decimal
    position_decay_base: Decimal
    fixed_decimal_places: int
    thread_pool_size: int
    snapshot_commit_interval_docs: int
    style_template_id: str
    synonym_table_version: str
    max_rewrite_chars: int
    vocabulary: tuple[str, ...]
    base_config_hash: str
    source_path: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "WorkbenchConfig":
        config_path = Path(path)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except Exception as exc:
            raise ConfigInvalid(f"无法解析配置文件: {exc}", details={"path": str(config_path)}) from exc
        if not isinstance(raw, dict):
            raise ConfigInvalid("配置根节点必须是映射", details={"path": str(config_path)})
        return cls.from_raw(raw, source_path=str(config_path),
                            base_config_hash=sha256_file(config_path))

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, source_path: str = "",
                 base_config_hash: str | None = None) -> "WorkbenchConfig":
        data = copy.deepcopy(raw)
        scoring = data.get("scoring")
        if not isinstance(scoring, dict):
            raise ConfigInvalid("scoring 必须是映射", details={"field": "scoring"})
        rewrite = data.get("rewrite")
        if not isinstance(rewrite, dict):
            raise ConfigInvalid("rewrite 必须是映射", details={"field": "rewrite"})

        max_sentences = _as_positive_int(data.get("max_summary_sentences"), "max_summary_sentences")
        word_weight = _as_decimal(scoring.get("word_frequency_weight"), "scoring.word_frequency_weight")
        entity_weight = _as_decimal(scoring.get("entity_coverage_weight"), "scoring.entity_coverage_weight")
        position_weight = _as_decimal(scoring.get("position_decay_weight"), "scoring.position_decay_weight")
        if min(word_weight, entity_weight, position_weight) < 0:
            raise ConfigInvalid("句子打分权重不得为负数", details={"field": "scoring"})
        if word_weight + entity_weight + position_weight <= 0:
            raise ConfigInvalid("句子打分权重之和必须大于零", details={"field": "scoring"})
        entity_min = _as_decimal(scoring.get("entity_coverage_min", "0"), "scoring.entity_coverage_min")
        if entity_min < 0 or entity_min > 1:
            raise ConfigInvalid("scoring.entity_coverage_min 必须位于 [0,1]", details={"field": "scoring.entity_coverage_min"})
        decay_base = _as_decimal(scoring.get("position_decay_base", "1"), "scoring.position_decay_base")
        if decay_base <= 0 or decay_base > 1:
            raise ConfigInvalid("scoring.position_decay_base 必须位于 (0,1]", details={"field": "scoring.position_decay_base"})

        precision = data.get("fixed_decimal_places")
        if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 9:
            raise ConfigInvalid("fixed_decimal_places 必须是 0..9 的整数", details={"field": "fixed_decimal_places", "value": precision})
        threads = data.get("thread_pool_size")
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
            raise ConfigInvalid("thread_pool_size 必须是 1..64 的整数", details={"field": "thread_pool_size", "value": threads})
        interval = _as_positive_int(data.get("snapshot_commit_interval_docs"), "snapshot_commit_interval_docs")

        style_id = _as_nonempty_string(rewrite.get("style_template_id"), "rewrite.style_template_id")
        synonym_version = _as_nonempty_string(
            rewrite.get("synonym_table_version", data.get("synonym_table_version")),
            "rewrite.synonym_table_version",
        )
        max_chars = rewrite.get("max_chars", data.get("max_rewrite_chars", 96))
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ConfigInvalid("rewrite.max_chars 必须是正整数", details={"field": "rewrite.max_chars", "value": max_chars})

        synonyms = data.get("synonym_tables")
        if not isinstance(synonyms, dict) or not synonyms:
            raise ConfigInvalid("synonym_tables 必须是非空映射", details={"field": "synonym_tables"})
        _validate_synonym_table_version(synonyms, synonym_version, style_id)

        vocabulary_raw = data.get("vocabulary", [])
        if not isinstance(vocabulary_raw, list) or any(not isinstance(item, str) or not item for item in vocabulary_raw):
            raise ConfigInvalid("vocabulary 必须是非空字符串列表", details={"field": "vocabulary"})

        stable_raw = copy.deepcopy(data)
        config_hash = base_config_hash or content_fingerprint(stable_raw)
        return cls(
            raw=stable_raw,
            max_summary_sentences=max_sentences,
            word_frequency_weight=word_weight,
            entity_coverage_weight=entity_weight,
            position_decay_weight=position_weight,
            entity_coverage_min=entity_min,
            position_decay_base=decay_base,
            fixed_decimal_places=precision,
            thread_pool_size=threads,
            snapshot_commit_interval_docs=interval,
            style_template_id=style_id,
            synonym_table_version=synonym_version,
            max_rewrite_chars=max_chars,
            vocabulary=tuple(vocabulary_raw),
            base_config_hash=config_hash,
            source_path=source_path,
        )

    @property
    def synonym_table(self) -> dict[str, Any]:
        return self.raw["synonym_tables"][self.synonym_table_version]

    @property
    def template(self) -> dict[str, Any]:
        return self.synonym_table["templates"][self.style_template_id]

    def with_overrides(self, overrides: dict[str, Any]) -> "WorkbenchConfig":
        if not overrides:
            return self
        raw = copy.deepcopy(self.raw)
        _apply_overrides_to_raw(raw, overrides)
        return WorkbenchConfig.from_raw(raw, source_path=self.source_path,
                                        base_config_hash=self.base_config_hash)

    def overrides_projection(self) -> dict[str, Any]:
        """Fields that commands are allowed to adjust, in a stable shape."""

        return {
            "max_summary_sentences": self.max_summary_sentences,
            "fixed_decimal_places": self.fixed_decimal_places,
            "scoring": {
                "word_frequency_weight": str(self.word_frequency_weight),
                "entity_coverage_weight": str(self.entity_coverage_weight),
                "position_decay_weight": str(self.position_decay_weight),
                "entity_coverage_min": str(self.entity_coverage_min),
                "position_decay_base": str(self.position_decay_base),
            },
            "rewrite": {
                "style_template_id": self.style_template_id,
                "synonym_table_version": self.synonym_table_version,
                "max_chars": self.max_rewrite_chars,
            },
        }

    def stage_config_fingerprint(self, stage: str) -> str:
        if stage == "split":
            value = {"algorithm": "split-v1"}
        elif stage == "scoring":
            value = {
                "algorithm": "score-v1",
                "weights": [str(self.word_frequency_weight), str(self.entity_coverage_weight),
                            str(self.position_decay_weight)],
                "position_decay_base": str(self.position_decay_base),
                "fixed_decimal_places": self.fixed_decimal_places,
                "vocabulary": list(self.vocabulary),
            }
        elif stage == "selection":
            value = {
                "algorithm": "select-v1",
                "max_summary_sentences": self.max_summary_sentences,
                "entity_coverage_min": str(self.entity_coverage_min),
            }
        elif stage == "rewrite":
            value = {
                "algorithm": "rewrite-v1",
                "style_template_id": self.style_template_id,
                "synonym_table_version": self.synonym_table_version,
                "synonym_table": self.synonym_table,
                "max_rewrite_chars": self.max_rewrite_chars,
            }
        elif stage == "summary":
            value = {"algorithm": "summary-v1", "synonym_table_version": self.synonym_table_version}
        else:
            raise ValueError(f"unknown stage: {stage}")
        return content_fingerprint(value)


def _apply_overrides_to_raw(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    allowed_top = {"max_summary_sentences", "fixed_decimal_places"}
    for key in allowed_top:
        if key in overrides:
            raw[key] = overrides[key]
    scoring = overrides.get("scoring")
    if isinstance(scoring, dict):
        raw.setdefault("scoring", {}).update(scoring)
    rewrite = overrides.get("rewrite")
    if isinstance(rewrite, dict):
        target = raw.setdefault("rewrite", {})
        if "max_chars" in rewrite:
            target["max_chars"] = rewrite["max_chars"]
        if "style_template_id" in rewrite:
            target["style_template_id"] = rewrite["style_template_id"]
        if "synonym_table_version" in rewrite:
            target["synonym_table_version"] = rewrite["synonym_table_version"]


def _validate_synonym_table_version(synonyms: dict[str, Any], version: str, style_id: str) -> None:
    table = synonyms.get(version)
    if not isinstance(table, dict):
        raise ConfigInvalid(f"同义替换表版本不存在: {version}", details={"synonym_table_version": version})
    templates = table.get("templates")
    if not isinstance(templates, dict) or not templates:
        raise ConfigInvalid(f"同义替换表版本 {version} 未声明 templates", details={"synonym_table_version": version})
    if style_id not in templates:
        raise ConfigInvalid(f"风格模板 ID {style_id} 未在版本 {version} 中声明",
                            details={"style_template_id": style_id, "synonym_table_version": version})
    for template_id, template in templates.items():
        if not isinstance(template_id, str) or not template_id:
            raise ConfigInvalid("模板 ID 必须是非空字符串", details={"synonym_table_version": version})
        if not isinstance(template, dict):
            raise ConfigInvalid(f"模板 {template_id} 必须是映射", details={"synonym_table_version": version})
        for field in ("prefix", "suffix"):
            if field in template and not isinstance(template[field], str):
                raise ConfigInvalid(f"模板 {template_id}.{field} 必须是字符串",
                                    details={"synonym_table_version": version})
        if "max_chars" in template:
            value = template["max_chars"]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigInvalid(f"模板 {template_id}.max_chars 必须是正整数",
                                    details={"synonym_table_version": version})
    entries = table.get("entries")
    if not isinstance(entries, list):
        raise ConfigInvalid(f"同义替换表版本 {version} 的 entries 必须是列表",
                            details={"synonym_table_version": version})
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigInvalid(f"同义替换表条目 {index} 必须是映射",
                                details={"synonym_table_version": version, "entry_index": index})
        template_id = entry.get("template_id")
        if template_id not in templates:
            raise ConfigInvalid(f"同义替换表条目 {index} 引用了不存在的模板: {template_id}",
                                details={"synonym_table_version": version, "entry_index": index,
                                         "style_template_id": template_id})
        source = entry.get("source")
        candidates = entry.get("candidates")
        if not isinstance(source, str) or not source:
            raise ConfigInvalid(f"同义替换表条目 {index}.source 必须是非空字符串",
                                details={"synonym_table_version": version, "entry_index": index})
        if not isinstance(candidates, list) or not candidates or any(
            not isinstance(candidate, str) or not candidate for candidate in candidates
        ):
            raise ConfigInvalid(f"同义替换表条目 {index}.candidates 必须是非空字符串列表",
                                details={"synonym_table_version": version, "entry_index": index})
