from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .canonical import digest
from .config import PipelineConfig, parse_config
from .errors import Failure
from .models import InputSlot, ResolvedSlot, SampleInput, ValidationProducts
from .templates import (
    EffectiveTemplate,
    TemplateCatalog,
    parse_templates,
    render_template,
)


class WorkspaceLoadError(RuntimeError):
    pass


def read_yaml(path: Path, *, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise WorkspaceLoadError(f"cannot read {path}: {exc}") from exc
    return default if value is None else value


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
        handle.flush()
    temporary.replace(path)


@dataclass
class RawWorkspace:
    templates: list[dict[str, Any]]
    sample_batches: list[dict[str, Any]]
    config: dict[str, Any]
    injections: list[dict[str, Any]]
    sample_files: list[Path]

    def all_raw_samples(self) -> list[tuple[str, str, dict[str, Any]]]:
        result: list[tuple[str, str, dict[str, Any]]] = []
        for batch_index, batch in enumerate(self.sample_batches):
            batch_id = str(batch.get("batch_id", f"batch-{batch_index + 1:03d}"))
            path = str(self.sample_files[batch_index]) if batch_index < len(self.sample_files) else ""
            for sample in batch.get("samples", []) or []:
                if isinstance(sample, Mapping):
                    result.append((batch_id, path, dict(sample)))
        return result


def _template_list(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        value = raw.get("templates", raw.get("revisions"))
        if value is None:
            value = list(raw.values())
    elif isinstance(raw, list):
        value = raw
    else:
        raise WorkspaceLoadError("templates file must contain a list or mapping")
    if not isinstance(value, list):
        raise WorkspaceLoadError("templates list is invalid")
    return [dict(item) for item in value if isinstance(item, Mapping)]


def load_raw_workspace(root: Path) -> RawWorkspace:
    root = Path(root)
    root_abs = root.resolve()
    templates_path = root / "data" / "templates.yaml"
    config_path = root / "config" / "pipeline.yaml"
    injections_path = root / "data" / "injections.yaml"
    if not injections_path.exists():
        injections_path = root / "injections" / "injections.yaml"
    templates_raw = read_yaml(templates_path, default={"templates": []})
    config_raw = read_yaml(config_path, default={})
    injections_raw = read_yaml(injections_path, default={"injections": []})
    templates = _template_list(templates_raw)
    if isinstance(injections_raw, Mapping):
        injections_value = injections_raw.get("injections", [])
    elif isinstance(injections_raw, list):
        injections_value = injections_raw
    else:
        injections_value = []
    injections = [
        dict(item) for item in injections_value if isinstance(item, Mapping)
    ] if isinstance(injections_value, list) else []
    sample_paths = sorted((root_abs / "data" / "samples").glob("*.yaml"))
    sample_files: list[Path] = []
    batches: list[dict[str, Any]] = []
    for sample_file in sample_paths:
        payload = read_yaml(sample_file, default={})
        if not isinstance(payload, Mapping):
            raise WorkspaceLoadError(f"sample file {sample_file} must contain a mapping")
        batches.append(dict(payload))
        try:
            sample_files.append(sample_file.relative_to(root_abs))
        except ValueError:
            sample_files.append(sample_file)
    if not isinstance(config_raw, Mapping):
        raise WorkspaceLoadError("config file must contain a mapping")
    return RawWorkspace(
        templates=templates,
        sample_batches=batches,
        config=dict(config_raw),
        injections=injections,
        sample_files=sample_files,
    )


@dataclass(frozen=True)
class InjectionRecord:
    injection_id: str
    injection_type: str
    target_kind: str
    target_id: str
    expected_stage: str
    applied: bool
    failure: Failure | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "type": self.injection_type,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "expected_stage": self.expected_stage,
            "applied": self.applied,
            "failure": None if self.failure is None else self.failure.as_dict(),
        }


def _injection_failure(record: InjectionRecord, message: str) -> Failure:
    return Failure(
        code="E_INJECTION_INVALID",
        stage="injection",
        message=message,
        target_kind=record.target_kind,
        target_id=record.target_id,
        injection_id=record.injection_id,
    )


def apply_injections(
    templates: list[dict[str, Any]],
    sample_batches: list[dict[str, Any]],
    injections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[InjectionRecord], list[Failure]]:
    template_data = copy.deepcopy(templates)
    batch_data = copy.deepcopy(sample_batches)
    records: list[InjectionRecord] = []
    failures: list[Failure] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(injections):
        injection_id = item.get("id", item.get("injection_id"))
        injection_type = item.get("type")
        if not isinstance(injection_id, str) or not injection_id:
            injection_id = f"injection-{index + 1:03d}"
        if injection_id in seen_ids:
            record = InjectionRecord(
                injection_id=injection_id,
                injection_type=str(injection_type or ""),
                target_kind="injection",
                target_id=injection_id,
                expected_stage="injection",
                applied=False,
            )
            failure = _injection_failure(record, "duplicate injection id")
            record = InjectionRecord(**{**record.__dict__, "failure": failure})
            records.append(record)
            failures.append(failure)
            continue
        seen_ids.add(injection_id)
        expected_stage = {
            "template_cycle": "load_templates",
            "missing_slot": "load_samples",
            "missing_revision": "load_samples",
            "nonfinite_quality": "score",
        }.get(str(injection_type), "injection")
        target_kind = "template" if injection_type == "template_cycle" else "sample"
        target_id = str(
            item.get("sample_id", item.get("template_id", item.get("target", "")))
        )
        record = InjectionRecord(
            injection_id=injection_id,
            injection_type=str(injection_type or ""),
            target_kind=target_kind,
            target_id=target_id,
            expected_stage=expected_stage,
            applied=False,
        )
        applied = False
        if injection_type == "template_cycle":
            template_id = item.get("template_id")
            revision_number = item.get("revision")
            for revision in template_data:
                if (
                    revision.get("template_id") == template_id
                    and revision.get("revision") == revision_number
                ):
                    revision["parent_revision"] = item.get("parent_revision", revision_number)
                    ids = list(revision.get("_injection_ids", []))
                    ids.append(injection_id)
                    revision["_injection_ids"] = ids
                    applied = True
                    break
        elif injection_type == "missing_slot":
            sample_id = item.get("sample_id")
            slot_name = item.get("slot", item.get("slot_name"))
            for batch in batch_data:
                for sample in batch.get("samples", []) or []:
                    if sample.get("sample_id") != sample_id:
                        continue
                    fills = list(sample.get("slot_fills", []) or [])
                    kept = []
                    removed = False
                    for fill in fills:
                        if not removed and isinstance(fill, Mapping) and fill.get("name") == slot_name:
                            removed = True
                            continue
                        kept.append(fill)
                    if removed:
                        sample["slot_fills"] = kept
                        ids = list(sample.get("_injection_ids", []))
                        ids.append(injection_id)
                        sample["_injection_ids"] = ids
                        applied = True
                    break
                if applied:
                    break
        elif injection_type == "missing_revision":
            sample_id = item.get("sample_id")
            replacement = item.get("revision", 999_999)
            for batch in batch_data:
                for sample in batch.get("samples", []) or []:
                    if sample.get("sample_id") != sample_id:
                        continue
                    sample["template_revision"] = replacement
                    ids = list(sample.get("_injection_ids", []))
                    ids.append(injection_id)
                    sample["_injection_ids"] = ids
                    applied = True
                    break
                if applied:
                    break
        elif injection_type == "nonfinite_quality":
            sample_id = item.get("sample_id")
            for batch in batch_data:
                for sample in batch.get("samples", []) or []:
                    if sample.get("sample_id") != sample_id:
                        continue
                    sample["_quality_override"] = "NaN"
                    ids = list(sample.get("_injection_ids", []))
                    ids.append(injection_id)
                    sample["_injection_ids"] = ids
                    applied = True
                    break
                if applied:
                    break
        record = InjectionRecord(**{**record.__dict__, "applied": applied})
        if not applied:
            failure = _injection_failure(
                record, f"injection target not found or unsupported type: {injection_type}"
            )
            record = InjectionRecord(**{**record.__dict__, "failure": failure})
            failures.append(failure)
        records.append(record)
    return template_data, batch_data, records, failures


def scalar_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numeric slot value")
        return format(value, ".15g")
    if value is None:
        return ""
    raise ValueError("slot value must be a scalar")


def _sample_failure(
    message: str,
    sample_id: str,
    injection_id: str | None,
    *,
    details: dict[str, Any] | None = None,
    stage: str = "load_samples",
    code: str = "E_SAMPLE_INVALID",
) -> Failure:
    return Failure(
        code=code,
        stage=stage,
        message=message,
        target_kind="sample",
        target_id=sample_id,
        injection_id=injection_id,
        details=details or {},
    )


def parse_input_slots(raw_fills: Any) -> list[InputSlot]:
    if not isinstance(raw_fills, list):
        raise ValueError("slot_fills must be a list")
    result: list[InputSlot] = []
    for index, item in enumerate(raw_fills):
        if isinstance(item, Mapping):
            name = item.get("name")
            value = item.get("value")
            priority = item.get("priority", 0)
        elif isinstance(item, list) and len(item) == 2:
            name, value = item
            priority = 0
        else:
            raise ValueError(f"slot fill at index {index} must be a mapping or [name, value]")
        if not isinstance(name, str) or not name:
            raise ValueError(f"slot fill at index {index} has invalid name")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise ValueError(f"slot fill {name} has non-integer priority")
        try:
            text = scalar_to_text(value)
        except ValueError as exc:
            raise ValueError(f"slot fill {name}: {exc}") from exc
        result.append(InputSlot(name=name, value=text, priority=priority, input_index=index))
    return result


@dataclass(frozen=True)
class ValidatedSample:
    sample: SampleInput
    instruction: EffectiveTemplate
    answer: EffectiveTemplate
    resolved_slots: tuple[ResolvedSlot, ...]
    fill_order: tuple[str, ...]
    sample_fingerprint: str

    def slot_values(self) -> dict[str, str]:
        return {slot.name: slot.value for slot in self.resolved_slots}

    def normalized_document(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample.sample_id,
            "batch_id": self.sample.batch_id,
            "batch_file": self.sample.batch_file,
            "source_shard": self.sample.source_shard,
            "template_id": self.sample.template_id,
            "template_revision": self.sample.template_revision,
            "answer_template_id": self.sample.answer_template_id,
            "answer_template_revision": self.sample.answer_template_revision,
            "slot_fills": [fill.as_dict() for fill in self.sample.slot_fills],
            "resolved_slots": [slot.as_dict() for slot in self.resolved_slots],
            "fill_order": list(self.fill_order),
            "sample_fingerprint": self.sample_fingerprint,
            "chain_fingerprint": self.instruction.chain_fingerprint,
            "answer_chain_fingerprint": self.answer.chain_fingerprint,
            "injection_ids": list(self.sample.injection_ids),
            "quality_override": self.sample.quality_override,
        }


def _resolve_duplicate_fills(sample_id: str, fills: list[InputSlot]) -> dict[str, ResolvedSlot]:
    grouped: dict[str, list[InputSlot]] = {}
    for fill in fills:
        grouped.setdefault(fill.name, []).append(fill)
    resolved: dict[str, ResolvedSlot] = {}
    for name in sorted(grouped):
        candidates = grouped[name]
        winner = max(candidates, key=lambda item: (item.priority, item.input_index))
        suppressed = []
        for candidate in sorted(candidates, key=lambda item: item.input_index):
            if candidate is winner:
                continue
            if candidate.priority < winner.priority:
                reason = "lower_priority"
            elif candidate.priority == winner.priority:
                reason = "later_equal_priority_lost"
            else:
                reason = "lower_input_index_on_tie"
            suppressed.append(
                {
                    "value": candidate.value,
                    "priority": candidate.priority,
                    "input_index": candidate.input_index,
                    "reason": reason,
                }
            )
        resolved[name] = ResolvedSlot(
            name=name,
            value=winner.value,
            priority=winner.priority,
            source_index=winner.input_index,
            suppressed=tuple(suppressed),
        )
    return resolved


def validate_samples(
    raw_workspace: RawWorkspace,
    template_catalog: TemplateCatalog,
) -> tuple[list[ValidatedSample], list[SampleInput], list[dict[str, Any]], list[Failure]]:
    valid: list[ValidatedSample] = []
    all_samples: list[SampleInput] = []
    normalized_raw: list[dict[str, Any]] = []
    failures: list[Failure] = []
    seen_ids: set[str] = set()
    seen_occurrences: set[tuple[str, str]] = set()
    for batch_index, batch in enumerate(raw_workspace.sample_batches):
        batch_id = str(batch.get("batch_id", f"batch-{batch_index + 1:03d}"))
        batch_file = (
            str(raw_workspace.sample_files[batch_index])
            if batch_index < len(raw_workspace.sample_files)
            else f"<batch-{batch_index + 1}>"
        )
        raw_samples = batch.get("samples", []) or []
        if not isinstance(raw_samples, list):
            failures.append(
                _sample_failure(
                    "samples must be a list", f"batch:{batch_id}", None, details={"batch": batch_id}
                )
            )
            continue
        for input_index, raw_sample in enumerate(raw_samples):
            normalized_raw.append({"batch_id": batch_id, "batch_file": batch_file, "sample": copy.deepcopy(raw_sample)})
            if not isinstance(raw_sample, Mapping):
                failures.append(
                    _sample_failure(
                        "sample must be a mapping",
                        f"batch:{batch_id}:index:{input_index}",
                        None,
                        details={"batch": batch_id, "input_index": input_index},
                    )
                )
                continue
            injection_ids = tuple(
                str(item) for item in raw_sample.get("_injection_ids", []) if item
            )
            injection_id = injection_ids[0] if injection_ids else None
            sample_id = raw_sample.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                failures.append(
                    _sample_failure(
                        "sample_id must be a non-empty string",
                        f"batch:{batch_id}:index:{input_index}",
                        injection_id,
                        details={"batch": batch_id, "input_index": input_index},
                    )
                )
                continue
            occurrence = (batch_id, sample_id)
            if sample_id in seen_ids or occurrence in seen_occurrences:
                failures.append(
                    _sample_failure(
                        "duplicate sample_id",
                        sample_id,
                        injection_id,
                        details={"batch": batch_id, "input_index": input_index},
                    )
                )
                continue
            seen_ids.add(sample_id)
            seen_occurrences.add(occurrence)
            template_id = raw_sample.get("template_id")
            answer_template_id = raw_sample.get(
                "answer_template_id", raw_sample.get("answer_id")
            )
            template_revision = raw_sample.get("template_revision")
            answer_revision = raw_sample.get(
                "answer_template_revision", raw_sample.get("answer_revision", template_revision)
            )
            source_shard = raw_sample.get("source_shard")
            if not isinstance(template_id, str) or not template_id:
                failures.append(
                    _sample_failure("template_id must be a non-empty string", sample_id, injection_id)
                )
                continue
            if not isinstance(answer_template_id, str) or not answer_template_id:
                failures.append(
                    _sample_failure(
                        "answer_template_id must be a non-empty string", sample_id, injection_id
                    )
                )
                continue
            if not isinstance(template_revision, int) or isinstance(template_revision, bool):
                failures.append(
                    _sample_failure(
                        "template_revision must be an integer", sample_id, injection_id
                    )
                )
                continue
            if not isinstance(answer_revision, int) or isinstance(answer_revision, bool):
                failures.append(
                    _sample_failure(
                        "answer_template_revision must be an integer", sample_id, injection_id
                    )
                )
                continue
            if not isinstance(source_shard, str) or not source_shard:
                failures.append(
                    _sample_failure("source_shard must be a non-empty string", sample_id, injection_id)
                )
                continue
            try:
                fills = parse_input_slots(raw_sample.get("slot_fills", []))
            except ValueError as exc:
                failures.append(
                    _sample_failure(str(exc), sample_id, injection_id)
                )
                continue
            quality_override = raw_sample.get("_quality_override")
            sample = SampleInput(
                sample_id=sample_id,
                template_id=template_id,
                template_revision=template_revision,
                answer_template_id=answer_template_id,
                answer_template_revision=answer_revision,
                source_shard=source_shard,
                slot_fills=fills,
                batch_id=batch_id,
                batch_file=batch_file,
                input_index=input_index,
                injection_ids=injection_ids,
                quality_override=quality_override,
                raw=dict(raw_sample),
            )
            all_samples.append(sample)
            instruction = template_catalog.get(template_id, template_revision)
            answer = template_catalog.get(answer_template_id, answer_revision)
            if instruction is None:
                failures.append(
                    _sample_failure(
                        f"instruction template revision does not exist or is invalid: {template_id}@{template_revision}",
                        sample_id,
                        injection_id,
                    )
                )
                continue
            if answer is None:
                failures.append(
                    _sample_failure(
                        f"answer template revision does not exist or is invalid: {answer_template_id}@{answer_revision}",
                        sample_id,
                        injection_id,
                    )
                )
                continue
            resolved = _resolve_duplicate_fills(sample_id, fills)
            effective_names = set(instruction.slot_names)
            provided_names = set(resolved)
            missing = sorted(effective_names - provided_names)
            extra = sorted(provided_names - effective_names)
            if missing or extra:
                failures.append(
                    _sample_failure(
                        "slot fills do not match the effective slot set",
                        sample_id,
                        injection_id,
                        details={"missing_slots": missing, "unexpected_slots": extra},
                    )
                )
                continue
            answer_missing = sorted(set(answer.slot_names) - provided_names)
            if answer_missing:
                failures.append(
                    _sample_failure(
                        "slot fills do not cover the answer template slot set",
                        sample_id,
                        injection_id,
                        details={"missing_answer_slots": answer_missing},
                    )
                )
                continue
            constraint_failure = None
            for name in sorted(resolved):
                value = resolved[name].value
                applicable_rules = [instruction.slot_map[name].rule]
                if name in answer.slot_map:
                    applicable_rules.append(answer.slot_map[name].rule)
                for rule in applicable_rules:
                    if rule.required and value == "":
                        constraint_failure = f"required slot is empty: {name}"
                        break
                    if rule.allowed_values and value not in rule.allowed_values:
                        constraint_failure = (
                            f"slot {name} value is not in allowed_values: {value!r}"
                        )
                        break
                    if rule.min_length is not None and len(value) < rule.min_length:
                        constraint_failure = f"slot {name} is shorter than min_length"
                        break
                    if rule.max_length is not None and len(value) > rule.max_length:
                        constraint_failure = f"slot {name} is longer than max_length"
                        break
                if constraint_failure is not None:
                    break
            if constraint_failure is not None:
                failures.append(
                    _sample_failure(constraint_failure, sample_id, injection_id)
                )
                continue
            try:
                # Rendering here is only a validation probe; the engine still uses the render cache.
                render_template(instruction.instruction_template, {k: v.value for k, v in resolved.items()})
                render_template(answer.answer_template, {k: v.value for k, v in resolved.items()})
            except Exception as exc:
                failures.append(
                    _sample_failure(f"render validation failed: {exc}", sample_id, injection_id)
                )
                continue
            fingerprint = digest(
                {"kind": "sample-content", "sample": sample.content_document()}
            )
            valid.append(
                ValidatedSample(
                    sample=sample,
                    instruction=instruction,
                    answer=answer,
                    resolved_slots=tuple(resolved[name] for name in sorted(resolved)),
                    fill_order=tuple(sorted(resolved)),
                    sample_fingerprint=fingerprint,
                )
            )
    valid.sort(key=lambda item: (item.sample.source_shard, item.sample.sample_id))
    all_samples.sort(key=lambda item: (item.source_shard, item.sample_id))
    failures.sort(key=Failure.sort_key)
    return valid, all_samples, normalized_raw, failures


@dataclass
class LoadedWorkspace:
    raw: RawWorkspace
    templates: TemplateCatalog
    valid_samples: list[ValidatedSample]
    all_samples: list[SampleInput]
    normalized_samples: list[dict[str, Any]]
    config: PipelineConfig
    failures: list[Failure]
    injection_records: list[InjectionRecord]
    template_revisions_data: list[dict[str, Any]]
    sample_batches_data: list[dict[str, Any]]
    config_data: dict[str, Any]
    global_version: str
    config_hash: str
    template_chain_fingerprint: str
    sample_fingerprints: dict[str, str]
    revision_fingerprints: dict[str, str]
    validation_products: ValidationProducts = field(init=False)

    def __post_init__(self) -> None:
        self.failures = sorted(self.failures, key=Failure.sort_key)
        self.revision_fingerprints = dict(self.templates.revision_fingerprints)
        self.sample_fingerprints = {
            sample.sample.sample_id: sample.sample_fingerprint
            for sample in self.valid_samples
        }
        samples_fp = digest(
            {
                "kind": "sample-set",
                "samples": [
                    {"sample_id": item.sample.sample_id, "fingerprint": item.sample_fingerprint}
                    for item in sorted(self.valid_samples, key=lambda item: item.sample.sample_id)
                ],
                "raw": self.normalized_samples,
            }
        )
        version = digest(
            {
                "kind": "workspace-version",
                "schema_version": 1,
                "templates": self.template_revisions_data,
                "samples": self.normalized_samples,
                "config": self.config.normalized,
                "injections": [record.as_dict() for record in self.injection_records],
            }
        )
        self.global_version = version
        self.config_hash = self.config.config_hash
        self.template_chain_fingerprint = self.templates.chain_fingerprint
        self.validation_products = ValidationProducts(
            normalized_templates=tuple(self.template_revisions_data),
            normalized_samples=tuple(self.normalized_samples),
            templates_fingerprint=digest(
                {"kind": "normalized-templates", "templates": self.template_revisions_data}
            ),
            samples_fingerprint=samples_fp,
            config_hash=self.config_hash,
            global_version=self.global_version,
            template_revision_fingerprints=self.revision_fingerprints,
            sample_fingerprints=self.sample_fingerprints,
        )


def prepare_workspace(raw: RawWorkspace) -> LoadedWorkspace:
    (
        template_revisions,
        sample_batches,
        injection_records,
        injection_failures,
    ) = apply_injections(raw.templates, raw.sample_batches, raw.injections)
    config, config_failures = parse_config(raw.config)
    catalog = parse_templates({"templates": template_revisions})
    sample_raw = RawWorkspace(
        templates=template_revisions,
        sample_batches=sample_batches,
        config=raw.config,
        injections=raw.injections,
        sample_files=raw.sample_files,
    )
    valid_samples, all_samples, normalized_samples, sample_failures = validate_samples(
        sample_raw, catalog
    )
    failures = [
        *injection_failures,
        *catalog.failures,
        *config_failures,
        *sample_failures,
    ]
    return LoadedWorkspace(
        raw=sample_raw,
        templates=catalog,
        valid_samples=valid_samples,
        all_samples=all_samples,
        normalized_samples=normalized_samples,
        config=config,
        failures=failures,
        injection_records=injection_records,
        template_revisions_data=template_revisions,
        sample_batches_data=sample_batches,
        config_data=config.normalized,
        global_version="",
        config_hash=config.config_hash,
        template_chain_fingerprint=catalog.chain_fingerprint,
        sample_fingerprints={},
        revision_fingerprints={},
    )


def load_workspace(root: Path) -> LoadedWorkspace:
    return prepare_workspace(load_raw_workspace(root))
