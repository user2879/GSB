from __future__ import annotations

import copy
import string
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .canonical import digest
from .errors import Failure
from .models import SlotRule


TEMPLATE_SCHEMA_VERSION = 1
SLOT_RULE_KEYS = {"required", "allowed_values", "min_length", "max_length", "default"}


class TemplateParseError(ValueError):
    pass


@dataclass(frozen=True)
class TemplateRevision:
    template_id: str
    revision: int
    parent_revision: int | None
    instruction_template: str | None
    answer_template: str | None
    slot_declarations: tuple[dict[str, Any], ...]
    removed_slots: tuple[str, ...]
    constraints: tuple[tuple[str, dict[str, Any]], ...]
    injection_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "revision": self.revision,
            "parent_revision": self.parent_revision,
            "instruction_template": self.instruction_template,
            "answer_template": self.answer_template,
            "slots": [copy.deepcopy(item) for item in self.slot_declarations],
            "remove_slots": list(self.removed_slots),
            "constraints": {key: copy.deepcopy(value) for key, value in self.constraints},
            "injection_ids": list(self.injection_ids),
        }

    @property
    def key(self) -> str:
        return f"{self.template_id}@{self.revision}"

    def fingerprint(self) -> str:
        return digest({"kind": "template-revision", "revision": self.as_dict()})


@dataclass(frozen=True)
class EffectiveSlot:
    rule: SlotRule
    origin: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        result = self.rule.as_dict()
        result["origin"] = copy.deepcopy(self.origin)
        return result


@dataclass(frozen=True)
class EffectiveTemplate:
    template_id: str
    revision: int
    instruction_template: str
    answer_template: str
    slots: tuple[EffectiveSlot, ...]
    instruction_placeholders: tuple[str, ...]
    answer_placeholders: tuple[str, ...]
    revision_path: tuple[int, ...]
    revision_fingerprints: tuple[tuple[int, str], ...]
    chain_fingerprint: str
    effective_fingerprint: str

    @property
    def slot_names(self) -> tuple[str, ...]:
        return tuple(slot.rule.name for slot in self.slots)

    @property
    def slot_map(self) -> dict[str, EffectiveSlot]:
        return {slot.rule.name: slot for slot in self.slots}

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "revision": self.revision,
            "instruction_template": self.instruction_template,
            "answer_template": self.answer_template,
            "slots": [slot.as_dict() for slot in self.slots],
            "instruction_placeholders": list(self.instruction_placeholders),
            "answer_placeholders": list(self.answer_placeholders),
            "revision_path": list(self.revision_path),
            "revision_fingerprints": [
                {"revision": rev, "fingerprint": fp}
                for rev, fp in self.revision_fingerprints
            ],
            "chain_fingerprint": self.chain_fingerprint,
            "effective_fingerprint": self.effective_fingerprint,
        }


@dataclass
class TemplateCatalog:
    revisions: dict[tuple[str, int], TemplateRevision]
    valid_revisions: dict[tuple[str, int], EffectiveTemplate]
    failures: list[Failure]
    normalized_revisions: list[dict[str, Any]]
    revision_fingerprints: dict[str, str]
    template_ids: tuple[str, ...]
    chain_fingerprint: str

    def get(self, template_id: str, revision: int) -> EffectiveTemplate | None:
        return self.valid_revisions.get((template_id, revision))

    def template_revision_fingerprint(self, template_id: str, revision: int) -> str | None:
        effective = self.get(template_id, revision)
        return None if effective is None else effective.chain_fingerprint

    def effective_templates(self) -> list[EffectiveTemplate]:
        return [
            self.valid_revisions[key]
            for key in sorted(self.valid_revisions, key=lambda item: (item[0], item[1]))
        ]


def _failure(
    message: str,
    *,
    target_id: str,
    revision: int | None,
    injection_id: str | None,
    details: dict[str, Any] | None = None,
) -> Failure:
    return Failure(
        code="E_TEMPLATE_INVALID",
        stage="load_templates",
        message=message,
        target_kind="template",
        target_id=target_id,
        revision=revision,
        injection_id=injection_id,
        details=details or {},
    )


def extract_placeholders(text: str, *, where: str) -> tuple[str, ...]:
    formatter = string.Formatter()
    try:
        parsed = list(formatter.parse(text))
    except ValueError as exc:
        raise TemplateParseError(f"{where}: malformed template braces: {exc}") from exc
    names: list[str] = []
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name.isidentifier():
            raise TemplateParseError(f"{where}: unsupported placeholder {{{field_name}}}")
        if conversion is not None or format_spec:
            raise TemplateParseError(
                f"{where}: conversions and format specs are not deterministic: {{{field_name}}}"
            )
        names.append(field_name)
    return tuple(names)


def render_template(text: str, values: Mapping[str, str]) -> str:
    """Render a pre-validated template using only simple named placeholders."""

    parts: list[str] = []
    formatter = string.Formatter()
    for literal, field_name, format_spec, conversion in formatter.parse(text):
        parts.append(literal)
        if field_name is None:
            continue
        if conversion is not None or format_spec or not field_name.isidentifier():
            raise TemplateParseError(f"unsupported placeholder: {field_name}")
        if field_name not in values:
            raise TemplateParseError(f"missing placeholder value: {field_name}")
        parts.append(values[field_name])
    return "".join(parts)


def _normalize_slot_declarations(raw_slots: Any) -> tuple[dict[str, Any], ...]:
    if raw_slots is None:
        return ()
    declarations: list[dict[str, Any]] = []
    if isinstance(raw_slots, Mapping):
        iterable: Iterable[Any] = [
            {"name": name, **(value if isinstance(value, Mapping) else {})}
            for name, value in raw_slots.items()
        ]
    elif isinstance(raw_slots, list):
        iterable = raw_slots
    else:
        raise TemplateParseError("slots must be a list or mapping")
    seen: set[str] = set()
    for item in iterable:
        if isinstance(item, str):
            entry = {"name": item}
        elif isinstance(item, Mapping):
            entry = copy.deepcopy(dict(item))
        else:
            raise TemplateParseError("each slot declaration must be a name or mapping")
        name = entry.get("name")
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise TemplateParseError(f"invalid slot name: {name!r}")
        if name in seen:
            raise TemplateParseError(f"duplicate slot declaration: {name}")
        seen.add(name)
        declarations.append(entry)
    return tuple(declarations)


def _normalize_constraints(raw: Any) -> tuple[tuple[str, dict[str, Any]], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise TemplateParseError("constraints must be a mapping keyed by slot name")
    result: list[tuple[str, dict[str, Any]]] = []
    for name in sorted(raw, key=str):
        if not isinstance(name, str) or not name:
            raise TemplateParseError("constraint slot name must be a non-empty string")
        value = raw[name]
        if not isinstance(value, Mapping):
            raise TemplateParseError(f"constraint {name} must be a mapping")
        normalized = copy.deepcopy(dict(value))
        unknown = sorted(set(normalized) - SLOT_RULE_KEYS)
        if unknown:
            raise TemplateParseError(f"constraint {name} has unknown fields: {unknown}")
        result.append((name, normalized))
    return tuple(result)


def parse_template_revision(raw: Any, index: int) -> tuple[TemplateRevision | None, Failure | None]:
    if not isinstance(raw, Mapping):
        return None, _failure(
            f"template revision at index {index} must be a mapping",
            target_id=f"template-index-{index}",
            revision=None,
            injection_id=None,
        )
    injection_ids = tuple(str(item) for item in raw.get("_injection_ids", []) if item)
    injection_id = injection_ids[0] if injection_ids else None
    template_id = raw.get("template_id")
    revision = raw.get("revision")
    if not isinstance(template_id, str) or not template_id.strip():
        return None, _failure(
            f"template revision at index {index} has no valid template_id",
            target_id=f"template-index-{index}",
            revision=None,
            injection_id=injection_id,
        )
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        return None, _failure(
            "revision must be a positive integer",
            target_id=template_id,
            revision=None,
            injection_id=injection_id,
        )
    parent = raw.get("parent_revision")
    if parent is not None and (
        not isinstance(parent, int) or isinstance(parent, bool) or parent < 1
    ):
        return None, _failure(
            "parent_revision must be null or a positive integer",
            target_id=template_id,
            revision=revision,
            injection_id=injection_id,
        )
    instruction = raw.get("instruction_template", raw.get("instruction"))
    answer = raw.get("answer_template", raw.get("answer"))
    if instruction is not None and not isinstance(instruction, str):
        return None, _failure(
            "instruction_template must be a string or null",
            target_id=template_id,
            revision=revision,
            injection_id=injection_id,
        )
    if answer is not None and not isinstance(answer, str):
        return None, _failure(
            "answer_template must be a string or null",
            target_id=template_id,
            revision=revision,
            injection_id=injection_id,
        )
    try:
        slots = _normalize_slot_declarations(raw.get("slots"))
        removed = raw.get("remove_slots", [])
        if removed is None:
            removed = []
        if not isinstance(removed, list) or any(
            not isinstance(item, str) or not item for item in removed
        ):
            raise TemplateParseError("remove_slots must be a list of slot names")
        constraints = _normalize_constraints(raw.get("constraints"))
    except TemplateParseError as exc:
        return None, _failure(
            str(exc),
            target_id=template_id,
            revision=revision,
            injection_id=injection_id,
        )
    return (
        TemplateRevision(
            template_id=template_id,
            revision=revision,
            parent_revision=parent,
            instruction_template=instruction,
            answer_template=answer,
            slot_declarations=slots,
            removed_slots=tuple(removed),
            constraints=constraints,
            injection_ids=injection_ids,
        ),
        None,
    )


def _revision_source(revision: TemplateRevision) -> str:
    return revision.injection_ids[0] if revision.injection_ids else ""


def _resolve_slot_rules(
    path: list[TemplateRevision],
) -> tuple[dict[str, SlotRule], dict[str, dict[str, Any]]]:
    rules: dict[str, SlotRule] = {}
    origins: dict[str, dict[str, Any]] = {}
    for revision in path:
        for removed in revision.removed_slots:
            rules.pop(removed, None)
            origin = origins.pop(removed, None)
            if origin is not None:
                origin.setdefault("removals", []).append(revision.revision)
        for declaration in revision.slot_declarations:
            name = str(declaration["name"])
            prior = rules.get(name)
            merged: dict[str, Any] = {}
            inherited_fields: list[str] = []
            if prior is not None:
                merged = {
                    "name": prior.name,
                    "required": prior.required,
                    "allowed_values": list(prior.allowed_values),
                    "min_length": prior.min_length,
                    "max_length": prior.max_length,
                    "default": prior.default,
                }
            else:
                merged = {
                    "name": name,
                    "required": True,
                    "allowed_values": [],
                    "min_length": None,
                    "max_length": None,
                    "default": None,
                }
            for key, value in declaration.items():
                if key == "name":
                    continue
                if key not in SLOT_RULE_KEYS:
                    raise TemplateParseError(f"slot {name} has unknown field: {key}")
                merged[key] = copy.deepcopy(value)
                if prior is not None:
                    inherited_fields.append(key)
            origin = origins.setdefault(name, {"overrides": []})
            origin["source_revision"] = revision.revision
            origin["overrides"].append(
                {
                    "revision": revision.revision,
                    "action": "override" if prior is not None else "append",
                    "fields": sorted(inherited_fields),
                    "injection_id": _revision_source(revision) or None,
                }
            )
            rules[name] = SlotRule(
                name=name,
                required=merged["required"],
                allowed_values=tuple(str(item) for item in merged["allowed_values"]),
                min_length=merged["min_length"],
                max_length=merged["max_length"],
                default=merged["default"],
                source_revision=revision.revision,
            )
        for name, constraint in revision.constraints:
            if name not in rules:
                # Constraints may target a slot inherited later in malformed data.
                # It remains invalid when the chain is validated.
                raise TemplateParseError(
                    f"constraint targets slot not declared through revision {revision.revision}: {name}"
                )
            prior = rules[name]
            merged = {
                "required": prior.required,
                "allowed_values": list(prior.allowed_values),
                "min_length": prior.min_length,
                "max_length": prior.max_length,
                "default": prior.default,
            }
            for key, value in constraint.items():
                merged[key] = copy.deepcopy(value)
            origin = origins.setdefault(name, {"overrides": []})
            origin["source_revision"] = revision.revision
            origin["overrides"].append(
                {
                    "revision": revision.revision,
                    "action": "constraint",
                    "fields": sorted(constraint),
                    "injection_id": _revision_source(revision) or None,
                }
            )
            rules[name] = SlotRule(
                name=name,
                required=merged["required"],
                allowed_values=tuple(str(item) for item in merged["allowed_values"]),
                min_length=merged["min_length"],
                max_length=merged["max_length"],
                default=merged["default"],
                source_revision=revision.revision,
            )
    return rules, origins


def _validate_slot_rule(name: str, rule: SlotRule) -> str | None:
    if not isinstance(rule.required, bool):
        return f"slot {name}: required must be boolean"
    if rule.allowed_values and any(not isinstance(item, str) for item in rule.allowed_values):
        return f"slot {name}: allowed_values must contain strings"
    for field_name in ("min_length", "max_length"):
        value = getattr(rule, field_name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            return f"slot {name}: {field_name} must be a non-negative integer or null"
    if (
        rule.min_length is not None
        and rule.max_length is not None
        and rule.min_length > rule.max_length
    ):
        return f"slot {name}: min_length exceeds max_length"
    if rule.default is not None and not isinstance(rule.default, str):
        return f"slot {name}: default must be a string or null"
    return None


def _cycle_members(edges: dict[int, int | None]) -> set[int]:
    members: set[int] = set()
    for start in edges:
        seen: dict[int, int] = {}
        current: int | None = start
        step = 0
        while current is not None and current in edges:
            if current in seen:
                cycle_start = seen[current]
                for node, position in seen.items():
                    if position >= cycle_start:
                        members.add(node)
                break
            seen[current] = step
            step += 1
            current = edges.get(current)
    return members


def parse_templates(raw_templates: Any) -> TemplateCatalog:
    failures: list[Failure] = []
    if raw_templates is None:
        raw_list: list[Any] = []
    elif isinstance(raw_templates, Mapping):
        value = raw_templates.get("templates", raw_templates.get("revisions"))
        if value is None and all(
            isinstance(item, Mapping) for item in raw_templates.values()
        ):
            value = list(raw_templates.values())
        raw_list = list(value or [])
    elif isinstance(raw_templates, list):
        raw_list = raw_templates
    else:
        failure = _failure(
            "templates must be a list or mapping",
            target_id="templates",
            revision=None,
            injection_id=None,
        )
        return TemplateCatalog(
            revisions={},
            valid_revisions={},
            failures=[failure],
            normalized_revisions=[],
            revision_fingerprints={},
            template_ids=(),
            chain_fingerprint=digest({"kind": "template-chain", "revisions": []}),
        )

    revisions: dict[tuple[str, int], TemplateRevision] = {}
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_list):
        revision, failure = parse_template_revision(item, index)
        if failure is not None:
            failures.append(failure)
            continue
        assert revision is not None
        key = (revision.template_id, revision.revision)
        if key in revisions:
            failures.append(
                _failure(
                    "duplicate template_id/revision pair",
                    target_id=revision.template_id,
                    revision=revision.revision,
                    injection_id=revision.injection_ids[0] if revision.injection_ids else None,
                )
            )
            continue
        revisions[key] = revision
        normalized.append(revision.as_dict())

    normalized.sort(key=lambda item: (item["template_id"], item["revision"]))
    by_template: dict[str, list[TemplateRevision]] = {}
    for revision in revisions.values():
        by_template.setdefault(revision.template_id, []).append(revision)

    invalid_keys: set[tuple[str, int]] = set()
    for template_id in sorted(by_template):
        template_revisions = sorted(by_template[template_id], key=lambda item: item.revision)
        keys = {item.revision for item in template_revisions}
        edges: dict[int, int | None] = {}
        roots: list[int] = []
        for revision in template_revisions:
            key = (template_id, revision.revision)
            if revision.parent_revision is None:
                roots.append(revision.revision)
            elif revision.parent_revision not in keys:
                invalid_keys.add(key)
                failures.append(
                    _failure(
                        f"parent revision {revision.parent_revision} does not exist",
                        target_id=template_id,
                        revision=revision.revision,
                        injection_id=revision.injection_ids[0] if revision.injection_ids else None,
                    )
                )
            edges[revision.revision] = revision.parent_revision
        if not template_revisions:
            continue
        if len(roots) != 1:
            failures.append(
                _failure(
                    f"template must have exactly one root revision, found {len(roots)}",
                    target_id=template_id,
                    revision=None,
                    injection_id=None,
                    details={"roots": sorted(roots)},
                )
            )
            for key in [(template_id, revision.revision) for revision in template_revisions]:
                invalid_keys.add(key)
        cycle = _cycle_members(edges)
        if cycle:
            cycle_sources = [
                revisions[(template_id, rev)].injection_ids[0]
                for rev in sorted(cycle)
                if revisions[(template_id, rev)].injection_ids
            ]
            failures.append(
                _failure(
                    "template revision parent chain contains a cycle",
                    target_id=template_id,
                    revision=min(cycle),
                    injection_id=cycle_sources[0] if cycle_sources else None,
                    details={"cycle_revisions": sorted(cycle)},
                )
            )
            for rev in sorted(cycle):
                invalid_keys.add((template_id, rev))

    valid_revisions: dict[tuple[str, int], EffectiveTemplate] = {}
    revision_fingerprints: dict[str, str] = {}
    for template_id in sorted(by_template):
        for revision in sorted(by_template[template_id], key=lambda item: item.revision):
            key = (template_id, revision.revision)
            path_revisions: list[TemplateRevision] = []
            current: TemplateRevision | None = revision
            visited: set[int] = set()
            path_invalid = False
            while current is not None:
                if current.revision in visited or (template_id, current.revision) in invalid_keys:
                    path_invalid = True
                    break
                visited.add(current.revision)
                path_revisions.append(current)
                if current.parent_revision is None:
                    break
                current = revisions.get((template_id, current.parent_revision))
                if current is None:
                    path_invalid = True
                    break
            if path_invalid:
                invalid_keys.add(key)
                continue
            path_revisions.reverse()
            try:
                rules, origins = _resolve_slot_rules(path_revisions)
            except TemplateParseError as exc:
                invalid_keys.add(key)
                failures.append(
                    _failure(
                        str(exc),
                        target_id=template_id,
                        revision=revision.revision,
                        injection_id=revision.injection_ids[0] if revision.injection_ids else None,
                    )
                )
                continue
            for name in sorted(rules):
                problem = _validate_slot_rule(name, rules[name])
                if problem:
                    invalid_keys.add(key)
                    failures.append(
                        _failure(
                            problem,
                            target_id=template_id,
                            revision=revision.revision,
                            injection_id=revision.injection_ids[0] if revision.injection_ids else None,
                        )
                    )
            if key in invalid_keys:
                continue
            instruction = None
            answer = None
            for path_revision in path_revisions:
                if path_revision.instruction_template is not None:
                    instruction = path_revision.instruction_template
                if path_revision.answer_template is not None:
                    answer = path_revision.answer_template
            if instruction is None or answer is None:
                invalid_keys.add(key)
                failures.append(
                    _failure(
                        "effective instruction_template and answer_template must both be defined",
                        target_id=template_id,
                        revision=revision.revision,
                        injection_id=revision.injection_ids[0] if revision.injection_ids else None,
                    )
                )
                continue
            try:
                instruction_placeholders = extract_placeholders(
                    instruction, where=f"{template_id}@{revision.revision}.instruction"
                )
                answer_placeholders = extract_placeholders(
                    answer, where=f"{template_id}@{revision.revision}.answer"
                )
            except TemplateParseError as exc:
                invalid_keys.add(key)
                failures.append(
                    _failure(
                        str(exc),
                        target_id=template_id,
                        revision=revision.revision,
                        injection_id=revision.injection_ids[0] if revision.injection_ids else None,
                    )
                )
                continue
            unknown_placeholders = sorted(
                (set(instruction_placeholders) | set(answer_placeholders)) - set(rules)
            )
            if unknown_placeholders:
                invalid_keys.add(key)
                failures.append(
                    _failure(
                        f"template placeholders are not declared as slots: {unknown_placeholders}",
                        target_id=template_id,
                        revision=revision.revision,
                        injection_id=revision.injection_ids[0] if revision.injection_ids else None,
                    )
                )
                continue
            revision_fingerprints[revision.key] = revision.fingerprint()
            path_fingerprints = tuple(
                (item.revision, revision_fingerprints.get(item.key, item.fingerprint()))
                for item in path_revisions
            )
            chain_fp = digest(
                {
                    "kind": "template-effective-chain",
                    "schema_version": TEMPLATE_SCHEMA_VERSION,
                    "template_id": template_id,
                    "revision": revision.revision,
                    "path": [
                        {"revision": rev, "fingerprint": fp}
                        for rev, fp in path_fingerprints
                    ],
                }
            )
            effective_slots = tuple(
                EffectiveSlot(rule=rules[name], origin=origins.get(name, {}))
                for name in sorted(rules)
            )
            effective_fp = digest(
                {
                    "kind": "template-effective",
                    "chain_fingerprint": chain_fp,
                    "instruction": instruction,
                    "answer": answer,
                    "slots": [slot.as_dict() for slot in effective_slots],
                }
            )
            valid_revisions[key] = EffectiveTemplate(
                template_id=template_id,
                revision=revision.revision,
                instruction_template=instruction,
                answer_template=answer,
                slots=effective_slots,
                instruction_placeholders=complete_sorted_unique(instruction_placeholders),
                answer_placeholders=complete_sorted_unique(answer_placeholders),
                revision_path=tuple(item.revision for item in path_revisions),
                revision_fingerprints=path_fingerprints,
                chain_fingerprint=chain_fp,
                effective_fingerprint=effective_fp,
            )

    # Removing one revision can also invalidate descendants. The cache lookup above
    # already refuses any path containing an invalid ancestor; record one concise
    # failure for each uncached descendant chain.
    for template_id in sorted(by_template):
        for revision in sorted(by_template[template_id], key=lambda item: item.revision):
            key = (template_id, revision.revision)
            if key in valid_revisions or key in {
                (failure.target_id, failure.revision)
                for failure in failures
                if failure.revision is not None
            }:
                continue
            ancestor = revision.parent_revision
            while ancestor is not None:
                if (template_id, ancestor) in invalid_keys:
                    failures.append(
                        _failure(
                            f"ancestor revision {ancestor} is invalid",
                            target_id=template_id,
                            revision=revision.revision,
                            injection_id=revision.injection_ids[0] if revision.injection_ids else None,
                        )
                    )
                    break
                parent = revisions.get((template_id, ancestor))
                ancestor = None if parent is None else parent.parent_revision

    deduped_failures: dict[tuple[Any, ...], Failure] = {}
    for failure in failures:
        deduped_failures[failure.sort_key()] = failure
    ordered_failures = sorted(deduped_failures.values(), key=Failure.sort_key)
    chain_fingerprint = digest(
        {
            "kind": "all-template-chains",
            "schema_version": TEMPLATE_SCHEMA_VERSION,
            "effective": [
                {"key": f"{key[0]}@{key[1]}", "fingerprint": value.chain_fingerprint}
                for key, value in sorted(valid_revisions.items())
            ],
            "invalid": sorted([f"{key[0]}@{key[1]}" for key in invalid_keys]),
        }
    )
    normalized_fp_map = {
        f"{revision.template_id}@{revision.revision}": revision.fingerprint()
        for revision in sorted(
            revisions.values(), key=lambda item: (item.template_id, item.revision)
        )
    }
    return TemplateCatalog(
        revisions=revisions,
        valid_revisions=valid_revisions,
        failures=ordered_failures,
        normalized_revisions=normalized,
        revision_fingerprints=normalized_fp_map,
        template_ids=tuple(sorted(by_template)),
        chain_fingerprint=chain_fingerprint,
    )


def complete_sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    """Return sorted unique values; placeholder multiplicity is retained in the text itself."""

    return tuple(sorted(set(values)))

