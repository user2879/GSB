"""Deterministic edit-command stream loading and application."""

import json

from .config import Config, RewriteRule, RewriteTable
from .errors import E_INPUT_INVALID, PipelineError
from .fixedpoint import make_qformat, validate_fixedpoint
from .hashing import fingerprint

DOCUMENT_OPS = {"replace_document", "insert_paragraph", "update_paragraph", "delete_paragraph"}
CONFIG_OPS = {
    "set_max_sentences",
    "set_coverage_floor",
    "set_weights",
    "set_position_decay",
    "set_style_template",
    "set_quantization",
}
TABLE_OPS = {"add_rule", "update_rule", "delete_rule", "bump_table_version"}


def load_edit_stream(path):
    commands = []
    failures = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    {"code": E_INPUT_INVALID, "object_id": "edit:%d" % line_no,
                     "message": "invalid JSON: %s" % exc.msg}
                )
                continue
            if not isinstance(command, dict) or "op" not in command:
                failures.append(
                    {"code": E_INPUT_INVALID, "object_id": "edit:%d" % line_no,
                     "message": "command must be an object with op"}
                )
                continue
            if command["op"] not in DOCUMENT_OPS | CONFIG_OPS | TABLE_OPS:
                failures.append(
                    {"code": E_INPUT_INVALID, "object_id": command.get("command_id", "edit:%d" % line_no),
                     "message": "unknown op %r" % command["op"]}
                )
                continue
            commands.append(command)
    return commands, failures


def apply_document_edits(documents, commands, failures):
    indexed = {doc["doc_id"]: doc for doc in documents}
    for command in commands:
        op = command["op"]
        if op not in DOCUMENT_OPS:
            continue
        doc_id = command.get("doc_id")
        if op == "replace_document":
            payload = command["document"]
            try:
                from .documents import _validate_document
                document = _validate_document(payload, 0)
            except PipelineError as exc:
                failures.append(exc.to_record())
                continue
            indexed[doc_id] = document
            continue
        document = indexed.get(doc_id)
        if document is None:
            failures.append({"code": E_INPUT_INVALID, "object_id": doc_id,
                             "message": "document not found for %s" % op})
            continue
        if op == "insert_paragraph":
            paragraph = {"pid": str(command["pid"]), "text": command["text"]}
            position = command.get("position", len(document["paragraphs"]))
            document["paragraphs"].insert(position, paragraph)
        elif op == "update_paragraph":
            found = False
            for paragraph in document["paragraphs"]:
                if paragraph["pid"] == str(command["pid"]):
                    paragraph["text"] = command["text"]
                    found = True
                    break
            if not found:
                failures.append({"code": E_INPUT_INVALID, "object_id": doc_id,
                                 "message": "paragraph %s not found" % command["pid"]})
                continue
        elif op == "delete_paragraph":
            before = len(document["paragraphs"])
            document["paragraphs"] = [p for p in document["paragraphs"] if p["pid"] != str(command["pid"])]
            if len(document["paragraphs"]) == before:
                failures.append({"code": E_INPUT_INVALID, "object_id": doc_id,
                                 "message": "paragraph %s not found" % command["pid"]})
                continue
            if not document["paragraphs"]:
                failures.append({"code": E_INPUT_INVALID, "object_id": doc_id,
                                 "message": "delete would empty the document"})
                continue
        document["content_fingerprint"] = fingerprint(
            {
                "doc_id": document["doc_id"],
                "title": document["title"],
                "paragraphs": document["paragraphs"],
                "entities": document["entities"],
                "events": document["events"],
            }
        )
    return sorted(indexed.values(), key=lambda item: item["doc_id"])


def _rebuild_config(config, **changes):
    values = {
        "max_sentences": config.max_sentences,
        "term_weight": config.term_weight,
        "entity_weight": config.entity_weight,
        "coverage_floor": config.coverage_floor,
        "position_decay": config.position_decay,
        "style_template_id": config.style_template_id,
        "synonym_version": config.synonym_version,
        "fixedpoint_width": config.fixedpoint_width,
        "quantization_tier": config.quantization_tier,
        "workers": config.workers,
        "snapshot_interval": config.snapshot_interval,
        "sentence_budget": config.sentence_budget,
    }
    values.update(changes)
    problem = validate_fixedpoint(values["fixedpoint_width"], values["quantization_tier"])
    if problem is not None:
        raise PipelineError(E_INPUT_INVALID, problem)
    rebuilt = Config(
        max_sentences=values["max_sentences"],
        term_weight=float(values["term_weight"]),
        entity_weight=float(values["entity_weight"]),
        coverage_floor=float(values["coverage_floor"]),
        position_decay=float(values["position_decay"]),
        style_template_id=str(values["style_template_id"]),
        synonym_version=str(values["synonym_version"]),
        fixedpoint_width=values["fixedpoint_width"],
        quantization_tier=values["quantization_tier"],
        workers=values["workers"],
        snapshot_interval=values["snapshot_interval"],
        sentence_budget=values["sentence_budget"],
        q_format=make_qformat(values["fixedpoint_width"], values["quantization_tier"]),
        raw=config.raw,
        fingerprint="",
    )
    object.__setattr__(rebuilt, "fingerprint", fingerprint(rebuilt.business_fingerprint_obj()))
    return rebuilt


def apply_config_edits(config, commands, failures):
    for command in commands:
        op = command["op"]
        if op not in CONFIG_OPS:
            continue
        try:
            if op == "set_max_sentences":
                if not isinstance(command["value"], int) or command["value"] <= 0:
                    raise PipelineError(E_INPUT_INVALID, "max_sentences must be a positive integer")
                config = _rebuild_config(config, max_sentences=command["value"])
            elif op == "set_coverage_floor":
                value = float(command["value"])
                if not 0.0 <= value <= 1.0:
                    raise PipelineError(E_INPUT_INVALID, "coverage_floor must lie within [0, 1]")
                config = _rebuild_config(config, coverage_floor=value)
            elif op == "set_weights":
                config = _rebuild_config(
                    config, term_weight=float(command["term_weight"]),
                    entity_weight=float(command["entity_weight"]),
                )
            elif op == "set_position_decay":
                config = _rebuild_config(config, position_decay=float(command["value"]))
            elif op == "set_style_template":
                config = _rebuild_config(config, style_template_id=str(command["template_id"]))
            elif op == "set_quantization":
                config = _rebuild_config(
                    config, fixedpoint_width=int(command["decimal_width"]),
                    quantization_tier=str(command["quantization_tier"]),
                )
        except PipelineError as exc:
            failures.append(exc.to_record())
    return config


def apply_table_edits(table, commands, config, failures):
    rules = list(table.rules)
    templates = list(table.templates)
    version = table.version
    invalid_rules = []
    for command in commands:
        op = command["op"]
        if op not in TABLE_OPS:
            continue
        if op == "bump_table_version":
            version = str(command["version"])
            continue
        if op == "add_rule":
            payload = command["rule"]
            reasons = []
            if payload["template"] not in templates:
                reasons.append("template %r is not declared" % payload["template"])
            if not payload.get("target"):
                reasons.append("target text is empty")
            if not payload.get("source"):
                reasons.append("source text is empty")
            if reasons:
                invalid_rules.append({"code": "E_REWRITE_RULE_INVALID", "object_id": payload["id"],
                                      "message": "; ".join(reasons)})
                continue
            rules = [rule for rule in rules if rule.rule_id != payload["id"]]
            rules.append(RewriteRule(
                rule_id=payload["id"], template=payload["template"], source=payload["source"],
                target=payload["target"], enabled=bool(payload.get("enabled", True)),
                order=len(rules),
            ))
        elif op == "update_rule":
            index = next((i for i, rule in enumerate(rules) if rule.rule_id == command["rule_id"]), None)
            if index is None:
                failures.append({"code": E_INPUT_INVALID, "object_id": command["rule_id"],
                                 "message": "rule not found"})
                continue
            current = rules[index]
            rules[index] = RewriteRule(
                rule_id=current.rule_id,
                template=command.get("template", current.template),
                source=command.get("source", current.source),
                target=command.get("target", current.target),
                enabled=bool(command.get("enabled", current.enabled)),
                order=current.order,
            )
        elif op == "delete_rule":
            rules = [rule for rule in rules if rule.rule_id != command["rule_id"]]
    rules = tuple(rules)
    rebuilt = RewriteTable(
        version=version,
        templates=tuple(templates),
        rules=rules,
        fingerprint=fingerprint({
            "version": version,
            "templates": templates,
            "rules": [
                {"id": r.rule_id, "template": r.template, "source": r.source,
                 "target": r.target, "enabled": r.enabled, "order": r.order}
                for r in rules
            ],
        }),
    )
    return rebuilt, invalid_rules


def apply_all(documents, config, table, commands, failures):
    documents = apply_document_edits(documents, commands, failures)
    config = apply_config_edits(config, commands, failures)
    table, invalid_rules = apply_table_edits(table, commands, config, failures)
    failures.extend(invalid_rules)
    return documents, config, table
