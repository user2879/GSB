"""Configuration and synonym rewrite-table loading with strict validation."""

from dataclasses import dataclass, field

from .errors import E_INPUT_INVALID, E_REWRITE_RULE_INVALID, PipelineError
from .fixedpoint import make_qformat, validate_fixedpoint
from .hashing import canonical, fingerprint
from . import yaml_lite


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_unit_interval(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


@dataclass(frozen=True)
class RewriteRule:
    rule_id: str
    template: str
    source: str
    target: str
    enabled: bool
    order: int


@dataclass(frozen=True)
class RewriteTable:
    version: str
    templates: tuple
    rules: tuple
    fingerprint: str

    def active_rules_for(self, template_id):
        return tuple(r for r in self.rules if r.enabled and r.template == template_id)


@dataclass(frozen=True)
class Config:
    max_sentences: int
    term_weight: float
    entity_weight: float
    coverage_floor: float
    position_decay: float
    style_template_id: str
    synonym_version: str
    fixedpoint_width: int
    quantization_tier: str
    workers: int
    snapshot_interval: int
    sentence_budget: int
    q_format: object
    raw: dict = field(repr=False)
    fingerprint: str

    def business_fingerprint_obj(self):
        obj = self.to_fingerprint_obj()
        for runtime_key in ("workers", "snapshot_interval", "sentence_budget"):
            obj.pop(runtime_key, None)
        return obj

    def runtime_fingerprint_obj(self):
        return {
            "workers": self.workers,
            "snapshot_interval": self.snapshot_interval,
            "sentence_budget": self.sentence_budget,
        }

    def to_fingerprint_obj(self):
        return {
            "max_sentences": self.max_sentences,
            "term_weight": self.term_weight,
            "entity_weight": self.entity_weight,
            "coverage_floor": self.coverage_floor,
            "position_decay": self.position_decay,
            "style_template_id": self.style_template_id,
            "synonym_version": self.synonym_version,
            "fixedpoint_width": self.fixedpoint_width,
            "quantization_tier": self.quantization_tier,
            "workers": self.workers,
            "snapshot_interval": self.snapshot_interval,
            "sentence_budget": self.sentence_budget,
        }


def _require(mapping, key, ctx):
    if key not in mapping:
        raise PipelineError(E_INPUT_INVALID, "%s: missing key %r" % (ctx, key))
    return mapping[key]


def load_config(path):
    data = yaml_lite.load(path)
    if not isinstance(data, dict):
        raise PipelineError(E_INPUT_INVALID, "config root must be a mapping")

    summ = _require(data, "summarization", "config")
    fixed = _require(data, "fixedpoint", "config")
    runtime = _require(data, "runtime", "config")

    max_sentences = _require(summ, "max_sentences", "summarization")
    if not _is_positive_int(max_sentences):
        raise PipelineError(E_INPUT_INVALID, "max_sentences must be a positive integer")

    term_weight = float(_require(summ, "term_weight", "summarization"))
    entity_weight = float(_require(summ, "entity_weight", "summarization"))
    coverage_floor = float(_require(summ, "coverage_floor", "summarization"))
    if not _is_unit_interval(coverage_floor):
        raise PipelineError(E_INPUT_INVALID, "coverage_floor must lie within [0, 1]")
    position_decay = float(_require(summ, "position_decay", "summarization"))
    if position_decay < 0 or position_decay > 1:
        raise PipelineError(E_INPUT_INVALID, "position_decay must lie within [0, 1]")
    if term_weight < 0 or entity_weight < 0:
        raise PipelineError(E_INPUT_INVALID, "weights must be non-negative")
    style_template_id = _require(summ, "style_template_id", "summarization")
    synonym_version = str(_require(summ, "synonym_table_version", "summarization"))

    width = _require(fixed, "decimal_width", "fixedpoint")
    tier = _require(fixed, "quantization_tier", "fixedpoint")
    if not _is_positive_int(width):
        raise PipelineError(E_INPUT_INVALID, "decimal_width must be a positive integer")
    problem = validate_fixedpoint(width, tier)
    if problem is not None:
        raise PipelineError(E_INPUT_INVALID, problem)

    workers = _require(runtime, "workers", "runtime")
    snapshot_interval = _require(runtime, "snapshot_interval_docs", "runtime")
    sentence_budget = _require(runtime, "sentence_budget_per_batch", "runtime")
    for name, value in (("workers", workers), ("snapshot_interval_docs", snapshot_interval),
                        ("sentence_budget_per_batch", sentence_budget)):
        if not _is_positive_int(value):
            raise PipelineError(E_INPUT_INVALID, "%s must be a positive integer" % name)

    config = Config(
        max_sentences=max_sentences,
        term_weight=term_weight,
        entity_weight=entity_weight,
        coverage_floor=coverage_floor,
        position_decay=position_decay,
        style_template_id=str(style_template_id),
        synonym_version=synonym_version,
        fixedpoint_width=width,
        quantization_tier=tier,
        workers=workers,
        snapshot_interval=snapshot_interval,
        sentence_budget=sentence_budget,
        q_format=make_qformat(width, tier),
        raw=data,
        fingerprint="",
    )
    object.__setattr__(config, "fingerprint", fingerprint(config.business_fingerprint_obj()))
    return config


def load_rewrite_table(path, config, collect_invalid=None):
    data = yaml_lite.load(path)
    if not isinstance(data, dict):
        raise PipelineError(E_INPUT_INVALID, "rewrite table root must be a mapping")
    version = str(_require(data, "version", "rewrite_table"))
    if version != config.synonym_version:
        raise PipelineError(
            E_INPUT_INVALID,
            "rewrite table version %r does not match config synonym version %r"
            % (version, config.synonym_version),
        )
    template_items = _require(data, "templates", "rewrite_table")
    templates = []
    for item in template_items:
        template_id = _require(item, "id", "template")
        templates.append(str(template_id))
    templates = tuple(templates)
    if config.style_template_id not in templates:
        raise PipelineError(
            E_INPUT_INVALID,
            "style template %r is not declared in rewrite table" % config.style_template_id,
        )

    rules = []
    raw_rules = _require(data, "rules", "rewrite_table")
    for order, item in enumerate(raw_rules):
        rule_id = str(_require(item, "id", "rule"))
        template = str(_require(item, "template", "rule %s" % rule_id))
        source = _require(item, "source", "rule %s" % rule_id)
        target = _require(item, "target", "rule %s" % rule_id)
        enabled = bool(item.get("enabled", True))
        invalid_reasons = []
        if template not in templates:
            invalid_reasons.append("template %r is not declared" % template)
        if not isinstance(target, str) or target.strip() == "":
            invalid_reasons.append("target text is empty")
        if not isinstance(source, str) or source.strip() == "":
            invalid_reasons.append("source text is empty")
        if invalid_reasons:
            record = {
                "code": E_REWRITE_RULE_INVALID,
                "object_id": rule_id,
                "message": "; ".join(invalid_reasons),
            }
            if collect_invalid is not None:
                collect_invalid.append(record)
            continue
        rules.append(
            RewriteRule(
                rule_id=rule_id,
                template=template,
                source=source,
                target=target,
                enabled=enabled,
                order=order,
            )
        )

    table = RewriteTable(
        version=version,
        templates=templates,
        rules=tuple(rules),
        fingerprint=fingerprint(
            {
                "version": version,
                "templates": list(templates),
                "rules": [
                    {
                        "id": r.rule_id,
                        "template": r.template,
                        "source": r.source,
                        "target": r.target,
                        "enabled": r.enabled,
                        "order": r.order,
                    }
                    for r in rules
                ],
            }
        ),
    )
    return table


def config_canonical(config):
    return canonical(config.to_fingerprint_obj())
