from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .canonical import sha256_json, sha256_text
from .errors import ValidationError


ACTION_NAMES = ("SHIFT", "LEFT_ARC", "RIGHT_ARC", "FINISH")
ACTION_CODES = {name: index for index, name in enumerate(ACTION_NAMES)}
ACTION_BY_CODE = {index: name for name, index in ACTION_CODES.items()}

ACTION_ALIASES = {
    "shift": "SHIFT",
    "s": "SHIFT",
    "left": "LEFT_ARC",
    "left_arc": "LEFT_ARC",
    "left_reduce": "LEFT_ARC",
    "reduce_left": "LEFT_ARC",
    "l": "LEFT_ARC",
    "right": "RIGHT_ARC",
    "right_arc": "RIGHT_ARC",
    "right_reduce": "RIGHT_ARC",
    "reduce_right": "RIGHT_ARC",
    "r": "RIGHT_ARC",
    "finish": "FINISH",
    "end": "FINISH",
    "f": "FINISH",
}

PARSE_MODES = ("greedy", "exhaustive")
ALLOWED_QUANTIZATION_LEVELS = (1, 10, 100, 1000, 10000)
MIN_FIXED_DECIMALS = 0
MAX_FIXED_DECIMALS = 6
MAX_THREADS = 32


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    words: tuple[str, ...]
    pos_tags: tuple[str, ...]
    gold_heads: tuple[int, ...] | None = None
    gold_relations: tuple[str, ...] | None = None
    line_no: int = 0
    fingerprint: str = ""

    def with_id(self, sentence_id: str) -> "Sentence":
        return Sentence(
            sentence_id=sentence_id,
            words=self.words,
            pos_tags=self.pos_tags,
            gold_heads=self.gold_heads,
            gold_relations=self.gold_relations,
            line_no=self.line_no,
            fingerprint=self.fingerprint,
        )

    def with_pos(self, position: int, pos_tag: str) -> "Sentence":
        tags = list(self.pos_tags)
        tags[position] = pos_tag
        payload = sentence_payload(
            self.sentence_id,
            self.words,
            tuple(tags),
            self.gold_heads,
            self.gold_relations,
        )
        return Sentence(
            sentence_id=self.sentence_id,
            words=self.words,
            pos_tags=tuple(tags),
            gold_heads=self.gold_heads,
            gold_relations=self.gold_relations,
            line_no=self.line_no,
            fingerprint=sha256_json(payload),
        )

    def with_words(self, words: tuple[str, ...], pos_tags: tuple[str, ...]) -> "Sentence":
        payload = sentence_payload(
            self.sentence_id,
            words,
            pos_tags,
            self.gold_heads,
            self.gold_relations,
        )
        return Sentence(
            sentence_id=self.sentence_id,
            words=words,
            pos_tags=pos_tags,
            gold_heads=self.gold_heads,
            gold_relations=self.gold_relations,
            line_no=self.line_no,
            fingerprint=sha256_json(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sentence_id": self.sentence_id,
            "words": list(self.words),
            "pos_tags": list(self.pos_tags),
            "fingerprint": self.fingerprint,
            "line_no": self.line_no,
        }
        if self.gold_heads is not None:
            result["gold_heads"] = list(self.gold_heads)
        if self.gold_relations is not None:
            result["gold_relations"] = list(self.gold_relations)
        return result


def sentence_payload(
    sentence_id: str,
    words: tuple[str, ...],
    pos_tags: tuple[str, ...],
    gold_heads: tuple[int, ...] | None = None,
    gold_relations: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sentence_id": sentence_id,
        "words": list(words),
        "pos_tags": list(pos_tags),
    }
    if gold_heads is not None:
        payload["gold_heads"] = list(gold_heads)
    if gold_relations is not None:
        payload["gold_relations"] = list(gold_relations)
    return payload


@dataclass(frozen=True)
class Model:
    feature_dim: int
    num_buckets: int
    action_weights: dict[str, tuple[float, ...]]
    feature_hashes: dict[str, int]
    vocabulary: tuple[str, ...] = ()
    pos_tags: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
    fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "num_buckets": self.num_buckets,
            "action_weights": {
                action: list(self.action_weights[action]) for action in ACTION_NAMES
            },
            "feature_hashes": dict(sorted(self.feature_hashes.items())),
            "vocabulary": list(self.vocabulary),
            "pos_tags": list(self.pos_tags),
        }


@dataclass(frozen=True)
class QuantizedModel:
    weights: dict[str, tuple[int, ...]]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {action: list(self.weights[action]) for action in ACTION_NAMES},
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class Constraints:
    max_arc_distance: int | None = None
    max_stack_size: int | None = None
    max_dependents_per_head: int | None = None
    forbid_left_dependent_pos: tuple[str, ...] = ()
    forbid_right_dependent_pos: tuple[str, ...] = ()
    finish_requires_single_root: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_arc_distance": self.max_arc_distance,
            "max_stack_size": self.max_stack_size,
            "max_dependents_per_head": self.max_dependents_per_head,
            "forbid_left_dependent_pos": list(self.forbid_left_dependent_pos),
            "forbid_right_dependent_pos": list(self.forbid_right_dependent_pos),
            "finish_requires_single_root": self.finish_requires_single_root,
        }


@dataclass(frozen=True)
class Config:
    mode: str
    max_dependency_distance: int
    max_transition_steps: int
    max_threads: int
    batch_parse_point_budget: int
    snapshot_commit_interval_sentences: int
    fixed_point_decimals: int
    quantization_level: int
    constraints: Constraints
    max_exhaustive_nodes: int = 250_000
    max_pruned_records: int = 2_000
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
    fingerprint: str = ""

    @property
    def quantization_fingerprint(self) -> str:
        return sha256_json(
            {
                "fixed_point_decimals": self.fixed_point_decimals,
                "quantization_level": self.quantization_level,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_dependency_distance": self.max_dependency_distance,
            "max_transition_steps": self.max_transition_steps,
            "max_threads": self.max_threads,
            "batch_parse_point_budget": self.batch_parse_point_budget,
            "snapshot_commit_interval_sentences": self.snapshot_commit_interval_sentences,
            "fixed_point_decimals": self.fixed_point_decimals,
            "quantization_level": self.quantization_level,
            "constraints": self.constraints.to_dict(),
            "max_exhaustive_nodes": self.max_exhaustive_nodes,
            "max_pruned_records": self.max_pruned_records,
        }


@dataclass(frozen=True)
class Failure:
    sentence_id: str
    code: str
    message: str
    line_no: int = 0
    injection_id: str | None = None
    stage: str = "sentence_load"

    def to_dict(self) -> dict[str, Any]:
        result = {
            "sentence_id": self.sentence_id,
            "code": self.code,
            "message": self.message,
            "line_no": self.line_no,
            "stage": self.stage,
        }
        if self.injection_id is not None:
            result["injection_id"] = self.injection_id
        return result

    @property
    def sort_key(self) -> tuple[str, int, str, str]:
        return (self.sentence_id, self.line_no, self.code, self.injection_id or "")


@dataclass(frozen=True)
class Injection:
    injection_id: str
    sentence_id: str
    stage: str
    kind: str
    token_index: int = 0
    pos_tag: str = "__INJECTED_UNKNOWN_POS__"
    weight_action: str = "SHIFT"
    weight_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "sentence_id": self.sentence_id,
            "stage": self.stage,
            "kind": self.kind,
            "token_index": self.token_index,
            "pos_tag": self.pos_tag,
            "weight_action": self.weight_action,
            "weight_index": self.weight_index,
        }


@dataclass(frozen=True)
class Project:
    sample_dir: Any
    sentences: tuple[Sentence, ...]
    failures: tuple[Failure, ...]
    model: Model
    config: Config
    injections: tuple[Injection, ...]
    vocabulary: tuple[str, ...]
    pos_tags: tuple[str, ...]
    corpus_fingerprint: str = ""
    raw_sentences: tuple[dict[str, Any], ...] = ()


def sentence_from_dict(obj: dict[str, Any], line_no: int = 0) -> Sentence:
    sid_value = obj.get("sentence_id", obj.get("id", obj.get("sid")))
    if not isinstance(sid_value, str) or not sid_value.strip():
        raise ValidationError("E_SENT_INVALID", "sentence id must be a non-empty string")
    words_value = obj.get("words", obj.get("tokens"))
    pos_value = obj.get("pos_tags", obj.get("pos", obj.get("tags")))
    words = _split_sequence(words_value, "words")
    pos_tags = _split_sequence(pos_value, "pos_tags")
    gold_heads = _optional_int_tuple(obj.get("gold_heads"), "gold_heads")
    gold_relations = _optional_str_tuple(obj.get("gold_relations"), "gold_relations")
    if gold_heads is not None and len(gold_heads) != len(words):
        raise ValidationError("E_SENT_INVALID", "gold_heads length differs from words length")
    if gold_relations is not None and len(gold_relations) != len(words):
        raise ValidationError("E_SENT_INVALID", "gold_relations length differs from words length")
    payload = sentence_payload(
        sid_value,
        words,
        pos_tags,
        gold_heads,
        gold_relations,
    )
    return Sentence(
        sentence_id=sid_value,
        words=words,
        pos_tags=pos_tags,
        gold_heads=gold_heads,
        gold_relations=gold_relations,
        line_no=line_no,
        fingerprint=sha256_json(payload),
    )


def _split_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part for part in value.strip().split(" ") if part != "")
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValidationError("E_SENT_INVALID", f"{field_name} must contain strings")
        return tuple(value)
    raise ValidationError("E_SENT_INVALID", f"missing or invalid {field_name}")


def _optional_int_tuple(value: Any, field_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValidationError("E_SENT_INVALID", f"{field_name} must be an integer list")
    return tuple(value)


def _optional_str_tuple(value: Any, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValidationError("E_SENT_INVALID", f"{field_name} must be a string list")
    return tuple(value)


def normalize_action_name(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in ACTION_ALIASES:
        return ACTION_ALIASES[key]
    if key.upper() in ACTION_CODES:
        return key.upper()
    raise ValidationError("E_MODEL_INVALID", f"unknown action name: {value}")


def model_from_dict(raw: dict[str, Any], vocabulary: list[str] | None = None,
                    pos_tags: list[str] | None = None) -> Model:
    if not isinstance(raw, dict):
        raise ValidationError("E_MODEL_INVALID", "model JSON must be an object")
    feature_dim = raw.get("feature_dim", raw.get("feature_dimension"))
    num_buckets = raw.get("num_buckets", raw.get("bucket_count", raw.get("hash_buckets")))
    if not _positive_int(feature_dim):
        raise ValidationError("E_MODEL_INVALID", "feature_dim must be a positive integer")
    if not _positive_int(num_buckets):
        raise ValidationError("E_MODEL_INVALID", "num_buckets must be a positive integer")

    raw_weights = raw.get("action_weights", raw.get("weights"))
    if not isinstance(raw_weights, dict):
        raise ValidationError("E_MODEL_INVALID", "action_weights must be an object")
    weight_by_action: dict[str, tuple[float, ...]] = {}
    for key, value in raw_weights.items():
        action = normalize_action_name(key)
        if action in weight_by_action:
            raise ValidationError("E_MODEL_INVALID", f"duplicate action weights for {action}")
        values = _weight_values(value, feature_dim, action)
        weight_by_action[action] = values
    missing = [action for action in ACTION_NAMES if action not in weight_by_action]
    if missing:
        raise ValidationError(
            "E_MODEL_INVALID",
            "missing action weights: " + ", ".join(missing),
        )

    raw_hashes = raw.get("feature_hashes", raw.get("feature_hash", raw.get("hashes")))
    if not isinstance(raw_hashes, dict):
        raise ValidationError("E_MODEL_INVALID", "feature_hashes must be an object")
    feature_hashes: dict[str, int] = {}
    for key, value in raw_hashes.items():
        if not isinstance(key, str) or not key:
            raise ValidationError("E_MODEL_INVALID", "feature hash keys must be non-empty strings")
        if isinstance(value, dict):
            value = value.get("bucket", value.get("hash"))
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError("E_MODEL_INVALID", f"feature hash for {key!r} is not an integer")
        if not 0 <= value < num_buckets:
            raise ValidationError(
                "E_MODEL_INVALID",
                f"feature hash for {key!r} is outside [0, {num_buckets})",
            )
        feature_hashes[key] = value

    vocab_source = raw.get("vocabulary", raw.get("words", []))
    pos_source = raw.get("pos_tags", raw.get("pos", []))
    if vocabulary is None:
        if isinstance(vocab_source, dict):
            vocab_source = list(vocab_source.keys())
        vocabulary = list(vocab_source) if isinstance(vocab_source, list) else []
    if pos_tags is None:
        if isinstance(pos_source, dict):
            pos_source = list(pos_source.keys())
        pos_tags = list(pos_source) if isinstance(pos_source, list) else []
    if not all(isinstance(item, str) and item for item in vocabulary):
        raise ValidationError("E_MODEL_INVALID", "model vocabulary must contain strings")
    if not all(isinstance(item, str) and item for item in pos_tags):
        raise ValidationError("E_MODEL_INVALID", "model pos_tags must contain strings")

    normalized = {
        "feature_dim": feature_dim,
        "num_buckets": num_buckets,
        "action_weights": {action: list(weight_by_action[action]) for action in ACTION_NAMES},
        "feature_hashes": dict(sorted(feature_hashes.items())),
        "vocabulary": list(vocabulary),
        "pos_tags": list(pos_tags),
    }
    return Model(
        feature_dim=feature_dim,
        num_buckets=num_buckets,
        action_weights=weight_by_action,
        feature_hashes=feature_hashes,
        vocabulary=tuple(vocabulary),
        pos_tags=tuple(pos_tags),
        raw=normalized,
        fingerprint=sha256_json(normalized),
    )


def _weight_values(value: Any, feature_dim: int, action: str) -> tuple[float, ...]:
    if isinstance(value, dict):
        value = value.get("values", value.get("weights"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        values = [float(value)] * feature_dim
    elif isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValidationError("E_MODEL_INVALID", f"action weight for {action} is not numeric")
            values.append(float(item))
    else:
        raise ValidationError("E_MODEL_INVALID", f"action weight for {action} has invalid shape")
    if len(values) != feature_dim:
        raise ValidationError(
            "E_MODEL_INVALID",
            f"action weight for {action} has length {len(values)}, expected {feature_dim}",
        )
    if not all(math.isfinite(item) for item in values):
        raise ValidationError("E_MODEL_INVALID", f"action weight for {action} is not finite")
    return tuple(values)


def config_from_dict(raw: dict[str, Any]) -> Config:
    if not isinstance(raw, dict):
        raise ValidationError("E_CONFIG_INVALID", "config JSON must be an object")
    mode = str(raw.get("mode", raw.get("parse_mode", "greedy"))).strip().lower()
    if mode not in PARSE_MODES:
        raise ValidationError("E_CONFIG_INVALID", f"mode must be one of {PARSE_MODES}")

    max_dependency_distance = _positive_int_value(
        raw.get("max_dependency_distance", raw.get("dependency_limit", 8)),
        "max_dependency_distance",
    )
    max_transition_steps = _positive_int_value(
        raw.get("max_transition_steps", raw.get("max_steps", 64)),
        "max_transition_steps",
    )
    max_threads = _positive_int_value(
        raw.get("max_threads", raw.get("thread_pool_size", 4)),
        "max_threads",
    )
    if max_threads > MAX_THREADS:
        raise ValidationError(
            "E_CONFIG_INVALID",
            f"max_threads must not exceed {MAX_THREADS}",
        )
    batch_budget = _positive_int_value(
        raw.get("batch_parse_point_budget", raw.get("batch_budget", 40)),
        "batch_parse_point_budget",
    )
    snapshot_interval = _positive_int_value(
        raw.get(
            "snapshot_commit_interval_sentences",
            raw.get("snapshot_interval", 2),
        ),
        "snapshot_commit_interval_sentences",
    )
    decimals = raw.get("fixed_point_decimals", raw.get("fixed_decimals", 3))
    if isinstance(decimals, bool) or not isinstance(decimals, int):
        raise ValidationError("E_CONFIG_INVALID", "fixed_point_decimals must be an integer")
    if not MIN_FIXED_DECIMALS <= decimals <= MAX_FIXED_DECIMALS:
        raise ValidationError(
            "E_CONFIG_INVALID",
            f"fixed_point_decimals must be in [{MIN_FIXED_DECIMALS}, {MAX_FIXED_DECIMALS}]",
        )
    quant_level = raw.get("quantization_level", raw.get("quant_level", 1))
    if isinstance(quant_level, bool) or not isinstance(quant_level, int):
        raise ValidationError("E_CONFIG_INVALID", "quantization_level must be an integer")
    if quant_level not in ALLOWED_QUANTIZATION_LEVELS:
        raise ValidationError(
            "E_CONFIG_INVALID",
            f"quantization_level must be one of {ALLOWED_QUANTIZATION_LEVELS}",
        )

    raw_constraints = raw.get("constraints", raw.get("action_constraints", {}))
    if raw_constraints is None:
        raw_constraints = {}
    if not isinstance(raw_constraints, dict):
        raise ValidationError("E_CONFIG_INVALID", "constraints must be an object")
    max_arc_distance = _optional_positive_int(
        raw_constraints.get("max_arc_distance"), "constraints.max_arc_distance"
    )
    max_stack_size = _optional_positive_int(
        raw_constraints.get("max_stack_size"), "constraints.max_stack_size"
    )
    max_dependents = _optional_positive_int(
        raw_constraints.get("max_dependents_per_head"),
        "constraints.max_dependents_per_head",
    )
    forbid_left = _pos_tuple(
        raw_constraints.get("forbid_left_dependent_pos", []),
        "constraints.forbid_left_dependent_pos",
    )
    forbid_right = _pos_tuple(
        raw_constraints.get("forbid_right_dependent_pos", []),
        "constraints.forbid_right_dependent_pos",
    )
    finish_single = raw_constraints.get("finish_requires_single_root", True)
    if not isinstance(finish_single, bool):
        raise ValidationError(
            "E_CONFIG_INVALID",
            "constraints.finish_requires_single_root must be boolean",
        )
    max_exhaustive_nodes = _positive_int_value(
        raw.get("max_exhaustive_nodes", 250_000),
        "max_exhaustive_nodes",
    )
    max_pruned_records = _positive_int_value(
        raw.get("max_pruned_records", 2_000),
        "max_pruned_records",
    )
    constraints = Constraints(
        max_arc_distance=max_arc_distance,
        max_stack_size=max_stack_size,
        max_dependents_per_head=max_dependents,
        forbid_left_dependent_pos=forbid_left,
        forbid_right_dependent_pos=forbid_right,
        finish_requires_single_root=finish_single,
    )
    normalized = {
        "mode": mode,
        "max_dependency_distance": max_dependency_distance,
        "max_transition_steps": max_transition_steps,
        "max_threads": max_threads,
        "batch_parse_point_budget": batch_budget,
        "snapshot_commit_interval_sentences": snapshot_interval,
        "fixed_point_decimals": decimals,
        "quantization_level": quant_level,
        "constraints": constraints.to_dict(),
        "max_exhaustive_nodes": max_exhaustive_nodes,
        "max_pruned_records": max_pruned_records,
    }
    return Config(
        mode=mode,
        max_dependency_distance=max_dependency_distance,
        max_transition_steps=max_transition_steps,
        max_threads=max_threads,
        batch_parse_point_budget=batch_budget,
        snapshot_commit_interval_sentences=snapshot_interval,
        fixed_point_decimals=decimals,
        quantization_level=quant_level,
        constraints=constraints,
        max_exhaustive_nodes=max_exhaustive_nodes,
        max_pruned_records=max_pruned_records,
        raw=normalized,
        fingerprint=sha256_json(normalized),
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_int_value(value: Any, field_name: str) -> int:
    if not _positive_int(value):
        raise ValidationError("E_CONFIG_INVALID", f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int_value(value, field_name)


def _pos_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValidationError("E_CONFIG_INVALID", f"{field_name} must be a string list")
    return tuple(value)


def copy_model_with_nonfinite(raw: dict[str, Any], action: str, index: int) -> dict[str, Any]:
    """Return a deep-enough copy containing a real non-finite input fault."""

    import copy

    result = copy.deepcopy(raw)
    weights = result.get("action_weights", result.get("weights"))
    if not isinstance(weights, dict):
        weights = {}
        result["action_weights"] = weights
    target_key = None
    for key in weights:
        try:
            if normalize_action_name(key) == normalize_action_name(action):
                target_key = key
                break
        except ValidationError:
            continue
    if target_key is None:
        target_key = action
        weights[target_key] = [0.0] * int(result.get("feature_dim", 1))
    value = weights[target_key]
    if isinstance(value, dict):
        value = value.get("values", value.get("weights", []))
        weights[target_key] = value
    if not isinstance(value, list) or not value:
        value = [0.0] * int(result.get("feature_dim", 1))
        weights[target_key] = value
    value[index % len(value)] = float("nan")
    return result

