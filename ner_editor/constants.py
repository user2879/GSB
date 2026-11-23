"""Shared deterministic constants for the local NER editor."""

from __future__ import annotations


LABELS = ("B", "I", "E", "S", "O")
LABEL_CODE = {label: index for index, label in enumerate(LABELS)}

# Type codes are deliberately lexical: LOC < ORG < PER < TIME. Every
# tie-break uses this order, including candidate types attached to a span.
TYPE_CODES = ("LOC", "ORG", "PER", "TIME")
TYPE_CODE = {type_code: index for index, type_code in enumerate(TYPE_CODES)}
TYPE_ALIASES = {
    "LOC": "LOC",
    "LOCATION": "LOC",
    "地点": "LOC",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "机构": "ORG",
    "PER": "PER",
    "PERSON": "PER",
    "人物": "PER",
    "TIME": "TIME",
    "时间": "TIME",
}

STAGE_ORDER = ("feature", "emission", "decode", "merge")

E_CORPUS_INVALID = "E_CORPUS_INVALID"
E_MODEL_PARAM_INVALID = "E_MODEL_PARAM_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_EMISSION_INVALID = "E_EMISSION_INVALID"
E_LATENCY_BUDGET_EXCEEDED = "E_LATENCY_BUDGET_EXCEEDED"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"

SCHEMA_VERSION = "ner-editor-v1"
FIXED_ALGORITHM = "sha256-rotate-dot-half-away-v1"
FEATURE_ALGORITHM = "char-context-position-hit-sha256-v1"
MERGE_ALGORITHM = "repair-then-type-vote-v1"

TERMINAL_COMMAND_TYPES = frozenset({"quit", "exit", "shutdown"})

COMMAND_ALIASES = {
    "vocab_add": "vocab_add",
    "add_vocab": "vocab_add",
    "add_vocabulary": "vocab_add",
    "vocab_delete": "vocab_delete",
    "delete_vocab": "vocab_delete",
    "delete_vocabulary": "vocab_delete",
    "vocab_replace": "vocab_replace",
    "replace_vocab": "vocab_replace",
    "replace_vocabulary": "vocab_replace",
    "corpus_replace": "corpus_replace",
    "replace_sample": "corpus_replace",
    "corpus_delete": "corpus_delete",
    "delete_sample": "corpus_delete",
    "emission_adjust": "emission_adjust",
    "adjust_emission": "emission_adjust",
    "transition_adjust": "transition_adjust",
    "adjust_transition": "transition_adjust",
    "fixed_point_set": "fixed_point_set",
    "set_fixed_point": "fixed_point_set",
    "quantization_set": "quantization_set",
    "set_quantization": "quantization_set",
    "noop": "noop",
    "ping": "noop",
    "quit": "quit",
    "exit": "quit",
    "shutdown": "quit",
}
