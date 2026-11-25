"""Local JSON loading and strict validation (requirement 1).

Every loader fails on the *first* offending entry with ``E_SPEC`` and reports
both a stable entry identifier and the offending field path. Validation order
is deterministic and independent of dictionary iteration order.
"""

import json
import os

from .canonical import canonical_bytes, digest
from .errors import SpecError
from .fixedpoint import (
    SUPPORTED_SCALES,
    SUPPORTED_WIDTHS,
    validate_scale,
    validate_width,
)

SPEC_FILES = (
    "vocabulary.json",
    "rules.json",
    "weights.json",
    "precision.json",
    "fragments.json",
    "sessions.json",
)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _fail(entry, field, message, **context):
    raise SpecError(message, entry=entry, field=field, **context)


def _require_type(payload, field, expected, entry):
    value = payload.get(field)
    if not isinstance(value, expected):
        _fail(entry, field, f"field '{field}' must be {expected}", value=repr(value))
    return value


def _body_without_digest(payload):
    return {key: value for key, value in payload.items() if key != "digest"}


def _check_digest(payload, entry):
    if not isinstance(payload, dict):
        _fail(entry, "$", "specification document must be a JSON object")
    declared = payload.get("digest")
    if not isinstance(declared, str) or len(declared) != 64:
        _fail(entry, "digest", "missing or malformed content digest")
    actual = digest(_body_without_digest(payload))
    if declared != actual:
        _fail(entry, "digest", "content digest does not match document body",
              expected=actual, actual=declared)
    return declared


class Config:
    def __init__(self, vocabulary, rule, weights, precision):
        self.vocabulary = vocabulary
        self.rule = rule
        self.weights = weights
        self.precision = precision
        self.versions = {
            "rule_version": rule["version"],
            "weight_version": weights["version"],
            "precision_version": precision["version"],
        }
        self.terms = set(vocabulary["terms"])
        self.channels = set(precision["channels"])
        self.topics = tuple(sorted(rule["topics"].keys()))
        self.digest = digest(
            {
                "vocabulary": vocabulary,
                "rule": rule,
                "weights": weights,
                "precision": precision,
            }
        )


class Dataset:
    def __init__(self, config, fragments, sessions):
        self.config = config
        self.fragments = fragments
        self.sessions = sessions


def _load_document(path):
    if not os.path.exists(path):
        _fail(os.path.basename(path), "$", "required specification file is missing")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        _fail(os.path.basename(path), "$", f"invalid JSON: {exc.msg}",
              line=exc.lineno, column=exc.colno)


def _load_vocabulary(path):
    payload = _load_document(path)
    entry = "vocabulary.json"
    _check_digest(payload, entry)
    version = _require_type(payload, "version", int, entry)
    if not _is_int(version) or version < 1:
        _fail(entry, "version", "version must be a positive integer")
    terms = _require_type(payload, "terms", list, entry)
    seen = set()
    if not terms:
        _fail(entry, "terms", "controlled vocabulary must not be empty")
    for index, term in enumerate(terms):
        if not isinstance(term, str) or not term:
            _fail(entry, ["terms", index], "term must be a non-empty string")
        if term in seen:
            _fail(entry, ["terms", index], "duplicate controlled term", term=term)
        seen.add(term)
    return payload


def _load_rule(path, vocabulary):
    payload = _load_document(path)
    entry = "rules.json"
    _check_digest(payload, entry)
    if not _is_int(payload.get("version")) or payload["version"] < 1:
        _fail(entry, "version", "rule version must be a positive integer")
    topics = _require_type(payload, "topics", dict, entry)
    if not topics:
        _fail(entry, "topics", "at least one topic rule is required")
    normalized = {}
    for topic in sorted(topics):
        spec = topics[topic]
        if not isinstance(topic, str) or not topic:
            _fail(entry, ["topics", topic], "topic label must be a non-empty string")
        if not isinstance(spec, dict):
            _fail(entry, ["topics", topic], "topic rule must be an object")
        anchors = spec.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            _fail(entry, ["topics", topic, "anchors"],
                  "anchors must be a non-empty list")
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, str) or anchor not in vocabulary["terms"]:
                _fail(entry, ["topics", topic, "anchors", index],
                      "anchor must reference a controlled term", anchor=anchor)
        normalized[topic] = {"anchors": sorted(set(anchors))}
    result = dict(payload)
    result["topics"] = normalized
    return result


def _load_weights(path, rule, precision):
    payload = _load_document(path)
    entry = "weights.json"
    _check_digest(payload, entry)
    if not _is_int(payload.get("version")) or payload["version"] < 1:
        _fail(entry, "version", "weight version must be a positive integer")
    channel_weights = _require_type(payload, "channel_weights", dict, entry)
    for channel in sorted(channel_weights):
        if channel not in precision["channels"]:
            _fail(entry, ["channel_weights", channel],
                  "weight references an unregistered channel")
        raw = channel_weights[channel]
        if not _is_int(raw) or raw < 0:
            _fail(entry, ["channel_weights", channel],
                  "raw channel weight must be a non-negative integer")
    for channel in sorted(precision["channels"]):
        if channel not in channel_weights:
            _fail(entry, "channel_weights",
                  f"channel '{channel}' has no raw weight")
    topic_weights = _require_type(payload, "topic_weights", dict, entry)
    for topic in sorted(topic_weights):
        if topic not in rule["topics"]:
            _fail(entry, ["topic_weights", topic],
                  "topic weight references an unknown topic label")
        raw = topic_weights[topic]
        if not _is_int(raw) or raw < 0:
            _fail(entry, ["topic_weights", topic],
                  "raw topic weight must be a non-negative integer")
    return payload


def _load_precision(path):
    payload = _load_document(path)
    entry = "precision.json"
    _check_digest(payload, entry)
    if not _is_int(payload.get("version")) or payload["version"] < 1:
        _fail(entry, "version", "precision version must be a positive integer")
    width = payload.get("width")
    if not _is_int(width) or width not in SUPPORTED_WIDTHS:
        _fail(entry, "width", "width must be one of the supported integers",
              supported=list(SUPPORTED_WIDTHS))
    validate_width(width)
    scale = payload.get("scale")
    if not _is_int(scale) or scale not in SUPPORTED_SCALES:
        _fail(entry, "scale", "scale must be one of the supported integers",
              supported=list(SUPPORTED_SCALES))
    validate_scale(scale)
    channels = _require_type(payload, "channels", list, entry)
    seen = set()
    for index, channel in enumerate(channels):
        if not isinstance(channel, str) or not channel:
            _fail(entry, ["channels", index], "channel must be a non-empty string")
        if channel in seen:
            _fail(entry, ["channels", index], "duplicate recall channel")
        seen.add(channel)
    if not channels:
        _fail(entry, "channels", "at least one recall channel is required")
    return payload


def _load_fragments(path, config):
    payload = _load_document(path)
    entry = "fragments.json"
    _check_digest(payload, entry)
    fragments = _require_type(payload, "fragments", list, entry)
    seen = set()
    for index, fragment in enumerate(fragments):
        if not isinstance(fragment, dict):
            _fail(entry, ["fragments", index], "fragment must be an object")
        fragment_id = fragment.get("fragment_id")
        if not isinstance(fragment_id, str) or not fragment_id:
            _fail(entry, ["fragments", index, "fragment_id"],
                  "fragment_id must be a non-empty string")
        if fragment_id in seen:
            _fail(entry, ["fragments", index, "fragment_id"],
                  "duplicate fragment identifier", value=fragment_id)
        seen.add(fragment_id)
        terms = fragment.get("terms")
        if not isinstance(terms, list) or not terms:
            _fail(entry, [fragment_id, "terms"],
                  "terms must be a non-empty list")
        for term_index, term in enumerate(terms):
            if term not in config.terms:
                _fail(entry, [fragment_id, "terms", term_index],
                      "term is outside the controlled vocabulary", term=term)
        topic = fragment.get("topic")
        if topic not in config.rule["topics"]:
            _fail(entry, [fragment_id, "topic"],
                  "fragment references an unknown topic label", topic=topic)
        channel = fragment.get("channel")
        if channel not in config.channels:
            _fail(entry, [fragment_id, "channel"],
                  "fragment references an unregistered recall channel",
                  channel=channel)
    return fragments


def _load_sessions(path, config):
    payload = _load_document(path)
    entry = "sessions.json"
    _check_digest(payload, entry)
    sessions = _require_type(payload, "sessions", list, entry)
    seen_sessions = set()
    if not sessions:
        _fail(entry, "sessions", "at least one session is required")
    for session in sessions:
        if not isinstance(session, dict):
            _fail(entry, "sessions", "session must be an object")
        session_id = session.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            _fail(entry, "sessions", "session_id must be a non-empty string")
        if session_id in seen_sessions:
            _fail(entry, session_id, "duplicate session identifier",
                  field="session_id")
        seen_sessions.add(session_id)
        rounds = session.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            _fail(entry, [session_id, "rounds"],
                  "rounds must be a non-empty list")
        seen_rounds = set()
        for round_request in rounds:
            if not isinstance(round_request, dict):
                _fail(entry, session_id, "round request must be an object")
            round_index = round_request.get("round")
            if not _is_int(round_index) or round_index < 0:
                _fail(entry, [session_id, "round"],
                      "round index must be a non-negative integer")
            if round_index in seen_rounds:
                _fail(entry, [session_id, "round"],
                      "duplicate round index in session", value=round_index)
            seen_rounds.add(round_index)
            rule_ref = round_request.get("rule_version_ref")
            if rule_ref != config.rule["version"]:
                _fail(entry, [session_id, round_index, "rule_version_ref"],
                      "round references an unknown topic-rule version",
                      value=rule_ref)
            precision_ref = round_request.get("precision_version_ref")
            if precision_ref != config.precision["version"]:
                _fail(entry,
                      [session_id, round_index, "precision_version_ref"],
                      "round references an unknown fixed-point version",
                      value=precision_ref)
            terms = round_request.get("query_terms")
            if not isinstance(terms, list) or not terms:
                _fail(entry, [session_id, round_index, "query_terms"],
                      "query_terms must be a non-empty list")
            for term_index, term in enumerate(terms):
                if term not in config.terms:
                    _fail(entry,
                          [session_id, round_index, "query_terms", term_index],
                          "query term is outside the controlled vocabulary",
                          term=term)
            scale = round_request.get("scale_decl")
            if not _is_int(scale) or scale not in SUPPORTED_SCALES:
                _fail(entry, [session_id, round_index, "scale_decl"],
                      "scale declaration must be a supported integer",
                      value=scale)
            if scale != config.precision["scale"]:
                _fail(entry, [session_id, round_index, "scale_decl"],
                      "round scale declaration conflicts with precision spec")
    return sessions


def load_dataset(sample_dir):
    vocabulary = _load_vocabulary(os.path.join(sample_dir, "vocabulary.json"))
    rule = _load_rule(os.path.join(sample_dir, "rules.json"), vocabulary)
    precision = _load_precision(os.path.join(sample_dir, "precision.json"))
    weights = _load_weights(
        os.path.join(sample_dir, "weights.json"), rule, precision
    )
    config = Config(vocabulary, rule, weights, precision)
    fragments = _load_fragments(
        os.path.join(sample_dir, "fragments.json"), config
    )
    sessions = _load_sessions(
        os.path.join(sample_dir, "sessions.json"), config
    )
    return Dataset(config, fragments, sessions)


def write_json(path, payload):
    body = _body_without_digest(payload) if isinstance(payload, dict) else payload
    if isinstance(payload, dict):
        payload = dict(body)
        payload["digest"] = digest(body)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(canonical_bytes(payload).decode("utf-8"))
        handle.write("\n")
    return payload
