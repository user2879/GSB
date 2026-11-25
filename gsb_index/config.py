"""Configuration loading and non-fatal validation.

Validation never stops at the first bad object.  Invalid documents or shard
descriptors are excluded from the build and represented by deterministic issue
records, while the remaining valid objects continue through the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import finite_number, fingerprint, read_json, stable_sorted_issues
from .codes import (
    E_BUCKET_CAPACITY_INDIVISIBLE,
    E_BUCKET_COUNT_INVALID,
    E_BUCKET_WEIGHT_COUNT_MISMATCH,
    E_CONFIG_READ,
    E_CONFIG_SHAPE,
    E_DOC_DUP_ID,
    E_DUP_PORT,
    E_DUP_SHARD_ID,
    E_INVALID_BUDGET,
    E_INVALID_DOCUMENT,
    E_INVALID_DOC_VERSION,
    E_INVALID_ENCODING,
    E_INVALID_FIELD,
    E_INVALID_PORT,
    E_INVALID_SCORING,
    E_INVALID_STOPWORD_FLAG,
    E_INVALID_VOCAB,
    E_INVALID_WEIGHT,
    E_NON_FINITE_WEIGHT,
    E_NON_POSITIVE_BUCKET_CAPACITY,
    E_NON_POSITIVE_CONCURRENCY,
    E_REVISION_NON_MONOTONIC,
    E_UNKNOWN_QUERY_TERM,
    E_UNKNOWN_SHARD,
    E_UNKNOWN_TERM,
    Issue,
)
from .models import (
    BuildBudget,
    CompressionProfile,
    Document,
    Project,
    Query,
    ScoringProfile,
    ShardSpec,
    Vocabulary,
)
from .revisions import build_revision_chain


FIELD_NAMES = ("title", "body")
DEFAULT_COMPRESSION = CompressionProfile(1, (1.0,), "plain-int-v1")


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _issue(code: str, message: str, object_type: str, object_id: str, field: str = "", **details: Any) -> Issue:
    return Issue(code, message, object_type, object_id, field, details)


def _parse_vocabulary(raw: Any, issues: list[Issue]) -> Vocabulary:
    if not isinstance(raw, dict):
        issues.append(_issue(E_INVALID_VOCAB, "vocabulary must be an object", "vocabulary", "vocabulary"))
        return Vocabulary(terms=(), stopwords=frozenset())
    terms_raw = raw.get("terms")
    stopwords_raw = raw.get("stopwords", [])
    if not isinstance(terms_raw, list) or not terms_raw:
        issues.append(_issue(E_INVALID_VOCAB, "vocabulary terms must be a non-empty list", "vocabulary", "vocabulary", "terms"))
        terms_raw = []
    if not isinstance(stopwords_raw, list):
        issues.append(_issue(E_INVALID_VOCAB, "vocabulary stopwords must be a list", "vocabulary", "vocabulary", "stopwords"))
        stopwords_raw = []
    terms: list[str] = []
    for index, term in enumerate(terms_raw):
        if not isinstance(term, str) or not term.strip():
            issues.append(_issue(E_INVALID_VOCAB, "term must be a non-empty string", "vocabulary", str(index), f"terms[{index}]"))
            continue
        terms.append(term)
    unique_terms = sorted(set(terms))
    if len(unique_terms) != len(terms):
        issues.append(_issue(E_INVALID_VOCAB, "vocabulary terms must be unique", "vocabulary", "vocabulary", "terms"))
    stops: list[str] = []
    term_set = set(unique_terms)
    for index, term in enumerate(stopwords_raw):
        if not isinstance(term, str) or not term.strip():
            issues.append(_issue(E_INVALID_VOCAB, "stopword must be a non-empty string", "vocabulary", str(index), f"stopwords[{index}]"))
            continue
        if term not in term_set:
            issues.append(_issue(E_UNKNOWN_TERM, "stopword must be in the controlled vocabulary", "vocabulary", term, "stopwords"))
            continue
        stops.append(term)
    return Vocabulary(terms=tuple(unique_terms), stopwords=frozenset(sorted(set(stops))))


def _parse_compression(raw: Any, issues: list[Issue], object_type: str = "compression") -> CompressionProfile:
    if not isinstance(raw, dict):
        issues.append(_issue(E_CONFIG_SHAPE, "compression must be an object", object_type, object_type))
        return DEFAULT_COMPRESSION
    bucket_count = raw.get("bucket_count")
    weights_raw = raw.get("bucket_weights")
    encoding = raw.get("encoding")
    bucket_count_ok = _positive_int(bucket_count)
    if not bucket_count_ok:
        issues.append(_issue(E_BUCKET_COUNT_INVALID, "bucket_count must be a positive integer", object_type, object_type, "bucket_count"))
        bucket_count = 1
    if not isinstance(weights_raw, list):
        issues.append(_issue(E_BUCKET_WEIGHT_COUNT_MISMATCH, "bucket_weights must be a list", object_type, object_type, "bucket_weights"))
        weights_raw = [1.0] * int(bucket_count)
    if len(weights_raw) != int(bucket_count):
        issues.append(_issue(E_BUCKET_WEIGHT_COUNT_MISMATCH, "bucket_weights length must equal bucket_count", object_type, object_type, "bucket_weights"))
    weights: list[float] = []
    for index, weight in enumerate(weights_raw):
        if not finite_number(weight):
            issues.append(_issue(E_NON_FINITE_WEIGHT, "bucket weight must be finite", object_type, object_type, f"bucket_weights[{index}]"))
            weights.append(0.0)
            continue
        if float(weight) < 0:
            issues.append(_issue(E_INVALID_WEIGHT, "bucket weight must be non-negative", object_type, object_type, f"bucket_weights[{index}]"))
            weights.append(0.0)
            continue
        weights.append(float(weight))
    if len(weights) != int(bucket_count):
        weights = (weights + [0.0] * int(bucket_count))[: int(bucket_count)]
    if not isinstance(encoding, str) or not encoding.strip():
        issues.append(_issue(E_INVALID_ENCODING, "encoding must be a non-empty string", object_type, object_type, "encoding"))
        encoding = DEFAULT_COMPRESSION.encoding
    if not bucket_count_ok:
        return DEFAULT_COMPRESSION
    return CompressionProfile(bucket_count=int(bucket_count), bucket_weights=tuple(weights), encoding=str(encoding))


def _parse_documents(
    raw_docs: Any,
    vocabulary: Vocabulary,
    shard_ids: set[str],
    issues: list[Issue],
) -> tuple[Document, ...]:
    if not isinstance(raw_docs, list):
        issues.append(_issue(E_CONFIG_SHAPE, "documents must be a list", "documents", "documents"))
        return ()
    documents: list[Document] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_docs):
        object_id = str(position)
        if not isinstance(raw, dict):
            issues.append(_issue(E_INVALID_DOCUMENT, "document must be an object", "document", object_id))
            continue
        doc_id = raw.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            issues.append(_issue(E_INVALID_DOCUMENT, "doc_id must be a non-empty string", "document", object_id, "doc_id"))
            continue
        object_id = doc_id
        if doc_id in seen_ids:
            issues.append(_issue(E_DOC_DUP_ID, "doc_id must be unique", "document", doc_id, "doc_id", position=position))
            continue
        version = raw.get("version")
        if not _positive_int(version):
            issues.append(_issue(E_INVALID_DOC_VERSION, "document version must be a positive integer", "document", doc_id, "version"))
            continue
        shard_id = raw.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id.strip():
            issues.append(_issue(E_INVALID_DOCUMENT, "shard_id must be a non-empty string", "document", doc_id, "shard_id"))
            continue
        if shard_id not in shard_ids:
            issues.append(_issue(E_UNKNOWN_SHARD, "document references an unknown shard", "document", doc_id, "shard_id", shard_id=shard_id))
            continue
        fields_raw = raw.get("fields")
        if not isinstance(fields_raw, dict):
            issues.append(_issue(E_INVALID_FIELD, "fields must be an object", "document", doc_id, "fields"))
            continue
        unknown_fields = sorted(set(fields_raw) - set(FIELD_NAMES))
        missing_fields = sorted(set(FIELD_NAMES) - set(fields_raw))
        if unknown_fields or missing_fields:
            issues.append(
                _issue(
                    E_INVALID_FIELD,
                    "fields must contain exactly title and body",
                    "document",
                    doc_id,
                    "fields",
                    unknown_fields=unknown_fields,
                    missing_fields=missing_fields,
                )
            )
            continue
        fields: dict[str, tuple[str, ...]] = {}
        field_ok = True
        for field_name in FIELD_NAMES:
            values = fields_raw.get(field_name)
            if not isinstance(values, list) or any(not isinstance(term, str) for term in values):
                issues.append(
                    _issue(E_INVALID_FIELD, "field terms must be a list of strings", "document", doc_id, f"fields.{field_name}")
                )
                field_ok = False
                continue
            unknown_terms = sorted(set(values) - vocabulary.term_set)
            if unknown_terms:
                issues.append(
                    _issue(
                        E_UNKNOWN_TERM,
                        "field contains terms outside the controlled vocabulary",
                        "document",
                        doc_id,
                        f"fields.{field_name}",
                        terms=unknown_terms,
                    )
                )
                field_ok = False
            fields[field_name] = tuple(values)
        if not field_ok:
            continue
        flags_raw = raw.get("field_stopwords")
        if not isinstance(flags_raw, dict):
            issues.append(_issue(E_INVALID_STOPWORD_FLAG, "field_stopwords must be an object", "document", doc_id, "field_stopwords"))
            continue
        flags: dict[str, bool] = {}
        flags_ok = True
        for field_name in FIELD_NAMES:
            if field_name not in flags_raw or not isinstance(flags_raw[field_name], bool):
                issues.append(
                    _issue(
                        E_INVALID_STOPWORD_FLAG,
                        "field stopword marker must be a boolean",
                        "document",
                        doc_id,
                        f"field_stopwords.{field_name}",
                    )
                )
                flags_ok = False
            else:
                flags[field_name] = flags_raw[field_name]
        unknown_flag_fields = sorted(set(flags_raw) - set(FIELD_NAMES))
        if unknown_flag_fields:
            issues.append(
                _issue(
                    E_INVALID_STOPWORD_FLAG,
                    "field_stopwords contains an unknown field",
                    "document",
                    doc_id,
                    "field_stopwords",
                    fields=unknown_flag_fields,
                )
            )
            flags_ok = False
        if not flags_ok:
            continue
        seen_ids.add(doc_id)
        documents.append(
            Document(
                doc_id=doc_id,
                version=version,
                shard_id=shard_id,
                fields={name: fields[name] for name in FIELD_NAMES},
                field_stopwords={name: flags[name] for name in FIELD_NAMES},
            )
        )
    return tuple(documents)


def _parse_shards(raw_layout: Any, issues: list[Issue]) -> tuple[ShardSpec, ...]:
    if not isinstance(raw_layout, dict) or not isinstance(raw_layout.get("shards"), list):
        issues.append(_issue(E_CONFIG_SHAPE, "layout.shards must be a list", "layout", "layout"))
        return ()
    shards: list[ShardSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    previous_revision: int | None = None
    for position, raw in enumerate(raw_layout["shards"]):
        if not isinstance(raw, dict):
            issues.append(_issue(E_CONFIG_SHAPE, "shard descriptor must be an object", "shard", str(position)))
            continue
        shard_id = raw.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id.strip():
            issues.append(_issue(E_CONFIG_SHAPE, "shard_id must be a non-empty string", "shard", str(position), "shard_id"))
            continue
        if shard_id in seen_ids:
            issues.append(_issue(E_DUP_SHARD_ID, "shard_id must be unique", "shard", shard_id, "shard_id"))
            continue
        port = raw.get("port")
        valid_port = isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535
        if not valid_port:
            issues.append(_issue(E_INVALID_PORT, "port must be an integer in 1..65535", "shard", shard_id, "port"))
            continue
        if port in seen_ports:
            issues.append(_issue(E_DUP_PORT, "loopback port must be unique", "shard", shard_id, "port", port=port))
            continue
        capacity = raw.get("term_bucket_capacity")
        if not _positive_int(capacity):
            issues.append(
                _issue(
                    E_NON_POSITIVE_BUCKET_CAPACITY,
                    "term_bucket_capacity must be a positive integer",
                    "shard",
                    shard_id,
                    "term_bucket_capacity",
                )
            )
            continue
        concurrency = raw.get("concurrency_limit")
        if not _positive_int(concurrency):
            issues.append(
                _issue(
                    E_NON_POSITIVE_CONCURRENCY,
                    "concurrency_limit must be a positive integer",
                    "shard",
                    shard_id,
                    "concurrency_limit",
                )
            )
            continue
        seed = raw.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            issues.append(_issue(E_CONFIG_SHAPE, "seed must be an integer", "shard", shard_id, "seed"))
            continue
        revision = raw.get("revision")
        if not _positive_int(revision):
            issues.append(_issue(E_CONFIG_SHAPE, "revision must be a positive integer", "shard", shard_id, "revision"))
            continue
        if previous_revision is not None and revision < previous_revision:
            issues.append(
                _issue(
                    E_REVISION_NON_MONOTONIC,
                    "shard revision must be monotonically non-decreasing in layout order",
                    "shard",
                    shard_id,
                    "revision",
                    previous=previous_revision,
                    current=revision,
                )
            )
            continue
        previous_revision = revision
        seen_ids.add(shard_id)
        seen_ports.add(port)
        shards.append(
            ShardSpec(
                shard_id=shard_id,
                port=port,
                term_bucket_capacity=capacity,
                concurrency_limit=concurrency,
                seed=seed,
                revision=revision,
            )
        )
    return tuple(shards)


def _parse_budget(raw: Any, issues: list[Issue]) -> BuildBudget:
    if not isinstance(raw, dict):
        issues.append(_issue(E_INVALID_BUDGET, "budget must be an object", "budget", "budget"))
        return BuildBudget(1_000_000, 1_000_000)
    global_budget = raw.get("global_term_entry_budget")
    shard_limit = raw.get("per_shard_term_entry_limit")
    if not _non_negative_int(global_budget):
        issues.append(
            _issue(
                E_INVALID_BUDGET,
                "global_term_entry_budget must be a non-negative integer",
                "budget",
                "budget",
                "global_term_entry_budget",
            )
        )
        global_budget = 1_000_000
    if not _non_negative_int(shard_limit):
        issues.append(
            _issue(
                E_INVALID_BUDGET,
                "per_shard_term_entry_limit must be a non-negative integer",
                "budget",
                "budget",
                "per_shard_term_entry_limit",
            )
        )
        shard_limit = 1_000_000
    return BuildBudget(int(global_budget), int(shard_limit))


def _parse_scoring(raw: Any, issues: list[Issue]) -> ScoringProfile:
    if raw is None:
        return ScoringProfile()
    if not isinstance(raw, dict):
        issues.append(_issue(E_INVALID_SCORING, "scoring must be an object", "scoring", "scoring"))
        return ScoringProfile()
    defaults = ScoringProfile()
    bit_width = raw.get("bit_width", defaults.bit_width)
    fraction_bits = raw.get("fraction_bits", defaults.fraction_bits)
    quantum_step = raw.get("quantum_step", defaults.quantum_step)
    title_weight = raw.get("title_weight", defaults.title_weight)
    body_weight = raw.get("body_weight", defaults.body_weight)
    saturation_k = raw.get("tf_saturation_k", defaults.tf_saturation_k)
    valid = (
        _positive_int(bit_width)
        and 4 <= bit_width <= 62
        and _non_negative_int(fraction_bits)
        and fraction_bits < bit_width
        and _positive_int(quantum_step)
        and _positive_int(title_weight)
        and _positive_int(body_weight)
        and _positive_int(saturation_k)
    )
    if not valid:
        issues.append(_issue(E_INVALID_SCORING, "scoring parameters are outside the fixed-point domain", "scoring", "scoring"))
        return defaults
    return ScoringProfile(int(bit_width), int(fraction_bits), int(quantum_step), int(title_weight), int(body_weight), int(saturation_k))


def _parse_queries(raw: Any, vocabulary: Vocabulary, issues: list[Issue]) -> tuple[Query, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        issues.append(_issue(E_CONFIG_SHAPE, "queries must be a list", "queries", "queries"))
        return ()
    queries: list[Query] = []
    seen: set[str] = set()
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            issues.append(_issue(E_CONFIG_SHAPE, "query must be an object", "query", str(position)))
            continue
        query_id = item.get("query_id", f"query-{position}")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in seen:
            issues.append(_issue(E_CONFIG_SHAPE, "query_id must be a unique non-empty string", "query", str(query_id)))
            continue
        terms_raw = item.get("terms")
        if not isinstance(terms_raw, list) or not terms_raw:
            issues.append(_issue(E_CONFIG_SHAPE, "query terms must be a non-empty list", "query", query_id, "terms"))
            continue
        terms: list[str] = []
        for term in terms_raw:
            if not isinstance(term, str):
                issues.append(_issue(E_CONFIG_SHAPE, "query term must be a string", "query", query_id, "terms"))
                continue
            if term not in vocabulary.term_set:
                issues.append(_issue(E_UNKNOWN_QUERY_TERM, "query term is outside the controlled vocabulary", "query", query_id, "terms", term=term))
                continue
            if term not in terms:
                terms.append(term)
        if not terms:
            continue
        top_k = item.get("top_k", 5)
        if not _positive_int(top_k):
            issues.append(_issue(E_CONFIG_SHAPE, "top_k must be a positive integer", "query", query_id, "top_k"))
            continue
        seen.add(query_id)
        queries.append(Query(query_id=query_id, terms=tuple(terms), top_k=top_k))
    return tuple(queries)


def parse_project(raw: Any) -> tuple[Project, list[Issue]]:
    issues: list[Issue] = []
    if not isinstance(raw, dict):
        issues.append(_issue(E_CONFIG_SHAPE, "project configuration must be an object", "project", "project"))
        raw = {}
    vocabulary = _parse_vocabulary(raw.get("vocabulary"), issues)
    compression = _parse_compression(raw.get("compression"), issues)

    revisions = raw.get("revisions")
    if isinstance(revisions, dict) and "revisions" in revisions:
        revision_list = revisions.get("revisions")
        current_revision = revisions.get("current_revision")
    elif isinstance(revisions, list):
        revision_list = revisions
        current_revision = raw.get("current_revision", compression.bucket_count)
    else:
        revision_list = None
        current_revision = raw.get("current_revision", 1)
    revision_nodes, effective_current, revision_issues = build_revision_chain(compression, revision_list, current_revision)
    issues.extend(revision_issues)
    effective_profile = next(node.profile for node in revision_nodes if node.version == effective_current)

    shards = _parse_shards(raw.get("layout", {}), issues)
    for shard in shards:
        if shard.term_bucket_capacity % effective_profile.bucket_count != 0:
            issues.append(
                _issue(
                    E_BUCKET_CAPACITY_INDIVISIBLE,
                    "term_bucket_capacity must be divisible by the effective bucket_count",
                    "shard",
                    shard.shard_id,
                    "term_bucket_capacity",
                    term_bucket_capacity=shard.term_bucket_capacity,
                    bucket_count=effective_profile.bucket_count,
                )
            )
    usable_shards = tuple(
        shard
        for shard in shards
        if not any(
            issue.object_type == "shard"
            and issue.object_id == shard.shard_id
            and issue.code
            in {
                E_NON_POSITIVE_BUCKET_CAPACITY,
                E_NON_POSITIVE_CONCURRENCY,
                E_INVALID_PORT,
                E_DUP_PORT,
                E_BUCKET_CAPACITY_INDIVISIBLE,
            }
            for issue in issues
        )
    )

    documents = _parse_documents(raw.get("documents", []), vocabulary, {s.shard_id for s in usable_shards}, issues)
    budget = _parse_budget(raw.get("budget"), issues)
    scoring = _parse_scoring(raw.get("scoring"), issues)
    queries = _parse_queries(raw.get("queries"), vocabulary, issues)

    project = Project(
        vocabulary=vocabulary,
        documents=documents,
        shards=usable_shards,
        compression=effective_profile,
        budget=budget,
        revisions=revision_nodes,
        current_revision=effective_current,
        scoring=scoring,
        queries=queries,
        raw=raw,
    )
    return project, stable_sorted_issues([issue.to_dict() for issue in issues])


def _load_component_directory(path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    component_files = {
        "vocabulary.json": "vocabulary",
        "documents.json": "documents",
        "layout.json": "layout",
        "compression.json": "compression",
        "budget.json": "budget",
        "revisions.json": "revisions",
    }
    for filename, key in component_files.items():
        candidate = path / filename
        if candidate.exists():
            value = read_json(candidate)
            if key == "documents" and isinstance(value, dict) and isinstance(value.get("documents"), list):
                value = value["documents"]
            raw[key] = value
    queries_path = path / "queries.json"
    if queries_path.exists():
        query_bundle = read_json(queries_path)
        if isinstance(query_bundle, dict):
            raw["scoring"] = query_bundle.get("scoring", raw.get("scoring"))
            raw["queries"] = query_bundle.get("queries", raw.get("queries"))
    return raw


def load_project(path: str | Path) -> tuple[Project, list[dict[str, Any]]]:
    path = Path(path)
    try:
        if path.is_dir():
            bundle = path / "project.json"
            raw = read_json(bundle) if bundle.exists() else _load_component_directory(path)
        else:
            raw = read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        issue = _issue(E_CONFIG_READ, "unable to read project JSON", "project", str(path), str(exc))
        empty_project, parse_issues = parse_project({})
        return empty_project, stable_sorted_issues([issue.to_dict()] + parse_issues)
    return parse_project(raw)


def project_config_hash(project: Project) -> str:
    return fingerprint(project.config_payload())
