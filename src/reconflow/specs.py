from __future__ import annotations

import json
import re
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import SpecError
from .fixedpoint import BP_SCALE

DOMAIN_FORMAT = "reconflow.domains/1"
CONFIG_FORMAT = "reconflow.config/1"
BATCH_FORMAT = "reconflow.batch/1"
LEDGER_FORMAT_VERSION = 1

MIN_THRESHOLD_BP = 5_000
MAX_THRESHOLD_BP = 9_999
MIN_SHARD_SIZE = 1
MAX_SHARD_SIZE = 1_024
MAX_INT = (1 << 63) - 1

_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")


@dataclass(frozen=True)
class SpecBundle:
    domains: list[dict[str, Any]]
    config: dict[str, Any]
    batches: list[dict[str, Any]]
    root: Path | None = None

    @property
    def domain_ids(self) -> list[str]:
        return sorted(str(domain["domain_id"]) for domain in self.domains)

    def domain(self, domain_id: str) -> dict[str, Any]:
        for domain in self.domains:
            if domain["domain_id"] == domain_id:
                return domain
        raise KeyError(domain_id)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _line_at(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _field_line(text: str, start: int, end: int, field: str) -> int:
    segment = text[start:end]
    match = re.search(r'"' + re.escape(field) + r'"\s*:', segment)
    if match:
        return _line_at(text, start + match.start())
    return _line_at(text, start)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read spec file: {exc}", path=str(path), line=1, field="file") from exc


def load_json(path: str | Path) -> Any:
    path = Path(path)
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON: {exc.msg}",
            path=str(path),
            line=exc.lineno,
            field="$",
        ) from exc


def load_ndjson(path: str | Path) -> list[Any]:
    path = Path(path)
    text = _read_text(path)
    records: list[Any] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise SpecError("blank NDJSON record", path=str(path), line=line_no, field="$")
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"invalid NDJSON: {exc.msg}",
                path=str(path),
                line=line_no,
                field="$",
            ) from exc
    return records


def _json_array_objects(text: str, key: str | None = None) -> list[tuple[dict[str, Any], int, int]]:
    if key is None:
        bracket = text.find("[")
        if bracket < 0:
            raise SpecError("JSON must contain an array", line=1, field="$")
    else:
        key_match = re.search(r'"' + re.escape(key) + r'"\s*:', text)
        if not key_match:
            raise SpecError(f"missing {key} array", line=1, field=key)
        bracket = text.find("[", key_match.end())
        if bracket < 0:
            raise SpecError(f"{key} must be an array", line=1, field=key)

    decoder = JSONDecoder()
    pos = bracket + 1
    results: list[tuple[dict[str, Any], int, int]] = []
    while True:
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text):
            raise SpecError("unterminated JSON array", line=_line_at(text, bracket), field=key or "$")
        if text[pos] == "]":
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            raise SpecError(f"invalid JSON record: {exc.msg}", line=exc.lineno, field="$") from exc
        if not isinstance(obj, dict):
            raise SpecError("JSON record must be an object", line=_line_at(text, pos), field="$")
        results.append((obj, pos, end))
        pos = end
    return results


def load_domain_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".ndjson":
        raw_records = load_ndjson(path)
        records: list[dict[str, Any]] = []
        line_by_id: dict[int, int] = {}
        for line, record in enumerate(raw_records, start=1):
            if not isinstance(record, dict):
                raise SpecError("domain record must be an object", path=str(path), line=line, field="$")
            records.append(record)
            line_by_id[id(record)] = line
        validate_domains(
            records,
            source_path=str(path),
            line_for=lambda record, field: line_by_id[id(record)],
        )
        return records

    text = _read_text(path)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc.msg}", path=str(path), line=exc.lineno, field="$") from exc
    if isinstance(parsed, dict) and "domains" in parsed:
        if not isinstance(parsed["domains"], list):
            raise SpecError("domains must be an array", path=str(path), line=1, field="domains")
        spans = _json_array_objects(text, "domains")
        records = [item for item, _, _ in spans]

        def line_for(record: dict[str, Any], field: str) -> int:
            index = records.index(record)
            _, start, end = spans[index]
            return _field_line(text, start, end, field)

    elif isinstance(parsed, list):
        spans = _json_array_objects(text)
        records = [item for item, _, _ in spans]

        def line_for(record: dict[str, Any], field: str) -> int:
            index = records.index(record)
            _, start, end = spans[index]
            return _field_line(text, start, end, field)

    else:
        raise SpecError("domains JSON must be an array or contain domains", path=str(path), line=1, field="domains")
    validate_domains(records, source_path=str(path), line_for=line_for)
    return records


def _fail(
    message: str,
    *,
    path: str | None,
    line: int,
    field: str,
    value: Any = None,
) -> None:
    raise SpecError(message, path=path, line=line, field=field, value=value)


def _require_int(
    record: dict[str, Any],
    field: str,
    *,
    path: str | None,
    line: int,
    minimum: int | None = 0,
    maximum: int | None = MAX_INT,
) -> int:
    if field not in record:
        _fail(f"missing integer field {field!r}", path=path, line=line, field=field)
    value = record[field]
    if not _is_int(value):
        _fail(f"{field} must be an integer", path=path, line=line, field=field, value=value)
    if minimum is not None and value < minimum:
        _fail(f"{field} must be >= {minimum}", path=path, line=line, field=field, value=value)
    if maximum is not None and value > maximum:
        _fail(f"{field} must be <= {maximum}", path=path, line=line, field=field, value=value)
    return value


def validate_domains(
    records: Iterable[dict[str, Any]],
    *,
    source_path: str | None = None,
    line_for: Callable[[dict[str, Any], str], int] | None = None,
) -> None:
    seen: set[str] = set()
    records = list(records)
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SpecError("domain record must be an object", path=source_path, line=index + 1, field="$")

        def line(field: str) -> int:
            if line_for is not None:
                return int(line_for(record, field))
            return int(record.get("__line__", index + 1))

        domain_id = record.get("domain_id")
        if not isinstance(domain_id, str) or not _ID_RE.match(domain_id):
            _fail(
                "domain_id must match ^[a-z][a-z0-9_-]{0,47}$",
                path=source_path,
                line=line("domain_id"),
                field="domain_id",
                value=domain_id,
            )
        if domain_id in seen:
            _fail("duplicate domain_id", path=source_path, line=line("domain_id"), field="domain_id", value=domain_id)
        seen.add(domain_id)

        _require_int(record, "seed", path=source_path, line=line("seed"), minimum=0)
        _require_int(record, "doc_count", path=source_path, line=line("doc_count"), minimum=0)
        _require_int(record, "batch_size", path=source_path, line=line("batch_size"), minimum=1, maximum=10_000)
        _require_int(record, "vocab_size", path=source_path, line=line("vocab_size"), minimum=1, maximum=256)
        _require_int(record, "motif_count", path=source_path, line=line("motif_count"), minimum=1, maximum=64)
        _require_int(record, "duplicate_rate_bp", path=source_path, line=line("duplicate_rate_bp"), minimum=0, maximum=BP_SCALE)
        _require_int(record, "missing_rate_bp", path=source_path, line=line("missing_rate_bp"), minimum=0, maximum=BP_SCALE)

        if record["batch_size"] > max(1, record["doc_count"]):
            # A final partial batch is allowed, but a declared positive stream
            # should have a batch size no larger than its first batch.
            if record["doc_count"] != 0 and record["batch_size"] > record["doc_count"]:
                _fail(
                    "batch_size must not exceed doc_count when doc_count > 0",
                    path=source_path,
                    line=line("batch_size"),
                    field="batch_size",
                    value=record["batch_size"],
                )


def validate_config(config: dict[str, Any], *, source_path: str | None = None, text: str | None = None) -> None:
    if not isinstance(config, dict):
        raise SpecError("config must be an object", path=source_path, line=1, field="$")

    def line(field: str) -> int:
        if text:
            match = re.search(r'"' + re.escape(field) + r'"\s*:', text)
            if match:
                return _line_at(text, match.start())
        return 1

    format_version = config.get("format_version")
    if format_version != 1:
        _fail("format_version must be integer 1", path=source_path, line=line("format_version"), field="format_version", value=format_version)
    _require_int(config, "config_version", path=source_path, line=line("config_version"), minimum=1)
    _require_int(config, "sampling_budget_points", path=source_path, line=line("sampling_budget_points"), minimum=1)
    _require_int(config, "global_sampling_point_budget", path=source_path, line=line("global_sampling_point_budget"), minimum=1)
    _require_int(config, "domain_sampling_point_quota", path=source_path, line=line("domain_sampling_point_quota"), minimum=1)

    threshold = config.get("similarity_threshold_bp")
    if not _is_int(threshold) or not (MIN_THRESHOLD_BP <= threshold <= MAX_THRESHOLD_BP):
        _fail(
            f"similarity_threshold_bp must be an integer in [{MIN_THRESHOLD_BP},{MAX_THRESHOLD_BP}]",
            path=source_path,
            line=line("similarity_threshold_bp"),
            field="similarity_threshold_bp",
            value=threshold,
        )
    shard_size = config.get("shard_size")
    if not _is_int(shard_size) or not (MIN_SHARD_SIZE <= shard_size <= MAX_SHARD_SIZE):
        _fail(
            f"shard_size must be an integer in [{MIN_SHARD_SIZE},{MAX_SHARD_SIZE}]",
            path=source_path,
            line=line("shard_size"),
            field="shard_size",
            value=shard_size,
        )

    weights = config.get("weights_bp")
    if not isinstance(weights, dict) or not weights:
        _fail("weights_bp must be a non-empty object", path=source_path, line=line("weights_bp"), field="weights_bp")
    positive = 0
    for domain_id, weight in weights.items():
        if not isinstance(domain_id, str) or not _ID_RE.match(domain_id):
            _fail("weights_bp contains invalid domain_id", path=source_path, line=line("weights_bp"), field=f"weights_bp.{domain_id}")
        if not _is_int(weight) or not (0 <= weight <= BP_SCALE):
            _fail(
                "sampling weight must be a ten-thousandths integer in [0,10000]",
                path=source_path,
                line=line("weights_bp"),
                field=f"weights_bp.{domain_id}",
                value=weight,
            )
        positive += int(weight)
    if positive <= 0:
        _fail("at least one sampling weight must be positive", path=source_path, line=line("weights_bp"), field="weights_bp")

    if config.get("bp_scale") != BP_SCALE:
        _fail("bp_scale must be 10000", path=source_path, line=line("bp_scale"), field="bp_scale", value=config.get("bp_scale"))
    if config.get("rounding_rule") != "floor_nonnegative_then_descending_remainder_then_id":
        _fail("unsupported rounding_rule", path=source_path, line=line("rounding_rule"), field="rounding_rule")
    if config.get("saturation_rule") != "saturate_to_int64_max":
        _fail("unsupported saturation_rule", path=source_path, line=line("saturation_rule"), field="saturation_rule")

    stage_params = config.get("stage_params")
    if not isinstance(stage_params, dict):
        _fail("stage_params must be an object", path=source_path, line=line("stage_params"), field="stage_params")
    for stage_name in ("weighted_sampling", "dedup", "normalization"):
        if not isinstance(stage_params.get(stage_name), dict):
            _fail(f"stage_params.{stage_name} must be an object", path=source_path, line=line("stage_params"), field=f"stage_params.{stage_name}")
    dedup_params = stage_params["dedup"]
    for field_name, minimum, maximum in (
        ("min_shared_terms", 1, 64),
        ("boundary_band_bp", 0, 500),
        ("rpc_timeout_ms", 10, 30_000),
    ):
        _require_int(
            dedup_params,
            field_name,
            path=source_path,
            line=line("stage_params"),
            minimum=minimum,
            maximum=maximum,
        )


def validate_doc(doc: Any, *, path: str | None = None, line: int = 1, domain: dict[str, Any] | None = None) -> None:
    if not isinstance(doc, dict):
        _fail("doc must be an object", path=path, line=line, field="docs")
    doc_seq = doc.get("doc_seq")
    if not _is_int(doc_seq) or doc_seq < 0:
        _fail("doc_seq must be a non-negative integer", path=path, line=line, field="docs.doc_seq", value=doc_seq)
    if not isinstance(doc.get("doc_id"), str) or not doc["doc_id"]:
        _fail("doc_id must be a non-empty string", path=path, line=line, field="docs.doc_id")
    counts = doc.get("term_counts")
    bitmap = doc.get("missing_bitmap")
    if not isinstance(counts, list) or not counts or not all(_is_int(x) and x >= 0 for x in counts):
        _fail("term_counts must be a non-empty array of non-negative integers", path=path, line=line, field="docs.term_counts")
    if not isinstance(bitmap, list) or not all(_is_int(x) and x in (0, 1) for x in bitmap):
        _fail("missing_bitmap must be an array of 0/1 integers", path=path, line=line, field="docs.missing_bitmap")
    if len(bitmap) != len(counts):
        _fail(
            f"missing_bitmap length {len(bitmap)} != term_counts length {len(counts)}",
            path=path,
            line=line,
            field="docs.missing_bitmap",
            value={"bitmap_length": len(bitmap), "term_count_length": len(counts)},
        )
    length = doc.get("length")
    if not _is_int(length) or length < 0:
        _fail("length must be a non-negative integer", path=path, line=line, field="docs.length", value=length)
    if length != sum(counts):
        _fail(
            f"length {length} != sum(term_counts) {sum(counts)}",
            path=path,
            line=line,
            field="docs.length",
            value={"length": length, "term_count_sum": sum(counts)},
        )
    if domain is not None and len(counts) != domain["vocab_size"]:
        _fail(
            f"term_counts length {len(counts)} != domain vocab_size {domain['vocab_size']}",
            path=path,
            line=line,
            field="docs.term_counts",
        )


def validate_batch(
    batch: dict[str, Any],
    *,
    line: int = 1,
    path: str | None = None,
    domains_by_id: dict[str, dict[str, Any]] | None = None,
    verify_digest: Callable[[dict[str, Any]], str] | None = None,
) -> None:
    if not isinstance(batch, dict):
        _fail("batch must be an object", path=path, line=line, field="$")
    for field_name in ("batch_id", "domain_id"):
        if not isinstance(batch.get(field_name), str) or not batch[field_name]:
            _fail(f"{field_name} must be a non-empty string", path=path, line=line, field=field_name)
    _require_int(batch, "batch_seq", path=path, line=line, minimum=0)
    if batch.get("format_version") != 1:
        _fail("format_version must be integer 1", path=path, line=line, field="format_version")
    domain = None
    if domains_by_id is not None:
        domain = domains_by_id.get(batch["domain_id"])
        if domain is None:
            _fail("batch references unknown domain_id", path=path, line=line, field="domain_id", value=batch["domain_id"])
        expected_batch_id = f"{batch['domain_id']}#b{int(batch['batch_seq']):04d}"
        if batch["batch_id"] != expected_batch_id:
            _fail("batch_id does not match domain_id/batch_seq", path=path, line=line, field="batch_id", value=batch["batch_id"])
    docs = batch.get("docs")
    if not isinstance(docs, list) or not docs:
        _fail("docs must be a non-empty array", path=path, line=line, field="docs")
    for offset, doc in enumerate(docs):
        if not isinstance(doc, dict):
            _fail("doc must be an object", path=path, line=line, field=f"docs[{offset}]")
        validate_doc(doc, path=path, line=line, domain=domain)
        expected_seq = int(batch["batch_seq"]) * int(domain["batch_size"]) + offset if domain else doc.get("doc_seq") - offset
        if doc.get("doc_seq") != expected_seq:
            _fail(
                "doc_seq does not match batch_seq/offset",
                path=path,
                line=line,
                field=f"docs[{offset}].doc_seq",
                value={"expected": expected_seq, "actual": doc.get("doc_seq")},
            )
        expected_batch = int(doc["doc_seq"]) // int(domain["batch_size"]) if domain else batch["batch_seq"]
        if expected_batch != batch["batch_seq"]:
            _fail("doc_seq belongs to another batch", path=path, line=line, field=f"docs[{offset}].doc_seq")
        if verify_digest is not None and doc.get("content_digest") != verify_digest(doc):
            _fail("doc content_digest mismatch", path=path, line=line, field=f"docs[{offset}].content_digest")


def load_bundle(root: str | Path, *, verify_digests: bool = True) -> SpecBundle:
    root = Path(root)
    domains_path = root / "specs" / "domains.json"
    config_path = root / "config" / "config.json"
    batches_path = root / "data" / "batches.ndjson"
    domains = sorted(load_domain_records(domains_path), key=lambda domain: str(domain["domain_id"]))
    config_text = _read_text(config_path)
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid config JSON: {exc.msg}", path=str(config_path), line=exc.lineno, field="$") from exc
    validate_config(config, source_path=str(config_path), text=config_text)

    domain_ids = [d["domain_id"] for d in domains]
    if set(config["weights_bp"]) != set(domain_ids):
        raise SpecError(
            "weights_bp domain set must exactly match domains",
            path=str(config_path),
            line=1,
            field="weights_bp",
            value={"domains": sorted(domain_ids), "weights": sorted(config["weights_bp"])},
        )

    domains_by_id = {d["domain_id"]: d for d in domains}
    raw_batches = load_ndjson(batches_path)
    from .synthetic import doc_content_digest

    seen_batch_ids: set[str] = set()
    batches: list[dict[str, Any]] = []
    for line_no, batch in enumerate(raw_batches, start=1):
        if not isinstance(batch, dict):
            raise SpecError("batch record must be an object", path=str(batches_path), line=line_no, field="$")
        validate_batch(
            batch,
            line=line_no,
            path=str(batches_path),
            domains_by_id=domains_by_id,
            verify_digest=doc_content_digest if verify_digests else None,
        )
        if batch["batch_id"] in seen_batch_ids:
            raise SpecError("duplicate batch_id", path=str(batches_path), line=line_no, field="batch_id")
        seen_batch_ids.add(batch["batch_id"])
        batches.append(batch)

    expected_docs = sum(int(d["doc_count"]) for d in domains)
    actual_docs = sum(len(b["docs"]) for b in batches)
    if actual_docs != expected_docs:
        raise SpecError(
            "batch document count does not match domain doc_count sum",
            path=str(batches_path),
            line=1,
            field="docs",
            value={"expected": expected_docs, "actual": actual_docs},
        )
    batches.sort(key=lambda batch: (int(batch["batch_seq"]), str(batch["domain_id"]), str(batch["batch_id"])))
    return SpecBundle(domains=domains, config=config, batches=batches, root=root)
