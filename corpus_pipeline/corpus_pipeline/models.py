from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from typing import Any, Iterable, Mapping

from .errors import Code, fail
from .fixed import (
    ALLOWED_DECIMAL_PLACES,
    ALLOWED_QUANTIZATION_TIERS,
    parse_decimal,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_int(value: Any, *, field_name: str, code: Code = Code.INVALID_SCHEMA, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(code, f"{field_name} must be an integer", field=field_name, value=value)
    if minimum is not None and value < minimum:
        fail(code, f"{field_name} must be >= {minimum}", field=field_name, value=value)
    return value


def require_text(value: Any, *, field_name: str, code: Code = Code.INVALID_SCHEMA) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(code, f"{field_name} must be a non-empty string", field=field_name, value=value)
    return value.strip()


@dataclass(frozen=True, slots=True)
class DomainSpec:
    id: str
    weight: Decimal
    seed: int
    batch_size: int
    vocab_size: int = 64
    length_min: int = 24
    length_max: int = 72
    duplicate_every: int = 5

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], default_seed: int) -> "DomainSpec":
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_SCHEMA, "domain declaration must be an object", value=repr(raw))
        domain_id = require_text(raw.get("id"), field_name="domains[].id")
        weight = parse_decimal(raw.get("weight", 1), field_name=f"domains[{domain_id}].weight")
        if not weight.is_finite():
            fail(Code.INVALID_WEIGHT, "domain weight must be finite", domain=domain_id)
        if weight < 0:
            fail(Code.INVALID_WEIGHT, "domain weight must be non-negative", domain=domain_id, value=str(weight))
        seed = require_int(raw.get("seed", default_seed), field_name=f"domains[{domain_id}].seed")
        batch_size = require_int(raw.get("batch_size", 16), field_name=f"domains[{domain_id}].batch_size", minimum=1)
        vocab_size = require_int(raw.get("vocab_size", 64), field_name=f"domains[{domain_id}].vocab_size", minimum=2)
        length_min = require_int(raw.get("length_min", 16), field_name=f"domains[{domain_id}].length_min", minimum=1)
        length_max = require_int(raw.get("length_max", 64), field_name=f"domains[{domain_id}].length_max", minimum=1)
        if length_max < length_min:
            fail(Code.INVALID_SCHEMA, "length_max must be >= length_min", domain=domain_id)
        duplicate_every = require_int(
            raw.get("duplicate_every", 0),
            field_name=f"domains[{domain_id}].duplicate_every",
            minimum=0,
        )
        if vocab_size < 3:
            fail(Code.INVALID_SCHEMA, "vocab_size must be at least 3", domain=domain_id)
        return cls(
            id=domain_id,
            weight=weight,
            seed=seed,
            batch_size=batch_size,
            vocab_size=vocab_size,
            length_min=length_min,
            length_max=length_max,
            duplicate_every=duplicate_every,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "weight": str(self.weight),
            "seed": self.seed,
            "batch_size": self.batch_size,
            "vocab_size": self.vocab_size,
            "length_min": self.length_min,
            "length_max": self.length_max,
            "duplicate_every": self.duplicate_every,
        }

    def fingerprint_payload(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True, slots=True)
class PrecisionSpec:
    decimal_places: int = 6
    quantization_tier: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PrecisionSpec":
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_SCHEMA, "precision must be an object")
        decimal_places = require_int(
            raw.get("decimal_places", 6), field_name="precision.decimal_places", code=Code.INVALID_PRECISION
        )
        quantization_tier = require_int(
            raw.get("quantization_tier", 1),
            field_name="precision.quantization_tier",
            code=Code.INVALID_QUANTIZATION_TIER,
        )
        if decimal_places not in ALLOWED_DECIMAL_PLACES:
            fail(
                Code.INVALID_PRECISION,
                "decimal_places is outside the allowed enumeration",
                value=decimal_places,
                allowed=list(ALLOWED_DECIMAL_PLACES),
            )
        if quantization_tier not in ALLOWED_QUANTIZATION_TIERS:
            fail(
                Code.INVALID_QUANTIZATION_TIER,
                "quantization_tier is outside the allowed enumeration",
                value=quantization_tier,
                allowed=list(ALLOWED_QUANTIZATION_TIERS),
            )
        return cls(decimal_places=decimal_places, quantization_tier=quantization_tier)

    @property
    def scale(self) -> int:
        return 10 ** self.decimal_places

    def higher_precision(self) -> "PrecisionSpec":
        if self.decimal_places >= max(ALLOWED_DECIMAL_PLACES):
            # The comparison path is internal; it may exceed the user-facing enum.
            return PrecisionSpec(decimal_places=self.decimal_places + 1, quantization_tier=self.quantization_tier)
        return PrecisionSpec(decimal_places=self.decimal_places + 1, quantization_tier=self.quantization_tier)

    def to_dict(self) -> dict[str, Any]:
        return {"decimal_places": self.decimal_places, "quantization_tier": self.quantization_tier}


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    budget: int = 64
    per_domain_quota: int = 32
    shard_size: int = 8
    max_dedup_comparisons: int = 1024

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SamplingSpec":
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_SCHEMA, "sampling must be an object")
        budget = require_int(raw.get("budget", 64), field_name="sampling.budget", code=Code.INVALID_BUDGET, minimum=0)
        per_domain_quota = require_int(
            raw.get("per_domain_quota", 32),
            field_name="sampling.per_domain_quota",
            code=Code.INVALID_BUDGET,
            minimum=1,
        )
        shard_size = require_int(
            raw.get("shard_size", 8), field_name="sampling.shard_size", code=Code.INVALID_SHARD_SIZE, minimum=1
        )
        max_dedup_comparisons = require_int(
            raw.get("max_dedup_comparisons", 1024),
            field_name="sampling.max_dedup_comparisons",
            code=Code.INVALID_BUDGET,
            minimum=0,
        )
        return cls(
            budget=budget,
            per_domain_quota=per_domain_quota,
            shard_size=shard_size,
            max_dedup_comparisons=max_dedup_comparisons,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DedupSpec:
    similarity_threshold_permille: int = 700
    boundary_rule: str = "include"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DedupSpec":
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_SCHEMA, "dedup must be an object")
        threshold = require_int(
            raw.get("similarity_threshold_permille", 700),
            field_name="dedup.similarity_threshold_permille",
            code=Code.THRESHOLD_OUT_OF_RANGE,
        )
        if threshold < 0 or threshold > 1000:
            fail(
                Code.THRESHOLD_OUT_OF_RANGE,
                "similarity threshold must be in the inclusive range 0..1000 permille",
                value=threshold,
            )
        boundary_rule = raw.get("boundary_rule", "include")
        if boundary_rule not in {"include", "exclude"}:
            fail(
                Code.INVALID_SCHEMA,
                "boundary_rule must be 'include' or 'exclude'",
                value=boundary_rule,
            )
        return cls(similarity_threshold_permille=threshold, boundary_rule=str(boundary_rule))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToleranceSpec:
    sampling_docs: int = 0
    similarity_permille: int = 1
    dedup_docs: int = 0
    composition_fixed: int = 10

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ToleranceSpec":
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_TOLERANCE, "tolerance must be an object")
        values: dict[str, int] = {}
        for key, default in {
            "sampling_docs": 0,
            "similarity_permille": 1,
            "dedup_docs": 0,
            "composition_fixed": 10,
        }.items():
            values[key] = require_int(
                raw.get(key, default), field_name=f"tolerance.{key}", code=Code.INVALID_TOLERANCE, minimum=0
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    version: int
    seed: int
    precision: PrecisionSpec
    sampling: SamplingSpec
    dedup: DedupSpec
    tolerance: ToleranceSpec
    domains: tuple[DomainSpec, ...]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PipelineConfig":
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_SCHEMA, "configuration root must be an object")
        schema_version = require_int(raw.get("schema_version", 1), field_name="schema_version", minimum=1)
        version = require_int(raw.get("version", 1), field_name="version", minimum=1)
        seed = require_int(raw.get("seed", 20260101), field_name="seed")
        precision = PrecisionSpec.from_dict(raw.get("precision", {}))
        sampling = SamplingSpec.from_dict(raw.get("sampling", {}))
        dedup = DedupSpec.from_dict(raw.get("dedup", {}))
        tolerance = ToleranceSpec.from_dict(raw.get("tolerance"))

        raw_domains = raw.get("domains")
        if not isinstance(raw_domains, list) or not raw_domains:
            fail(Code.INVALID_SCHEMA, "domains must be a non-empty list")
        domains: list[DomainSpec] = []
        seen: set[str] = set()
        for item in raw_domains:
            domain = DomainSpec.from_dict(item, seed)
            if domain.id in seen:
                fail(Code.DUPLICATE_DOMAIN, "domain identifiers must be unique", domain=domain.id)
            seen.add(domain.id)
            domains.append(domain)
        weight_sum = sum((domain.weight for domain in domains), Decimal(0))
        if weight_sum <= 0:
            fail(Code.WEIGHT_SUM_ZERO, "sum of domain weights must be greater than zero", weight_sum=str(weight_sum))
        return cls(
            schema_version=schema_version,
            version=version,
            seed=seed,
            precision=precision,
            sampling=sampling,
            dedup=dedup,
            tolerance=tolerance,
            domains=tuple(domains),
        )

    @property
    def domain_ids(self) -> tuple[str, ...]:
        return tuple(domain.id for domain in self.domains)

    def domain(self, domain_id: str) -> DomainSpec:
        for domain in self.domains:
            if domain.id == domain_id:
                return domain
        fail(Code.UNKNOWN_DOMAIN, "unknown domain", domain=domain_id)

    def replace_domain_weight(self, domain_id: str, weight: Any) -> "PipelineConfig":
        parsed = parse_decimal(weight, field_name=f"edit.set_weight.{domain_id}")
        if not parsed.is_finite() or parsed < 0:
            fail(Code.INVALID_WEIGHT, "domain weight must be finite and non-negative", domain=domain_id)
        new_domains = tuple(
            replace(domain, weight=parsed) if domain.id == domain_id else domain for domain in self.domains
        )
        if sum((domain.weight for domain in new_domains), Decimal(0)) <= 0:
            fail(Code.WEIGHT_SUM_ZERO, "sum of domain weights must be greater than zero")
        return replace(self, domains=new_domains)

    def with_precision(self, precision: PrecisionSpec) -> "PipelineConfig":
        return replace(self, precision=precision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "seed": self.seed,
            "precision": self.precision.to_dict(),
            "sampling": self.sampling.to_dict(),
            "dedup": self.dedup.to_dict(),
            "tolerance": self.tolerance.to_dict(),
            "domains": [domain.to_dict() for domain in self.domains],
        }

    def config_fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "seed": self.seed,
            "precision": self.precision.to_dict(),
            "sampling": self.sampling.to_dict(),
            "dedup": self.dedup.to_dict(),
            "tolerance": self.tolerance.to_dict(),
            "domains": [domain.fingerprint_payload() for domain in self.domains],
        }


@dataclass(frozen=True, slots=True)
class Document:
    domain: str
    doc_id: str
    term_counts: Mapping[str, int]
    length: int
    subwords: frozenset[str]
    batch_index: int = 0
    generator_seed: int = 0
    source: str = "generated"
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.length <= 0:
            fail(Code.INVALID_DOCUMENT, "document length must be positive", domain=self.domain, doc_id=self.doc_id)
        total = sum(self.term_counts.values())
        if total != self.length:
            fail(
                Code.INVALID_DOCUMENT,
                "sum of term counts must equal document length",
                domain=self.domain,
                doc_id=self.doc_id,
                term_count_sum=total,
                length=self.length,
            )
        if any(count <= 0 for count in self.term_counts.values()):
            fail(Code.INVALID_DOCUMENT, "term counts must be positive", domain=self.domain, doc_id=self.doc_id)

    @property
    def effective_terms(self) -> int:
        return self.length

    @property
    def unique_terms(self) -> int:
        return len(self.term_counts)

    @property
    def frequency_fingerprint(self) -> str:
        return content_hash(
            {
                "domain": self.domain,
                "doc_id": self.doc_id,
                "term_counts": dict(self.term_counts),
                "length": self.length,
                "subwords": sorted(self.subwords),
                "batch_index": self.batch_index,
                "generator_seed": self.generator_seed,
            }
        )

    def with_content_hash(self) -> "Document":
        fingerprint = self.frequency_fingerprint
        if fingerprint == self.content_hash:
            return self
        return replace(self, content_hash=fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "doc_id": self.doc_id,
            "term_counts": dict(sorted(self.term_counts.items())),
            "length": self.length,
            "subwords": sorted(self.subwords),
            "batch_index": self.batch_index,
            "generator_seed": self.generator_seed,
            "source": self.source,
            "content_hash": self.content_hash or self.frequency_fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, known_domains: set[str] | None = None) -> "Document":
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_DOCUMENT, "document record must be an object")
        domain = require_text(raw.get("domain"), field_name="document.domain", code=Code.INVALID_DOCUMENT)
        if known_domains is not None and domain not in known_domains:
            fail(Code.UNKNOWN_DOMAIN, "document references an unknown domain", domain=domain)
        doc_id = require_text(raw.get("doc_id"), field_name="document.doc_id", code=Code.INVALID_DOCUMENT)
        raw_counts = raw.get("term_counts", raw.get("terms"))
        if not isinstance(raw_counts, Mapping) or not raw_counts:
            fail(Code.INVALID_DOCUMENT, "term_counts must be a non-empty object", domain=domain, doc_id=doc_id)
        counts: dict[str, int] = {}
        for term, count in raw_counts.items():
            term_text = require_text(term, field_name="document.term_counts key", code=Code.INVALID_DOCUMENT)
            counts[term_text] = require_int(
                count, field_name=f"document.term_counts[{term_text}]", code=Code.INVALID_DOCUMENT, minimum=1
            )
        length_raw = raw.get("length")
        if length_raw is None:
            length = sum(counts.values())
        else:
            length = require_int(length_raw, field_name="document.length", code=Code.INVALID_DOCUMENT, minimum=1)
        if raw.get("subwords") is None:
            subwords = frozenset(counts)
        else:
            if not isinstance(raw.get("subwords"), (list, tuple, set, frozenset)):
                fail(Code.INVALID_DOCUMENT, "subwords must be a list", domain=domain, doc_id=doc_id)
            subwords = frozenset(
                require_text(item, field_name="document.subwords item", code=Code.INVALID_DOCUMENT)
                for item in raw["subwords"]
            )
            if not subwords:
                fail(Code.INVALID_DOCUMENT, "subwords must be non-empty", domain=domain, doc_id=doc_id)
        batch_index = require_int(
            raw.get("batch_index", 0), field_name="document.batch_index", code=Code.INVALID_DOCUMENT, minimum=0
        )
        generator_seed = require_int(
            raw.get("generator_seed", 0), field_name="document.generator_seed", code=Code.INVALID_DOCUMENT
        )
        source = require_text(raw.get("source", "loaded"), field_name="document.source", code=Code.INVALID_DOCUMENT)
        doc = cls(
            domain=domain,
            doc_id=doc_id,
            term_counts=counts,
            length=length,
            subwords=subwords,
            batch_index=batch_index,
            generator_seed=generator_seed,
            source=source,
            content_hash=str(raw.get("content_hash", "")),
        )
        expected = doc.frequency_fingerprint
        provided = str(raw.get("content_hash", ""))
        if provided and provided != expected:
            fail(
                Code.INVALID_DOCUMENT,
                "document content hash does not match content",
                domain=domain,
                doc_id=doc_id,
            )
        return doc.with_content_hash()


@dataclass(frozen=True, slots=True)
class EditCommand:
    seq: int
    op: str
    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EditCommand":
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_EDIT, "edit command must be an object")
        seq = require_int(raw.get("seq"), field_name="edit.seq", code=Code.INVALID_EDIT, minimum=1)
        op = require_text(raw.get("op"), field_name="edit.op", code=Code.INVALID_EDIT)
        if op not in {"set_weight", "replace_document", "append_batch"}:
            fail(Code.INVALID_EDIT, "unsupported edit operation", op=op, seq=seq)
        return cls(seq=seq, op=op, payload=dict(raw))

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "op": self.op, **dict(self.payload)}


@dataclass(frozen=True, slots=True)
class Receipt:
    receipt_id: str
    domain: str
    shard_id: str
    parameter_version: int
    elapsed_ms: int
    covers: tuple[str, ...]
    source_file: str
    sample_count: int | None = None
    sample_fingerprint: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, source_file: str, known_domains: set[str]) -> "Receipt":
        if not isinstance(raw, Mapping):
            fail(Code.INVALID_RECEIPT, "receipt must be an object", file=source_file)
        receipt_id = require_text(raw.get("receipt_id", raw.get("id")), field_name="receipt_id", code=Code.INVALID_RECEIPT)
        domain = require_text(raw.get("domain"), field_name="receipt.domain", code=Code.INVALID_RECEIPT)
        if domain not in known_domains:
            fail(Code.UNKNOWN_DOMAIN, "receipt references an unknown domain", domain=domain, file=source_file)
        shard_id = require_text(raw.get("shard_id", raw.get("shard")), field_name="receipt.shard_id", code=Code.INVALID_RECEIPT)
        parameter_version = require_int(
            raw.get("parameter_version", raw.get("version")),
            field_name="receipt.parameter_version",
            code=Code.INVALID_RECEIPT,
            minimum=1,
        )
        elapsed_ms = require_int(
            raw.get("elapsed_ms", 0), field_name="receipt.elapsed_ms", code=Code.INVALID_RECEIPT, minimum=0
        )
        covers_raw = raw.get("covers", raw.get("override_refs", raw.get("covered_refs", [])))
        if covers_raw is None:
            covers_raw = []
        if not isinstance(covers_raw, list):
            fail(Code.INVALID_RECEIPT, "receipt covers must be a list", file=source_file, receipt_id=receipt_id)
        covers = tuple(
            require_text(item, field_name="receipt.covers item", code=Code.INVALID_RECEIPT) for item in covers_raw
        )
        sample_count_raw = raw.get("sample_count")
        sample_count = None
        if sample_count_raw is not None:
            sample_count = require_int(
                sample_count_raw,
                field_name="receipt.sample_count",
                code=Code.INVALID_RECEIPT,
                minimum=0,
            )
        sample_fingerprint = raw.get("sample_fingerprint")
        if sample_fingerprint is not None and not isinstance(sample_fingerprint, str):
            fail(Code.INVALID_RECEIPT, "receipt sample_fingerprint must be a string", receipt_id=receipt_id)
        return cls(
            receipt_id=receipt_id,
            domain=domain,
            shard_id=shard_id,
            parameter_version=parameter_version,
            elapsed_ms=elapsed_ms,
            covers=covers,
            source_file=source_file,
            sample_count=sample_count,
            sample_fingerprint=sample_fingerprint,
            raw=dict(raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "domain": self.domain,
            "shard_id": self.shard_id,
            "parameter_version": self.parameter_version,
            "elapsed_ms": self.elapsed_ms,
            "covers": list(self.covers),
            "source_file": self.source_file,
            "sample_count": self.sample_count,
            "sample_fingerprint": self.sample_fingerprint,
        }


def ensure_unique_documents(documents: Iterable[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = {}
    seen: dict[str, set[str]] = {}
    for doc in documents:
        doc = doc.with_content_hash()
        domain_seen = seen.setdefault(doc.domain, set())
        if doc.doc_id in domain_seen:
            fail(Code.DUPLICATE_DOCUMENT, "document identifiers must be unique within a domain", domain=doc.domain, doc_id=doc.doc_id)
        domain_seen.add(doc.doc_id)
        grouped.setdefault(doc.domain, []).append(doc)
    return grouped
