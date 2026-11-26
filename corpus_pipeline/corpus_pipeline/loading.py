from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .errors import Code, PipelineError, fail
from .fixed import parse_decimal
from .models import (
    Document,
    DomainSpec,
    EditCommand,
    PipelineConfig,
    canonical_json,
    content_hash,
    ensure_unique_documents,
    require_int,
)
from .synthetic import append_batch, derive_seed, generate_domain_streams, replacement_document


@dataclass(frozen=True, slots=True)
class LoadedInputs:
    config: PipelineConfig
    documents: Mapping[str, tuple[Document, ...]]
    edits: tuple[EditCommand, ...]
    domains_fingerprint: str
    documents_fingerprint: str
    config_hash: str
    input_hash: str
    applied_edit_fingerprint: str
    source: Mapping[str, str] = field(default_factory=dict)

    def domain_documents(self, domain_id: str) -> tuple[Document, ...]:
        return self.documents.get(domain_id, ())

    def domain_fingerprint(self, domain_id: str) -> str:
        domain = self.config.domain(domain_id)
        docs = self.documents.get(domain_id, ())
        payload = {
            "domain": domain.fingerprint_payload(),
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "content_hash": doc.content_hash,
                    "batch_index": doc.batch_index,
                    "generator_seed": doc.generator_seed,
                    "source": doc.source,
                }
                for doc in sorted(docs, key=lambda item: item.doc_id)
            ],
        }
        return content_hash(payload)

    def domain_component_fingerprints(self, domain_id: str) -> dict[str, str]:
        domain = self.config.domain(domain_id)
        docs = self.documents.get(domain_id, ())
        return {
            "declaration": content_hash(domain.to_dict()),
            "weight": content_hash(str(domain.weight)),
            "documents": content_hash(
                [
                    [doc.doc_id, doc.content_hash, doc.batch_index, doc.generator_seed]
                    for doc in sorted(docs, key=lambda item: item.doc_id)
                ]
            ),
            "combined": self.domain_fingerprint(domain_id),
        }

    def effective_version(self) -> int:
        edit_cutoff = max((edit.seq for edit in self.edits), default=0)
        return self.config.version + edit_cutoff

    def metadata(self) -> dict[str, Any]:
        return {
            "global_version": self.effective_version(),
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "domains_fingerprint": self.domains_fingerprint,
            "documents_fingerprint": self.documents_fingerprint,
            "applied_edit_fingerprint": self.applied_edit_fingerprint,
        }


def read_yaml(path: str | Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except FileNotFoundError:
        fail(Code.INVALID_SCHEMA, "file does not exist", path=str(path))
    except yaml.YAMLError as exc:
        fail(Code.INVALID_SCHEMA, "YAML parse failure", path=str(path), error=str(exc))


def write_yaml(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
    temporary.replace(target)


def load_config(path_or_value: str | Path | Mapping[str, Any]) -> PipelineConfig:
    raw = read_yaml(path_or_value) if isinstance(path_or_value, (str, Path)) else path_or_value
    return PipelineConfig.from_dict(raw)


def load_document_stream(path: str | Path, known_domains: set[str]) -> dict[str, list[Document]]:
    documents: list[Document] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    fail(
                        Code.INVALID_DOCUMENT,
                        "invalid JSON document stream line",
                        path=str(path),
                        line=line_number,
                        error=str(exc),
                    )
                documents.append(Document.from_dict(raw, known_domains=known_domains))
    except FileNotFoundError:
        fail(Code.INVALID_SCHEMA, "document stream does not exist", path=str(path))
    grouped = ensure_unique_documents(documents)
    for domain in grouped:
        if domain not in known_domains:
            fail(Code.UNKNOWN_DOMAIN, "document stream references an unknown domain", domain=domain)
    return grouped


def write_document_stream(path: str | Path, documents: Mapping[str, Sequence[Document]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for domain in sorted(documents):
            for doc in sorted(documents[domain], key=lambda item: item.doc_id):
                handle.write(json.dumps(doc.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(target)


def load_edit_stream(path: str | Path | None, known_domains: set[str]) -> list[EditCommand]:
    if path is None:
        return []
    commands: list[EditCommand] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    fail(Code.INVALID_EDIT, "invalid edit stream JSON", path=str(path), line=line_number, error=str(exc))
                command = EditCommand.from_dict(raw)
                domain = raw.get("domain")
                if not isinstance(domain, str) or domain not in known_domains:
                    fail(Code.UNKNOWN_DOMAIN, "edit references an unknown domain", domain=domain, seq=command.seq)
                commands.append(command)
    except FileNotFoundError:
        fail(Code.INVALID_EDIT, "edit stream does not exist", path=str(path))
    for previous, current in zip(commands, commands[1:]):
        if current.seq <= previous.seq:
            fail(
                Code.INVALID_EDIT,
                "edit command sequence numbers must be strictly increasing",
                previous=previous.seq,
                current=current.seq,
            )
    return commands


def apply_edit_stream(
    config: PipelineConfig,
    documents: Mapping[str, Sequence[Document]],
    edits: Iterable[EditCommand],
) -> tuple[PipelineConfig, dict[str, list[Document]]]:
    current_config = config
    current_docs: dict[str, list[Document]] = {
        domain: sorted(list(items), key=lambda doc: doc.doc_id) for domain, items in documents.items()
    }
    for edit in edits:
        raw = edit.payload
        domain_id = raw.get("domain")
        if not isinstance(domain_id, str) or domain_id not in current_config.domain_ids:
            fail(Code.UNKNOWN_DOMAIN, "edit references an unknown domain", domain=domain_id, seq=edit.seq)
        domain = current_config.domain(domain_id)
        doc_list = current_docs.setdefault(domain_id, [])
        if edit.op == "set_weight":
            current_config = current_config.replace_domain_weight(domain_id, raw.get("weight"))
        elif edit.op == "replace_document":
            doc_id = raw.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                fail(Code.INVALID_EDIT, "replace_document requires doc_id", seq=edit.seq, domain=domain_id)
            index = next((i for i, doc in enumerate(doc_list) if doc.doc_id == doc_id), None)
            if index is None:
                fail(Code.INVALID_EDIT, "replace_document targets an unknown document", seq=edit.seq, domain=domain_id, doc_id=doc_id)
            doc_list[index] = replacement_document(domain, raw)
        elif edit.op == "append_batch":
            count = require_int(raw.get("count", 1), field_name="edit.append_batch.count", code=Code.INVALID_EDIT, minimum=1)
            append_seed = require_int(
                raw.get("seed", derive_seed(domain.seed, domain.id, "edit-append", edit.seq)),
                field_name="edit.append_batch.seed",
                code=Code.INVALID_EDIT,
            )
            new_documents = append_batch(domain, doc_list, count, append_seed)
            doc_list.extend(new_documents)
        else:  # Defensive: EditCommand validation should already prevent this.
            fail(Code.INVALID_EDIT, "unsupported edit operation", op=edit.op, seq=edit.seq)
        current_docs[domain_id] = sorted(doc_list, key=lambda doc: doc.doc_id)
    grouped = ensure_unique_documents(doc for docs in current_docs.values() for doc in docs)
    for required_domain in current_config.domain_ids:
        grouped.setdefault(required_domain, [])
    if sum((domain.weight for domain in current_config.domains), parse_decimal(0)) <= 0:
        fail(Code.WEIGHT_SUM_ZERO, "sum of effective domain weights must be greater than zero")
    return current_config, grouped


def build_inputs(
    config: PipelineConfig | Mapping[str, Any] | str | Path,
    *,
    documents: Mapping[str, Sequence[Document]] | None = None,
    documents_path: str | Path | None = None,
    edits_path: str | Path | None = None,
    generate_missing_documents: bool = True,
    source: Mapping[str, str] | None = None,
) -> LoadedInputs:
    config_obj = load_config(config) if not isinstance(config, PipelineConfig) else config
    known_domains = set(config_obj.domain_ids)
    if documents is not None:
        baseline = dict(documents)
    elif documents_path is not None:
        baseline = load_document_stream(documents_path, known_domains)
    elif generate_missing_documents:
        baseline = generate_domain_streams(config_obj.domains)
    else:
        fail(Code.INVALID_SCHEMA, "no document stream supplied and generation is disabled")

    for domain in known_domains:
        if domain not in baseline:
            baseline[domain] = []
    baseline = {domain: sorted(docs, key=lambda doc: doc.doc_id) for domain, docs in baseline.items()}
    # Validate even in-memory streams with the same boundary as file input.
    baseline = ensure_unique_documents(doc.with_content_hash() for docs in baseline.values() for doc in docs)
    unknown = sorted(set(baseline) - known_domains)
    if unknown:
        fail(Code.UNKNOWN_DOMAIN, "document stream contains unknown domains", domains=unknown)

    edits = load_edit_stream(edits_path, known_domains) if isinstance(edits_path, (str, Path)) else []
    effective_config, effective_docs = apply_edit_stream(config_obj, baseline, edits)

    documents_fingerprint = content_hash(
        {
            domain: [
                [doc.doc_id, doc.content_hash, doc.batch_index, doc.generator_seed]
                for doc in sorted(items, key=lambda item: item.doc_id)
            ]
            for domain, items in sorted(effective_docs.items())
        }
    )
    edit_payload = [edit.to_dict() for edit in edits]
    applied_edit_fingerprint = content_hash(edit_payload)
    domains_fingerprint = content_hash(
        {
            "domains": [domain.fingerprint_payload() for domain in effective_config.domains],
            "documents": documents_fingerprint,
            "edits": applied_edit_fingerprint,
        }
    )
    config_hash = content_hash(
        {
            "config": effective_config.config_fingerprint_payload(),
            "edits": edit_payload,
        }
    )
    input_hash = content_hash(
        {
            "config_hash": config_hash,
            "documents_fingerprint": documents_fingerprint,
            "domains_fingerprint": domains_fingerprint,
        }
    )
    return LoadedInputs(
        config=effective_config,
        documents={domain: tuple(docs) for domain, docs in sorted(effective_docs.items())},
        edits=tuple(edits),
        domains_fingerprint=domains_fingerprint,
        documents_fingerprint=documents_fingerprint,
        config_hash=config_hash,
        input_hash=input_hash,
        applied_edit_fingerprint=applied_edit_fingerprint,
        source=dict(source or {}),
    )


def load_inputs(
    config_path: str | Path,
    *,
    documents_path: str | Path | None = None,
    edits_path: str | Path | None = None,
    generate_missing_documents: bool = True,
) -> LoadedInputs:
    return build_inputs(
        config_path,
        documents_path=documents_path,
        edits_path=edits_path,
        generate_missing_documents=generate_missing_documents,
        source={
            "config": str(config_path),
            **({"documents": str(documents_path)} if documents_path else {}),
            **({"edits": str(edits_path)} if edits_path else {}),
        },
    )


DEMO_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "version": 3,
    "seed": 20261126,
    "precision": {"decimal_places": 3, "quantization_tier": 10},
    "sampling": {
        "budget": 32,
        "per_domain_quota": 12,
        "shard_size": 4,
        "max_dedup_comparisons": 80,
    },
    "dedup": {"similarity_threshold_permille": 700, "boundary_rule": "include"},
    "tolerance": {
        "sampling_docs": 0,
        "similarity_permille": 1,
        "dedup_docs": 0,
        "composition_fixed": 2,
    },
    "domains": [
        {
            "id": "biology",
            "weight": 1,
            "seed": 101,
            "batch_size": 20,
            "vocab_size": 48,
            "length_min": 28,
            "length_max": 64,
            "duplicate_every": 4,
        },
        {
            "id": "finance",
            "weight": 2,
            "seed": 202,
            "batch_size": 20,
            "vocab_size": 48,
            "length_min": 30,
            "length_max": 70,
            "duplicate_every": 5,
        },
        {
            "id": "law",
            "weight": 1,
            "seed": 303,
            "batch_size": 20,
            "vocab_size": 48,
            "length_min": 26,
            "length_max": 68,
            "duplicate_every": 4,
        },
    ],
}


def generate_workspace(
    root: str | Path,
    *,
    config_override: Mapping[str, Any] | None = None,
    write_receipts: bool = True,
) -> dict[str, Any]:
    """Create a deterministic, self-contained local demo workspace.

    The project template itself ships empty ``data``/``config``/``state``
    directories.  This function is the only path that materializes sample
    inputs, and it is used by the CLI ``demo``/``generate`` commands.
    """

    root_path = Path(root)
    for name in ("data", "config", "state"):
        (root_path / name).mkdir(parents=True, exist_ok=True)
    raw_config = dict(DEMO_CONFIG)
    if config_override:
        raw_config.update(config_override)
    config = PipelineConfig.from_dict(raw_config)
    config_path = root_path / "config" / "pipeline.yaml"
    write_yaml(config_path, raw_config)
    baseline = generate_domain_streams(config.domains)
    documents_path = root_path / "data" / "documents.jsonl"
    write_document_stream(documents_path, baseline)
    inputs = build_inputs(
        config,
        documents=baseline,
        source={"config": str(config_path), "documents": str(documents_path)},
    )

    receipt_paths: list[str] = []
    shard_counts: dict[str, int] = {}
    if write_receipts:
        # Import locally to keep the loading layer independent from scheduling.
        from .sampling import build_sample_plan

        plan = build_sample_plan(inputs.config, inputs.documents)
        receipts_dir = root_path / "config" / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)
        shards = sorted(plan["shards"], key=lambda shard: shard["shard_id"])
        v1: list[dict[str, Any]] = []
        v2: list[dict[str, Any]] = []
        v3: list[dict[str, Any]] = []
        for index, shard in enumerate(shards):
            shard_id = shard["shard_id"]
            domain = shard["domain"]
            sample_count = len(shard["documents"])
            fingerprint = shard["sampling_fingerprint"]
            shard_counts[shard_id] = sample_count
            receipt_id = f"r1-{domain}-{index:03d}"
            v1.append(
                {
                    "receipt_id": receipt_id,
                    "domain": domain,
                    "shard_id": shard_id,
                    "parameter_version": 1,
                    "elapsed_ms": 40 + index,
                    "sample_count": sample_count,
                    "sample_fingerprint": fingerprint,
                }
            )
            if index == 0:
                v2.append(
                    {
                        "receipt_id": "r2-000",
                        "domain": domain,
                        "shard_id": shard_id,
                        "parameter_version": 2,
                        "elapsed_ms": 30,
                        "covers": [receipt_id],
                        "sample_count": sample_count,
                        "sample_fingerprint": fingerprint,
                    }
                )
                v3.append(
                    {
                        "receipt_id": "r3-000",
                        "domain": domain,
                        "shard_id": shard_id,
                        "parameter_version": 3,
                        "elapsed_ms": 12,
                        "covers": ["r2-000"],
                        "sample_count": sample_count,
                        "sample_fingerprint": fingerprint,
                    }
                )
            elif index == 1:
                # Same-version ambiguity resolved by integer elapsed time.
                v2.append(
                    {
                        "receipt_id": "r2-001",
                        "domain": domain,
                        "shard_id": shard_id,
                        "parameter_version": 2,
                        "elapsed_ms": 25,
                        "sample_count": sample_count,
                    }
                )
                v2.append(
                    {
                        "receipt_id": "r2-002",
                        "domain": domain,
                        "shard_id": shard_id,
                        "parameter_version": 2,
                        "elapsed_ms": 9,
                        "sample_count": sample_count,
                    }
                )
        for version, items in ((1, v1), (2, v2), (3, v3)):
            if not items:
                continue
            path = receipts_dir / f"receipts_v{version}.yaml"
            write_yaml(path, {"schema_version": 1, "receipts": items})
            receipt_paths.append(str(path))

    return {
        "root": str(root_path),
        "config": str(config_path),
        "documents": str(documents_path),
        "receipts": receipt_paths,
        "receipt_shard_counts": shard_counts,
        "config_hash": inputs.config_hash,
        "input_hash": inputs.input_hash,
    }


def copy_workspace(source: str | Path, target: str | Path) -> None:
    """Copy a generated workspace while resetting its state directory."""

    source_path = Path(source)
    target_path = Path(target)
    for name in ("config", "data"):
        source_dir = source_path / name
        target_dir = target_path / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
    (target_path / "state").mkdir(parents=True, exist_ok=True)
