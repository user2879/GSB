from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, sha256_bytes
from .specs import CONFIG_FORMAT, DOMAIN_FORMAT

DEMO_DOMAINS = [
    {
        "domain_id": "alpha",
        "seed": 71017,
        "doc_count": 24,
        "batch_size": 8,
        "vocab_size": 12,
        "motif_count": 3,
        "duplicate_rate_bp": 3800,
        "missing_rate_bp": 350,
    },
    {
        "domain_id": "bravo",
        "seed": 82031,
        "doc_count": 24,
        "batch_size": 8,
        "vocab_size": 12,
        "motif_count": 4,
        "duplicate_rate_bp": 3400,
        "missing_rate_bp": 500,
    },
    {
        "domain_id": "cyan",
        "seed": 93047,
        "doc_count": 24,
        "batch_size": 8,
        "vocab_size": 12,
        "motif_count": 3,
        "duplicate_rate_bp": 4200,
        "missing_rate_bp": 250,
    },
    {
        "domain_id": "delta",
        "seed": 104009,
        "doc_count": 24,
        "batch_size": 8,
        "vocab_size": 12,
        "motif_count": 4,
        "duplicate_rate_bp": 3000,
        "missing_rate_bp": 450,
    },
]


def hash_u64(*parts: Any) -> int:
    payload = canonical_bytes(["reconflow-rng/v1", *parts])
    import hashlib

    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=False)


def doc_content_digest(doc: dict[str, Any]) -> str:
    payload = {
        "format_version": 1,
        "domain_id": doc["domain_id"],
        "batch_seq": doc["batch_seq"],
        "doc_seq": doc["doc_seq"],
        "term_counts": list(doc["term_counts"]),
        "length": doc["length"],
        "missing_bitmap": list(doc["missing_bitmap"]),
    }
    return sha256_bytes(canonical_bytes(payload))


def selection_rank(domain: dict[str, Any], doc: dict[str, Any]) -> str:
    value = hash_u64(
        "weighted-selection/v1",
        int(domain["seed"]),
        str(domain["domain_id"]),
        int(doc["doc_seq"]),
        str(doc["content_digest"]),
    )
    return f"{value:016x}:{int(doc['doc_seq']):08d}"


def _base_vector(domain: dict[str, Any], motif: int) -> list[int]:
    vocab = int(domain["vocab_size"])
    motif_count = int(domain["motif_count"])
    center = (motif * max(1, vocab // motif_count) + hash_u64("center", domain["seed"], motif) % 3) % vocab
    vector: list[int] = []
    for term in range(vocab):
        circular_distance = min(abs(term - center), vocab - abs(term - center))
        value = max(0, 6 - circular_distance // 2) + hash_u64("base-term", domain["seed"], motif, term) % 3
        vector.append(int(value))
    return vector


def generate_domain_doc(domain: dict[str, Any], doc_seq: int) -> dict[str, Any]:
    domain_id = str(domain["domain_id"])
    seed = int(domain["seed"])
    batch_size = int(domain["batch_size"])
    vocab = int(domain["vocab_size"])
    motif_count = int(domain["motif_count"])

    motif = hash_u64("motif", seed, doc_seq) % motif_count
    kind_roll = hash_u64("kind", seed, doc_seq) % 10_000
    base = _base_vector(domain, motif)

    if kind_roll < int(domain["duplicate_rate_bp"]):
        variant = hash_u64("near-variant", seed, doc_seq) % 4
        counts = list(base)
        if variant == 1:
            index = hash_u64("perturb-index", seed, doc_seq, 1) % vocab
            counts[index] = max(0, counts[index] - 1)
        elif variant == 2:
            index = hash_u64("perturb-index", seed, doc_seq, 2) % vocab
            counts[index] += 1
        elif variant == 3:
            left = hash_u64("swap", seed, doc_seq, "a") % vocab
            right = hash_u64("swap", seed, doc_seq, "b") % vocab
            counts[left], counts[right] = counts[right], counts[left]
    else:
        counts = [hash_u64("unique-count", seed, doc_seq, term) % 4 for term in range(vocab)]
        spike = hash_u64("unique-spike", seed, doc_seq) % vocab
        counts[spike] += 5 + hash_u64("unique-spike-size", seed, doc_seq) % 5

    missing_bitmap = [
        1 if hash_u64("missing", seed, doc_seq, term) % 10_000 < int(domain["missing_rate_bp"]) else 0
        for term in range(vocab)
    ]
    for term, missing in enumerate(missing_bitmap):
        if missing:
            counts[term] = 0

    if sum(counts) < 6:
        for term in range(min(4, vocab)):
            if missing_bitmap[term] == 1:
                missing_bitmap[term] = 0
            counts[term] = max(counts[term], 2 + term % 3)

    batch_seq = doc_seq // batch_size
    doc = {
        "doc_id": f"{domain_id}:d{doc_seq:05d}",
        "domain_id": domain_id,
        "doc_seq": doc_seq,
        "batch_seq": batch_seq,
        "length": int(sum(counts)),
        "term_counts": [int(value) for value in counts],
        "missing_bitmap": [int(value) for value in missing_bitmap],
    }
    doc["content_digest"] = doc_content_digest(doc)
    return doc


def generate_domain_docs(domain: dict[str, Any]) -> list[dict[str, Any]]:
    return [generate_domain_doc(domain, seq) for seq in range(int(domain["doc_count"]))]


def generate_batches(domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for domain in sorted(domains, key=lambda item: item["domain_id"]):
        batch_size = int(domain["batch_size"])
        docs = generate_domain_docs(domain)
        for batch_seq in range((len(docs) + batch_size - 1) // batch_size):
            slice_docs = docs[batch_seq * batch_size : (batch_seq + 1) * batch_size]
            batch = {
                "format_version": 1,
                "record_type": "sample_batch",
                "batch_id": f"{domain['domain_id']}#b{batch_seq:04d}",
                "domain_id": domain["domain_id"],
                "batch_seq": batch_seq,
                "arrival_ordinal": len(batches),
                "docs": slice_docs,
            }
            batch["batch_digest"] = sha256_bytes(
                canonical_bytes([doc["content_digest"] for doc in slice_docs])
            )
            batches.append(batch)
    batches.sort(key=lambda batch: (int(batch["batch_seq"]), str(batch["domain_id"]), int(batch["arrival_ordinal"])))
    for ordinal, batch in enumerate(batches):
        batch["stream_ordinal"] = ordinal
    return batches


def default_config(domain_ids: list[str], *, config_version: int = 1) -> dict[str, Any]:
    weight = 10_000 // len(domain_ids)
    remainder = 10_000 - weight * len(domain_ids)
    weights = {domain: weight for domain in sorted(domain_ids)}
    for domain in sorted(domain_ids)[:remainder]:
        weights[domain] += 1
    return {
        "format_version": 1,
        "spec_format": CONFIG_FORMAT,
        "config_version": config_version,
        "bp_scale": 10_000,
        "rounding_rule": "floor_nonnegative_then_descending_remainder_then_id",
        "saturation_rule": "saturate_to_int64_max",
        "sampling_budget_points": 48,
        "global_sampling_point_budget": 18,
        "domain_sampling_point_quota": 8,
        "similarity_threshold_bp": 8_500,
        "shard_size": 6,
        "weights_bp": weights,
        "stage_params": {
            "weighted_sampling": {"selection": "blake2b_rank_then_quota", "overflow_saturation": "int64_max"},
            "dedup": {"min_shared_terms": 2, "boundary_band_bp": 25, "rpc_timeout_ms": 2_000},
            "normalization": {"remainder_distribution": "descending_remainder_then_domain_id"},
        },
    }


def write_ndjson(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def generate_workspace(root: str | Path, *, overwrite: bool = True) -> Path:
    root = Path(root)
    specs_dir = root / "specs"
    config_dir = root / "config"
    data_dir = root / "data"
    for directory in (specs_dir, config_dir, data_dir, root / "cache", root / "ledger", root / "snapshots"):
        directory.mkdir(parents=True, exist_ok=True)

    domains_path = specs_dir / "domains.json"
    config_path = config_dir / "config.json"
    batches_path = data_dir / "batches.ndjson"
    if not overwrite and domains_path.exists():
        return root

    domains_payload = {"format_version": 1, "spec_format": DOMAIN_FORMAT, "domains": DEMO_DOMAINS}
    domains_path.write_text(json.dumps(domains_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = default_config([d["domain_id"] for d in DEMO_DOMAINS])
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_ndjson(batches_path, generate_batches(DEMO_DOMAINS))
    from .workspace import default_candidates

    (root / "config" / "candidates.json").write_text(
        json.dumps({"format_version": 1, "candidates": default_candidates()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root
