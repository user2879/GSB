from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import fingerprint
from .errors import InputValidationError


@dataclass(frozen=True)
class SampleSet:
    samples: dict[str, dict[str, Any]]
    sample_order: tuple[str, ...]
    batches: tuple[dict[str, Any], ...]
    source_paths: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return sample_set_fingerprint(self)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputValidationError(f"cannot read batch file: {exc}", field=f"samples.{path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InputValidationError(
                f"invalid JSONL at line {line_number}: {exc.msg}",
                field=f"samples.{path.name}.line-{line_number}",
            ) from exc
    return rows


def load_sample_set(sample_dir: str | Path, config: dict[str, Any]) -> SampleSet:
    root = Path(sample_dir)
    if not root.exists():
        raise InputValidationError(f"sample directory not found: {root}", field="samples")
    paths = sorted([path for path in root.rglob("*.jsonl") if path.is_file()], key=lambda p: p.as_posix())
    if not paths:
        raise InputValidationError(f"no JSONL batch files found under {root}", field="samples")

    samples: dict[str, dict[str, Any]] = {}
    batches: list[dict[str, Any]] = []
    source_paths: list[str] = []
    known_domains = set(config["domain_names"])
    width = config["missing_bitmap_width"]

    for path in paths:
        rows = _read_jsonl(path)
        if not rows:
            raise InputValidationError("batch file is empty", field=f"samples.{path.name}")
        header = rows[0]
        if not isinstance(header, dict) or header.get("record_type") != "batch_header":
            raise InputValidationError(
                "first JSONL record must be a batch_header",
                field=f"samples.{path.name}.line-1",
            )
        batch_id = header.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise InputValidationError(
                "batch_header.batch_id must be a nonempty string",
                field="batch_header.batch_id",
                batch_id=str(batch_id) if batch_id is not None else None,
            )
        declared_fields = header.get("declared_fields")
        if not isinstance(declared_fields, list) or not all(isinstance(item, str) for item in declared_fields):
            raise InputValidationError(
                "batch_header.declared_fields must be a list of strings",
                field="batch_header.declared_fields",
                batch_id=batch_id,
            )
        impute_values = header.get("impute_values")
        if not isinstance(impute_values, dict):
            raise InputValidationError(
                "batch_header.impute_values must be an object",
                field="batch_header.impute_values",
                batch_id=batch_id,
            )
        length_fill = impute_values.get("length_distribution")
        if not isinstance(length_fill, list) or len(length_fill) != width:
            raise InputValidationError(
                f"batch_header.impute_values.length_distribution must have width {width}",
                field="batch_header.impute_values.length_distribution",
                batch_id=batch_id,
            )
        if not all(_is_plain_int(item) and item >= 0 for item in length_fill):
            raise InputValidationError(
                "batch_header.impute_values.length_distribution must contain nonnegative integers",
                field="batch_header.impute_values.length_distribution",
                batch_id=batch_id,
            )

        batch_sample_ids: list[str] = []
        for line_number, row in enumerate(rows[1:], start=2):
            field_prefix = f"batches.{batch_id}.line-{line_number}"
            if not isinstance(row, dict):
                raise InputValidationError("sample record must be an object", field=field_prefix, batch_id=batch_id)
            record_type = row.get("record_type", "sample")
            if record_type != "sample":
                raise InputValidationError(
                    "every non-header JSONL row must have record_type=sample",
                    field=f"{field_prefix}.record_type",
                    batch_id=batch_id,
                )
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise InputValidationError(
                    "sample_id must be a nonempty string",
                    field=f"{field_prefix}.sample_id",
                    batch_id=batch_id,
                )
            if sample_id in samples:
                raise InputValidationError(
                    "sample_id is not globally unique",
                    field="samples.sample_id",
                    batch_id=batch_id,
                    details={"sample_id": sample_id, "first_batch": samples[sample_id]["batch_id"]},
                )

            domain = row.get("domain")
            if domain not in known_domains:
                raise InputValidationError(
                    "sample domain is not declared in config.domains",
                    field=f"samples.{sample_id}.domain",
                    batch_id=batch_id,
                    details={"domain": domain, "declared_domains": sorted(known_domains)},
                )

            counts_key = "token_counts"
            counts = row.get(counts_key)
            if counts is None:
                for alias in ("term_counts", "word_counts"):
                    if row.get(alias) is not None:
                        counts = row[alias]
                        counts_key = alias
                        break
            if not isinstance(counts, dict):
                raise InputValidationError(
                    "token_counts must be an object mapping terms to nonnegative integer counts",
                    field=f"samples.{sample_id}.token_counts",
                    batch_id=batch_id,
                )
            normalized_counts: dict[str, int] = {}
            for term, count in counts.items():
                if not isinstance(term, str) or not term:
                    raise InputValidationError(
                        "token_counts keys must be nonempty strings",
                        field=f"samples.{sample_id}.token_counts",
                        batch_id=batch_id,
                    )
                if not _is_plain_int(count) or count < 0:
                    raise InputValidationError(
                        "token_counts values must be nonnegative integers",
                        field=f"samples.{sample_id}.token_counts.{term}",
                        batch_id=batch_id,
                    )
                normalized_counts[term] = count

            lengths = row.get("length_distribution")
            if not isinstance(lengths, list) or len(lengths) != width:
                raise InputValidationError(
                    f"length_distribution must be a list of width {width}",
                    field=f"samples.{sample_id}.length_distribution",
                    batch_id=batch_id,
                )
            if not all(_is_plain_int(item) and item >= 0 for item in lengths):
                raise InputValidationError(
                    "length_distribution entries must be nonnegative integers",
                    field=f"samples.{sample_id}.length_distribution",
                    batch_id=batch_id,
                )

            bitmap = row.get("missing_bitmap")
            if not isinstance(bitmap, list) or len(bitmap) != width:
                raise InputValidationError(
                    f"missing_bitmap must be a list of width {width}",
                    field=f"samples.{sample_id}.missing_bitmap",
                    batch_id=batch_id,
                )
            if not all(_is_plain_int(item) and item in (0, 1) for item in bitmap):
                raise InputValidationError(
                    "missing_bitmap entries must be 0 or 1 integers",
                    field=f"samples.{sample_id}.missing_bitmap",
                    batch_id=batch_id,
                )

            declared_batch = row.get("batch_id")
            if declared_batch is not None and declared_batch != batch_id:
                raise InputValidationError(
                    "sample batch_id does not match its JSONL batch header",
                    field=f"samples.{sample_id}.batch_id",
                    batch_id=batch_id,
                    details={"declared_batch_id": declared_batch},
                )

            sample = {
                "sample_id": sample_id,
                "batch_id": batch_id,
                "domain": domain,
                "token_counts": normalized_counts,
                "length_distribution": list(lengths),
                "missing_bitmap": list(bitmap),
                "batch_impute_length_distribution": list(length_fill),
                "source_path": path.name,
                "term_count_total": sum(normalized_counts.values()),
            }
            if row.get("seed") is not None:
                seed = row["seed"]
                if not _is_plain_int(seed):
                    raise InputValidationError(
                        "seed must be an integer when provided",
                        field=f"samples.{sample_id}.seed",
                        batch_id=batch_id,
                    )
                sample["seed"] = seed
            samples[sample_id] = sample
            batch_sample_ids.append(sample_id)

        if not batch_sample_ids:
            raise InputValidationError(
                "batch contains no sample records",
                field=f"batches.{batch_id}",
                batch_id=batch_id,
            )
        batches.append(
            {
                "batch_id": batch_id,
                "path": path.name,
                "declared_fields": list(declared_fields),
                "impute_values": {"length_distribution": list(length_fill)},
                "sample_ids": batch_sample_ids,
            }
        )
        source_paths.append(path.name)

    return SampleSet(
        samples=samples,
        sample_order=tuple(sorted(samples)),
        batches=tuple(batches),
        source_paths=tuple(source_paths),
    )


def sample_content_fingerprint(sample: dict[str, Any]) -> str:
    return fingerprint(
        {
            "sample_id": sample["sample_id"],
            "domain": sample["domain"],
            "batch_id": sample["batch_id"],
            "token_counts": sample["token_counts"],
            "length_distribution": sample["length_distribution"],
            "missing_bitmap": sample["missing_bitmap"],
            "batch_impute_length_distribution": sample["batch_impute_length_distribution"],
        }
    )


def sample_set_fingerprint(sample_set: SampleSet | Iterable[dict[str, Any]]) -> str:
    if isinstance(sample_set, SampleSet):
        items = [sample_content_fingerprint(sample_set.samples[sample_id]) for sample_id in sample_set.sample_order]
    else:
        items = [sample_content_fingerprint(sample) for sample in sample_set]
    return fingerprint(items)
