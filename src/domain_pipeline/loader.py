from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json_text, fingerprint, sorted_failures
from .config import PipelineConfig


@dataclass(frozen=True)
class ValidSample:
    sample_id: str
    line: int
    numeric: dict[str, float | None]
    categorical: dict[str, Any]
    label: int
    split: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ParsedSamples:
    samples: list[ValidSample]
    failures: list[dict[str, Any]]

    @property
    def train_samples(self) -> list[ValidSample]:
        return [sample for sample in self.samples if sample.split == "train"]

    @property
    def validation_samples(self) -> list[ValidSample]:
        return [sample for sample in self.samples if sample.split == "validation"]


@dataclass
class SampleStore:
    lines: list[str]

    @classmethod
    def from_path(cls, path: str | Path) -> "SampleStore":
        text = Path(path).read_text(encoding="utf-8-sig")
        return cls(lines=text.splitlines())

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "SampleStore":
        return cls(lines=[line.rstrip("\r\n") for line in lines])

    def text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")

    def raw_fingerprint(self) -> str:
        return fingerprint("raw-sample-lines:v1", self.lines)

    def append_records(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.lines.append(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )

    def correct_record(
        self,
        sample_id: str,
        patch: dict[str, Any],
        *,
        id_field: str = "id",
    ) -> tuple[bool, str | None]:
        matches: list[tuple[int, dict[str, Any]]] = []
        for index, line in enumerate(self.lines):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get(id_field) == sample_id:
                matches.append((index, parsed))
        if not matches:
            return False, "E_SAMPLE_NOT_FOUND"
        if len(matches) != 1:
            return False, "E_SAMPLE_AMBIGUOUS"
        if id_field in patch and patch[id_field] != sample_id:
            return False, "E_EDIT_ID_IMMUTABLE"
        index, parsed = matches[0]
        parsed.update(patch)
        self.lines[index] = json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return True, None


def _sample_failure(
    line: int,
    codes: list[str],
    messages: list[str],
    sample_id: str | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "scope": "sample",
        "line": line,
        "codes": codes,
        "message": "; ".join(messages),
    }
    if sample_id is not None:
        failure["sample_id"] = sample_id
    return failure


def _is_missing(value: Any) -> bool:
    return value is None


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_label(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def parse_samples(lines: Iterable[str], config: PipelineConfig) -> ParsedSamples:
    samples: list[ValidSample] = []
    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    id_field = config.data["id_field"]
    label_field = config.data["label_field"]
    split_field = config.data["split_field"]
    numeric_columns = list(config.data["numeric_columns"])
    categorical_columns = list(config.data["categorical_columns"])
    domains = config.data["categorical_domains"]
    domain_tokens = {
        column: {canonical_json_text(value) for value in domains.get(column, [])}
        for column in categorical_columns
    }

    for zero_index, line in enumerate(lines):
        line_number = zero_index + 1
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(
                _sample_failure(
                    line_number,
                    ["E_JSON_PARSE"],
                    [f"invalid JSON: {exc.msg}"],
                )
            )
            continue
        if not isinstance(record, dict):
            failures.append(
                _sample_failure(
                    line_number,
                    ["E_SAMPLE_OBJECT"],
                    ["sample line must be a JSON object"],
                )
            )
            continue

        codes: list[str] = []
        messages: list[str] = []
        sample_id = record.get(id_field)
        if not isinstance(sample_id, str) or not sample_id:
            codes.append("E_SAMPLE_ID_INVALID")
            messages.append("sample id must be a non-empty string")
            sample_id = None
        elif sample_id in seen_ids:
            codes.append("E_SAMPLE_ID_DUPLICATE")
            messages.append(f"duplicate sample id: {sample_id}")
        else:
            seen_ids.add(sample_id)

        numeric: dict[str, float | None] = {}
        for column in numeric_columns:
            value = record.get(column)
            if _is_missing(value):
                numeric[column] = None
            elif _is_finite_number(value):
                numeric[column] = float(value)
            else:
                codes.append("E_NUM_NONFINITE")
                messages.append(f"numeric column {column!r} must be finite or null")
                numeric[column] = None

        categorical: dict[str, Any] = {}
        for column in categorical_columns:
            value = record.get(column)
            if _is_missing(value):
                categorical[column] = None
            elif isinstance(value, (dict, list)):
                codes.append("E_CATEGORY_DOMAIN")
                messages.append(f"categorical column {column!r} must be a scalar")
                categorical[column] = None
            elif canonical_json_text(value) not in domain_tokens[column]:
                codes.append("E_CATEGORY_DOMAIN")
                messages.append(
                    f"categorical value {value!r} for {column!r} is outside its domain"
                )
                categorical[column] = value
            else:
                categorical[column] = value

        label = record.get(label_field)
        if not _is_label(label):
            codes.append("E_LABEL_BINARY")
            messages.append("label must be integer 0 or 1")
            label = 0

        split = record.get(split_field)
        if split not in ("train", "validation"):
            codes.append("E_SPLIT_VALUE")
            messages.append("split must be 'train' or 'validation'")
            split = "train"

        if codes:
            failures.append(
                _sample_failure(
                    line_number,
                    list(dict.fromkeys(codes)),
                    messages,
                    sample_id if isinstance(sample_id, str) else None,
                )
            )
            continue

        samples.append(
            ValidSample(
                sample_id=sample_id,
                line=line_number,
                numeric=numeric,
                categorical=categorical,
                label=label,
                split=split,
                raw=record,
            )
        )

    samples.sort(key=lambda sample: sample.sample_id)
    return ParsedSamples(samples=samples, failures=failures)


def feature_input_fingerprint(samples: Iterable[ValidSample]) -> str:
    values = [
        {
            "sample_id": sample.sample_id,
            "numeric": {
                key: sample.numeric[key]
                for key in sorted(sample.numeric)
            },
            "categorical": {
                key: sample.categorical[key]
                for key in sorted(sample.categorical)
            },
        }
        for sample in sorted(samples, key=lambda item: item.sample_id)
    ]
    return fingerprint("feature-input:v1", values)


def full_data_fingerprint(samples: Iterable[ValidSample]) -> str:
    values = [
        {
            "sample_id": sample.sample_id,
            "numeric": {
                key: sample.numeric[key]
                for key in sorted(sample.numeric)
            },
            "categorical": {
                key: sample.categorical[key]
                for key in sorted(sample.categorical)
            },
            "label": sample.label,
            "split": sample.split,
        }
        for sample in sorted(samples, key=lambda item: item.sample_id)
    ]
    return fingerprint("full-data:v1", values)


def state_failures(
    config_failures: Iterable[dict[str, Any]],
    sample_failures: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    combined = [dict(item) for item in config_failures] + [
        dict(item) for item in sample_failures
    ]
    return sorted_failures(combined)
