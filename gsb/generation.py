from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import GSBError
from .model import ColumnSpec, ScenarioSpec, TableSpec
from .util import (
    INT64_NULL,
    atomic_write_text,
    logical_content_digest,
    read_npz,
    sha256_file,
    sha256_json,
    stable_seed,
    write_npz_deterministic,
)


@dataclass(frozen=True)
class TableSample:
    table: str
    columns: Mapping[str, np.ndarray]
    content_digest: str
    file_digest: str
    path: str
    manifest_path: str

    @property
    def rows(self) -> int:
        if not self.columns:
            return 0
        return int(len(next(iter(self.columns.values()))))

    def to_summary(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "rows": self.rows,
            "columns": {
                name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                for name, array in sorted(self.columns.items())
            },
            "content_digest": self.content_digest,
            "file_digest": self.file_digest,
            "path": self.path,
            "manifest_path": self.manifest_path,
        }


def _effective_seed(value: int, *parts: Any) -> int:
    return stable_seed(int(value) & ((1 << 128) - 1), *parts) & ((1 << 63) - 1)


def _domain_for(column: ColumnSpec, rows: int) -> int | float:
    if column.domain is not None:
        return column.domain
    minimum = max(rows * 4, 256)
    if column.kind == "int":
        return int(minimum)
    return float(minimum)


def _generate_column(
    table: TableSpec,
    column: ColumnSpec,
    rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    domain = _domain_for(column, rows)
    if column.kind == "int":
        domain_int = max(int(domain), rows if column.unique else 1)
    else:
        domain_float = max(float(domain), float(rows) if column.unique else 1.0)
    if column.unique:
        if column.kind == "int":
            if domain_int < rows:
                raise GSBError(
                    "E_SPEC",
                    "unique 列的值域不足以容纳声明行数",
                    {"table": table.name, "column": column.name, "domain": domain_int, "rows": rows},
                )
            values = np.arange(1, domain_int + 1, dtype=np.int64)
            rng.shuffle(values)
            result = values[:rows].astype(np.int64, copy=False)
        else:
            base = np.arange(1, rows + 1, dtype=np.float64) / 7.0
            jitter = rng.random(rows) * 1e-10
            result = base + jitter
    elif column.kind == "int":
        if table.skewed and table.skew_hot_fraction > 0:
            hot_rows = int(round(rows * table.skew_hot_fraction))
            hot_domain = max(2, int(round(domain_int * table.skew_hot_domain_fraction)))
            hot_values = rng.integers(1, hot_domain + 1, size=hot_rows, dtype=np.int64)
            cold_values = rng.integers(1, domain_int + 1, size=rows - hot_rows, dtype=np.int64)
            result = np.concatenate([hot_values, cold_values])
            rng.shuffle(result)
        else:
            result = rng.integers(1, domain_int + 1, size=rows, dtype=np.int64)
    else:
        if table.skewed and table.skew_hot_fraction > 0:
            hot_rows = int(round(rows * table.skew_hot_fraction))
            hot_domain = max(1.0, domain_float * table.skew_hot_domain_fraction)
            hot_values = np.round(rng.uniform(1.0, hot_domain, size=hot_rows), 3)
            cold_values = np.round(rng.uniform(1.0, domain_float, size=rows - hot_rows), 3)
            result = np.concatenate([hot_values, cold_values])
            rng.shuffle(result)
        else:
            result = np.round(rng.uniform(1.0, domain_float, size=rows), 3)
        if column.step is not None:
            result = np.round(result / column.step) * column.step
        result = result.astype(np.float64, copy=False)
    if column.null_fraction > 0.0:
        null_count = int(round(rows * column.null_fraction))
        if null_count > 0:
            indices = rng.choice(rows, size=null_count, replace=False)
            if column.kind == "int":
                result[int(indices[0])] = INT64_NULL
                if null_count > 1:
                    result[indices] = INT64_NULL
                result = result.astype(np.int64, copy=False)
            else:
                result[indices] = np.nan
    return np.ascontiguousarray(result, dtype=column.dtype)


def generate_table_arrays(table: TableSpec) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for column in table.columns:
        rng = np.random.Generator(np.random.PCG64(_effective_seed(table.seed, table.name, column.name, table.skewed)))
        arrays[column.name] = _generate_column(table, column, table.declared_rows, rng)
    return arrays


def _sample_paths(workdir: Path, scenario: ScenarioSpec, table_name: str) -> tuple[Path, Path]:
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in table_name)
    table_digest = sha256_json({"schema_version": scenario.schema_version, "table": scenario.table(table_name).to_dict()})
    base = workdir / "samples" / f"{table_digest[:16]}_{safe_name}"
    return base.with_suffix(".npz"), base.with_suffix(".manifest.json")


def _manifest_payload(
    scenario: ScenarioSpec,
    table: TableSpec,
    arrays: Mapping[str, np.ndarray],
    content_digest: str,
    file_digest: str,
) -> dict[str, Any]:
    return {
        "format": "gsb-sample-v1",
        "spec_digest": scenario.digest,
        "table": table.name,
        "table_seed": table.seed,
        "skewed": table.skewed,
        "declared_rows": table.declared_rows,
        "rows": table.declared_rows,
        "columns": [
            {
                "name": column.name,
                "type": column.kind,
                "dtype": str(arrays[column.name].dtype),
                "null_fraction": column.null_fraction,
                "unique": column.unique,
                "step": column.step,
            }
            for column in table.columns
        ],
        "content_digest": content_digest,
        "file_digest": file_digest,
        "generation_rule": "pcg64/sha256-seed/pcg-shuffle/skew-v1",
    }


def _validate_loaded_arrays(
    table: TableSpec,
    arrays: Mapping[str, np.ndarray],
    *,
    path: Path,
) -> None:
    expected = [column.name for column in table.columns]
    if sorted(arrays) != sorted(expected):
        raise GSBError(
            "E_SAMPLE",
            "样本列集合与表声明不一致",
            {"table": table.name, "path": str(path), "expected": sorted(expected), "actual": sorted(arrays)},
        )
    rows = None
    for column in table.columns:
        array = arrays[column.name]
        if array.dtype != column.dtype:
            raise GSBError(
                "E_SAMPLE",
                "样本列类型与表声明不一致",
                {
                    "table": table.name,
                    "column": column.name,
                    "path": str(path),
                    "expected_dtype": str(column.dtype),
                    "actual_dtype": str(array.dtype),
                },
            )
        if array.ndim != 1:
            raise GSBError(
                "E_SAMPLE",
                "样本列必须为一维列式数组",
                {"table": table.name, "column": column.name, "shape": list(array.shape)},
            )
        if rows is None:
            rows = len(array)
        elif len(array) != rows:
            raise GSBError(
                "E_SAMPLE",
                "样本各列长度不一致",
                {"table": table.name, "column": column.name, "rows": len(array), "expected_rows": rows},
            )
    if rows != table.declared_rows:
        raise GSBError(
            "E_SAMPLE",
            "样本行数与声明行数不一致",
            {"table": table.name, "path": str(path), "expected_rows": table.declared_rows, "actual_rows": rows},
        )


def ensure_table_sample(scenario: ScenarioSpec, table_name: str, workdir: str | Path) -> TableSample:
    table = scenario.table(table_name)
    base = Path(workdir)
    npz_path, manifest_path = _sample_paths(base, scenario, table.name)
    expected_arrays = generate_table_arrays(table)
    expected_content = logical_content_digest(
        expected_arrays, names=[column.name for column in table.columns]
    )
    if npz_path.exists():
        manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise GSBError(
                    "E_SAMPLE",
                    "样本摘要清单损坏",
                    {"path": str(manifest_path), "reason": str(exc)},
                ) from exc
            if manifest.get("content_digest") != expected_content:
                raise GSBError(
                    "E_SAMPLE",
                    "样本内容摘要与当前表声明不匹配",
                    {
                        "table": table.name,
                        "path": str(npz_path),
                        "expected_content_digest": expected_content,
                        "actual_content_digest": manifest.get("content_digest"),
                    },
                )
            arrays = read_npz(npz_path, expected_sha256=manifest.get("file_digest"))
        else:
            arrays = read_npz(npz_path)
        _validate_loaded_arrays(table, arrays, path=npz_path)
        actual_content = logical_content_digest(arrays, names=[column.name for column in table.columns])
        if actual_content != expected_content:
            raise GSBError(
                "E_SAMPLE",
                "样本内容摘要与确定性生成结果不一致",
                {
                    "table": table.name,
                    "path": str(npz_path),
                    "expected_content_digest": expected_content,
                    "actual_content_digest": actual_content,
                },
            )
        return TableSample(
            table=table.name,
            columns=arrays,
            content_digest=actual_content,
            file_digest=sha256_file(npz_path),
            path=str(npz_path),
            manifest_path=str(manifest_path),
        )
    file_digest = write_npz_deterministic(npz_path, expected_arrays)
    manifest = _manifest_payload(
        scenario,
        table,
        expected_arrays,
        expected_content,
        file_digest,
    )
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return TableSample(
        table=table.name,
        columns=expected_arrays,
        content_digest=expected_content,
        file_digest=file_digest,
        path=str(npz_path),
        manifest_path=str(manifest_path),
    )


def ensure_scenario_samples(
    scenario: ScenarioSpec,
    workdir: str | Path,
) -> dict[str, TableSample]:
    return {
        name: ensure_table_sample(scenario, name, workdir)
        for name in sorted(scenario.tables)
    }


def scenario_data_digest(samples: Mapping[str, TableSample]) -> str:
    return sha256_json(
        {
            name: {
                "content_digest": sample.content_digest,
                "file_digest": sample.file_digest,
                "rows": sample.rows,
            }
            for name, sample in sorted(samples.items())
        }
    )


def corrupt_sample_file(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise GSBError("E_SAMPLE", "待注入损坏的样本文件不存在", {"path": str(target)})
    original = target.read_bytes()
    damaged = original[: max(0, len(original) // 2)]
    target.write_bytes(damaged)
    return {
        "injection": "sample_corruption",
        "path": str(target),
        "original_bytes": len(original),
        "damaged_bytes": len(damaged),
        "damaged_sha256": sha256_file(target),
    }
