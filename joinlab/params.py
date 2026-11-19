from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .canonical import digest_json
from .errors import fail


QUANTILE_POLICIES = ("nearest_rank", "lower_rank")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class AnalysisParams:
    bucket_count: int
    sample_size: int
    seed: int
    quantile_policy: str = "nearest_rank"
    attribution_k: int = 5

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket_count": self.bucket_count,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "quantile_policy": self.quantile_policy,
            "attribution_k": self.attribution_k,
        }

    def digest(self) -> str:
        return digest_json(self.as_dict())


def normalise_params(
    raw: Mapping[str, Any] | None,
    defaults: Mapping[str, Any] | None = None,
    k_default: int = 5,
) -> AnalysisParams:
    merged: dict[str, Any] = dict(defaults or {})
    if raw:
        merged.update(raw)
    bucket_count = merged.get("bucket_count")
    if not _is_int(bucket_count) or bucket_count <= 0:
        fail("E_HIST", "bucket_count 必须是正整数", "params.bucket_count", value=bucket_count)
    sample_size = merged.get("sample_size")
    if not _is_int(sample_size) or sample_size <= 0:
        fail("E_SAMPLE", "sample_size 必须是正整数", "params.sample_size", value=sample_size)
    seed = merged.get("seed")
    if not _is_int(seed):
        fail("E_SAMPLE", "seed 必须是整数", "params.seed", value=seed)
    quantile_policy = merged.get("quantile_policy", "nearest_rank")
    if not isinstance(quantile_policy, str) or quantile_policy not in QUANTILE_POLICIES:
        fail(
            "E_HIST",
            "quantile_policy 不受支持",
            "params.quantile_policy",
            value=quantile_policy,
            allowed=list(QUANTILE_POLICIES),
        )
    attribution_k = merged.get("attribution_k", k_default)
    if not _is_int(attribution_k) or attribution_k <= 0:
        fail("E_TABLE", "attribution_k 必须是正整数", "params.attribution_k", value=attribution_k)
    return AnalysisParams(
        bucket_count=bucket_count,
        sample_size=sample_size,
        seed=seed,
        quantile_policy=quantile_policy,
        attribution_k=attribution_k,
    )


def load_params_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        fail("E_TABLE", "参数文件不存在", "session", file=str(source))
    except json.JSONDecodeError as exc:
        fail("E_TABLE", "参数文件 JSON 无法解析", "session", line=exc.lineno, column=exc.colno)
    except OSError as exc:
        fail("E_TABLE", "参数文件无法读取", "session", reason=str(exc))
    if not isinstance(document, dict):
        fail("E_TABLE", "参数文件顶层必须是对象", "session")
    if "params" in document:
        nested = document["params"]
        if not isinstance(nested, dict):
            fail("E_TABLE", "session.params 必须是对象", "session.params")
        result = dict(nested)
        if "attribution_k" in document and "attribution_k" not in result:
            result["attribution_k"] = document["attribution_k"]
        return result
    return dict(document)


def with_overrides(params: AnalysisParams, overrides: Mapping[str, Any] | None) -> AnalysisParams:
    clean = {key: value for key, value in (overrides or {}).items() if value is not None}
    if not clean:
        return params
    return replace(params, **clean)

