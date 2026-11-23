from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ErrorCode, Failure
from .util import is_finite_number, is_int, read_json, sha256_json


@dataclass(frozen=True)
class PairRange:
    start_ratio: float
    end_ratio: float


@dataclass(frozen=True)
class PruningPolicy:
    policy_id: str
    description: str
    pair_ranges: tuple[PairRange, ...]
    pair_stride: int

    def resolve_dimensions(self, dimension: int) -> tuple[int, ...]:
        pair_count = dimension // 2
        selected_pairs: set[int] = set()
        for item in self.pair_ranges:
            start = int(item.start_ratio * pair_count)
            end = int(item.end_ratio * pair_count)
            if end <= start and pair_count > 0:
                end = min(pair_count, start + 1)
            selected_pairs.update(range(max(0, start), min(pair_count, end), self.pair_stride))
        if not selected_pairs:
            selected_pairs.add(0)
        dimensions: list[int] = []
        for pair in sorted(selected_pairs):
            dimensions.extend((pair * 2, pair * 2 + 1))
        return tuple(dimensions)

    def fingerprint(self, dimension: int) -> str:
        return sha256_json(
            {
                "policy_id": self.policy_id,
                "pair_ranges": [
                    {"start_ratio": item.start_ratio, "end_ratio": item.end_ratio}
                    for item in self.pair_ranges
                ],
                "pair_stride": self.pair_stride,
                "dimension": dimension,
                "resolved_dimensions": list(self.resolve_dimensions(dimension)),
            }
        )


def _failure(message: str, details: dict[str, Any] | None = None) -> Failure:
    return Failure(
        code=ErrorCode.CONFIG_INVALID,
        message=f"pruning_strategies: {message}",
        stage="pruning",
        details=details or {},
    )


def parse_pruning_policies(raw: Any) -> tuple[list[PruningPolicy], list[Failure]]:
    failures: list[Failure] = []
    if isinstance(raw, dict):
        raw = raw.get("policies")
    if not isinstance(raw, list) or not raw:
        return [], [_failure("必须为非空数组或含 policies 的对象")]

    policies: list[PruningPolicy] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            failures.append(_failure(f"policies[{index}] 必须为对象"))
            continue
        policy_id = item.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id.strip():
            failures.append(_failure(f"policies[{index}].policy_id 必须为非空字符串"))
            continue
        policy_id = policy_id.strip()
        if policy_id in seen:
            failures.append(_failure(f"policy_id 重复: {policy_id}"))
            continue
        seen.add(policy_id)
        description = item.get("description", "")
        if not isinstance(description, str):
            failures.append(_failure(f"policies[{index}].description 必须为字符串"))
            continue
        ranges_raw = item.get("pair_ranges")
        if not isinstance(ranges_raw, list) or not ranges_raw:
            failures.append(_failure(f"policies[{index}].pair_ranges 必须为非空数组"))
            continue
        ranges: list[PairRange] = []
        ranges_valid = True
        for range_index, range_item in enumerate(ranges_raw):
            if not isinstance(range_item, dict):
                failures.append(_failure(f"policies[{index}].pair_ranges[{range_index}] 必须为对象"))
                ranges_valid = False
                continue
            start = range_item.get("start_ratio")
            end = range_item.get("end_ratio")
            if (
                not is_finite_number(start)
                or not is_finite_number(end)
                or float(start) < 0.0
                or float(end) > 1.0
                or float(start) >= float(end)
            ):
                failures.append(
                    _failure(
                        f"policies[{index}].pair_ranges[{range_index}] 需满足 0 <= start < end <= 1"
                    )
                )
                ranges_valid = False
                continue
            ranges.append(PairRange(float(start), float(end)))
        stride = item.get("pair_stride", 1)
        if not is_int(stride) or stride <= 0:
            failures.append(_failure(f"policies[{index}].pair_stride 必须为正整数"))
            ranges_valid = False
        if ranges_valid:
            policies.append(
                PruningPolicy(
                    policy_id=policy_id,
                    description=description,
                    pair_ranges=tuple(ranges),
                    pair_stride=int(stride),
                )
            )
    return policies, failures


def load_pruning_policies(path: Path) -> tuple[list[PruningPolicy], list[Failure], str]:
    try:
        raw = read_json(path)
        digest = sha256_json(
            {"content": raw, "schema": "rope-diag-pruning-v1", "path": path.as_posix()}
        )
    except (OSError, ValueError) as exc:
        failure = Failure(
            code=ErrorCode.IO_ERROR,
            message=f"无法读取剪枝策略: {exc}",
            stage="pruning",
            details={"path": path.as_posix()},
        )
        return [], [failure], ""
    policies, failures = parse_pruning_policies(raw)
    return policies, failures, digest


def find_policy(policies: list[PruningPolicy], policy_id: str) -> PruningPolicy | None:
    for policy in policies:
        if policy.policy_id == policy_id:
            return policy
    return None


def policies_public(policies: list[PruningPolicy], dimension: int) -> list[dict[str, Any]]:
    return [
        {
            "policy_id": policy.policy_id,
            "description": policy.description,
            "resolved_dimensions": list(policy.resolve_dimensions(dimension)),
            "fingerprint": policy.fingerprint(dimension),
        }
        for policy in sorted(policies, key=lambda item: item.policy_id)
    ]
