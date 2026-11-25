"""独立重放校验：只读取账本与原始声明，重推全部窗口与告警结论。"""

from __future__ import annotations

from pathlib import Path

from .budget import run_exact
from .errors import ReplayError
from .ledger import Ledger
from .loader import load_batch_directory, load_service_specs
from .models import MinutePoint, WindowConclusion
from .params import build_version_chain, spec_at_minute
from .stream import merge_streams


def _conclusion_key(item: WindowConclusion | dict) -> tuple:
    if isinstance(item, dict):
        return (
            item["service_id"],
            item["minute"],
            item["remaining_budget_scaled"],
            item["burn_rate_num"],
            item["burn_rate_den"],
            int(item["alert"]),
        )
    return (
        item.service_id,
        item.minute,
        item.remaining_budget_scaled,
        item.burn_rate_num,
        item.burn_rate_den,
        int(item.alert),
    )


def replay_from_sources(
    specs_path: str | Path,
    batch_dir: str | Path,
    edits_path: str | Path | None,
) -> list[WindowConclusion]:
    """只读原始声明与批次，按精确路径重推权威结论。"""
    specs = load_service_specs(specs_path)
    regular, late = load_batch_directory(batch_dir, specs)
    merged, _ = merge_streams(regular, late)
    chains = None
    if edits_path is not None and Path(edits_path).exists():
        from .loader import load_param_edits

        edits = load_param_edits(edits_path, specs)
        chains = build_version_chain(specs, edits)

    conclusions: list[WindowConclusion] = []
    for spec in sorted(specs, key=lambda item: item.service_id):
        points = merged.get(spec.service_id, [])
        if chains is not None:
            chain = chains[spec.service_id]
            by_version: dict[int, list[MinutePoint]] = {}
            for point in points:
                versioned = spec_at_minute(chain, point.minute)
                by_version.setdefault(versioned.spec.param_version, []).append(point)
            for _version, version_points in sorted(by_version.items()):
                versioned = spec_at_minute(chain, version_points[0].minute)
                conclusions.extend(run_exact(versioned.spec, version_points))
        else:
            conclusions.extend(run_exact(spec, points))
    # 与账本展示口径一致：每服务每分钟仅保留当前（最新参数版本）结论，
    # 旧版本结论在账本中整体标记陈旧且不参与展示/重放比对。
    latest_by_key: dict[tuple[str, int], WindowConclusion] = {}
    for item in conclusions:
        key = (item.service_id, item.minute)
        incumbent = latest_by_key.get(key)
        if incumbent is None or item.param_version > incumbent.param_version:
            latest_by_key[key] = item
    return [latest_by_key[key] for key in sorted(latest_by_key)]


def verify_ledger(
    ledger_path: str | Path,
    specs_path: str | Path,
    batch_dir: str | Path,
    edits_path: str | Path | None = None,
) -> dict:
    """独立校验：账本活跃结论必须与原始输入重推结果逐字段一致。"""
    expected = replay_from_sources(specs_path, batch_dir, edits_path)
    with Ledger(ledger_path) as ledger:
        ledger.verify_format()
        stored = ledger.active_conclusions()

    expected_keys = [_conclusion_key(item) for item in expected]
    stored_keys = [
        (
            item["service_id"],
            item["minute"],
            item["remaining_budget_scaled"],
            item["burn_rate_num"],
            item["burn_rate_den"],
            int(item["alert"]),
        )
        for item in stored
    ]
    if len(expected_keys) != len(stored_keys):
        first = expected_keys[len(stored_keys)] if len(expected_keys) > len(stored_keys) else stored_keys[len(expected_keys)]
        raise ReplayError(
            "重放结论数量与账本不一致",
            service_id=first[0],
            minute=first[1],
            field="row-count",
            expected=len(expected_keys),
            stored=len(stored_keys),
        )
    for index, (expected_key, stored_key) in enumerate(zip(expected_keys, stored_keys)):
        if expected_key != stored_key:
            field_name = _first_diff_field(expected_key, stored_key)
            raise ReplayError(
                "账本重放校验失败",
                service_id=expected_key[0],
                minute=expected_key[1],
                field=field_name,
                expected=expected_key,
                stored=stored_key,
                row_index=index,
            )
    return {
        "ok": True,
        "rows": len(stored_keys),
        "last_commit": Ledger(ledger_path).latest_commit(),
    }


def _first_diff_field(expected_key: tuple, stored_key: tuple) -> str:
    names = (
        "service_id",
        "minute",
        "remaining_budget_scaled",
        "burn_rate_num",
        "burn_rate_den",
        "alert",
    )
    for name, left, right in zip(names, expected_key, stored_key):
        if left != right:
            return name
    return "unknown"
