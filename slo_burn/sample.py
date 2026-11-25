"""固定整数种子的本地确定性样本生成（不访问网络）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261126


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def generate_workspace(root: str | Path) -> Path:
    root = Path(root)
    batch_dir = root / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "service_id": "checkout-api",
            "target_permyriad": 9990,
            "window_minutes": 30,
            "burn_threshold": 20000,
            "budget_floor": 0,
        },
        {
            "service_id": "auth-service",
            "target_permyriad": 9950,
            "window_minutes": 90,
            "burn_threshold": 15000,
            "budget_floor": 1000,
        },
        {
            "service_id": "search-edge",
            "target_permyriad": 9900,
            "window_minutes": 120,
            "burn_threshold": 30000,
            "budget_floor": 500,
        },
    ]
    (root / "services.json").write_text(
        json.dumps({"seed": SEED, "services": specs}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rng = _rng()
    regular_lines: list[str] = []
    late_lines: list[str] = []
    horizon = 150
    profiles = {
        "checkout-api": (180, 0.01),
        "auth-service": (240, 0.03),
        "search-edge": (320, 0.06),
    }
    for spec in specs:
        service_id = spec["service_id"]
        traffic, fail_rate = profiles[service_id]
        for minute in range(horizon):
            successes = int(max(rng.poisson(traffic), 1))
            failures = int(rng.binomial(successes, fail_rate))
            regular_lines.append(
                json.dumps(
                    {
                        "service_id": service_id,
                        "minute": minute,
                        "successes": successes,
                        "failures": failures,
                    },
                    ensure_ascii=False,
                )
            )
        # 迟到批：携带更早分钟（窗口归属需重算）
        late_carriers = [12, 47, 88, 131]
        for minute in late_carriers:
            successes = int(max(rng.poisson(traffic // 3), 1))
            failures = int(rng.binomial(successes, fail_rate * 4))
            late_lines.append(
                json.dumps(
                    {
                        "service_id": service_id,
                        "minute": minute,
                        "successes": successes,
                        "failures": failures,
                    },
                    ensure_ascii=False,
                )
            )
        # 重复分钟重发：与 late 中 minute=47 冲突，验证稳定择一
        duplicate = {
            "service_id": service_id,
            "minute": 47,
            "successes": 7,
            "failures": 9,
        }
        late_lines.append(json.dumps(duplicate, ensure_ascii=False))

    (batch_dir / "regular.ndjson").write_text("\n".join(regular_lines) + "\n", encoding="utf-8")
    (batch_dir / "late.ndjson").write_text("\n".join(late_lines) + "\n", encoding="utf-8")

    edits = {
        "edits": [
            {
                "service_id": "auth-service",
                "param_version": 2,
                "effective_after_minute": 60,
                "target_permyriad": 9970,
                "burn_threshold": 12000,
            },
            {
                "service_id": "auth-service",
                "param_version": 3,
                "effective_after_minute": 120,
                "window_minutes": 60,
            },
            {
                "service_id": "search-edge",
                "param_version": 2,
                "effective_after_minute": 90,
                "budget_floor": 900,
                "burn_threshold": 25000,
            },
        ]
    }
    (root / "param_edits.json").write_text(
        json.dumps(edits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    invalid_dir = root / "invalid"
    invalid_dir.mkdir(exist_ok=True)
    (invalid_dir / "bad_services.json").write_text(
        json.dumps(
            {
                "services": [
                    specs[0],
                    {
                        "service_id": "broken-svc",
                        "target_permyriad": 12000,
                        "window_minutes": 10,
                        "burn_threshold": 10000,
                        "budget_floor": 0,
                    },
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bad_batches = [
        {"service_id": "checkout-api", "minute": 0, "successes": 10, "failures": 1},
        {"service_id": "checkout-api", "minute": 0, "successes": 11, "failures": 2},
    ]
    (invalid_dir / "bad_batches.ndjson").write_text(
        "\n".join(json.dumps(row) for row in bad_batches) + "\n", encoding="utf-8"
    )
    return root
