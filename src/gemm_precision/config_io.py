from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, BusinessError
from .model import (
    AccuracyTolerance,
    BlockShape,
    ExperimentConfig,
    InjectionSpec,
    LoopOrder,
    SampleSpec,
)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError(
            code=E_CONFIG_INVALID,
            stage="config_load",
            message=f"Cannot load experiment config: {source}",
            details={"error": str(exc)},
        ) from exc
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    try:
        samples = tuple(
            SampleSpec(
                name=str(item["name"]),
                category=str(item["category"]),
                seed=int(item["seed"]),
                M=int(item["M"]),
                N=int(item["N"]),
                K=int(item["K"]),
                batch_size=int(item.get("batch_size", 1)),
                batched=item.get("batched"),
                alpha=float(item.get("alpha", 1.0)),
                beta=float(item.get("beta", 0.0)),
                special_fraction=float(item.get("special_fraction", 0.0)),
                description=str(item.get("description", "")),
            )
            for item in raw["samples"]
        )
        blocks = tuple(BlockShape.from_any(item) for item in raw["block_shapes"])
        orders = tuple(str(item) for item in raw["loop_orders"])
        tolerance_raw = raw["tolerance"]
        injections = tuple(
            InjectionSpec(
                injection_id=str(item["injection_id"]),
                kind=str(item["kind"]),  # validated by ExperimentConfig/injection runner
                sample=str(item["sample"]),
                target=item.get("target"),
                index=tuple(item["index"]) if item.get("index") is not None else None,
                value=item.get("value"),
                block_shape=tuple(item["block_shape"]) if item.get("block_shape") else None,
                atol=item.get("atol"),
                rtol=item.get("rtol"),
                expected_code=str(item.get("expected_code", "")),
            )
            for item in raw.get("injections", [])
        )
        config = ExperimentConfig(
            seed=int(raw.get("seed", 0)),
            samples=samples,
            block_shapes=blocks,
            loop_orders=orders,  # type: ignore[arg-type]
            tolerance=AccuracyTolerance(float(tolerance_raw["atol"]), float(tolerance_raw["rtol"])),
            layer_exceed_limit=float(raw.get("layer_exceed_limit", 0.05)),
            layer_count=int(raw.get("layer_count", 4)),
            top_k_contributors=int(raw.get("top_k_contributors", 5)),
            max_workers=int(raw.get("max_workers", 4)),
            serial_workers=int(raw.get("serial_workers", 1)),
            report_path=str(raw.get("report_path", "artifacts/precision_report.json")),
            injections=injections,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(
            code=E_CONFIG_INVALID,
            stage="config_parse",
            message="Experiment configuration is missing a required field or has a bad type.",
            details={"error": str(exc)},
        ) from exc
    config.validate()
    return config


def default_config_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "configs" / "default_experiment.json"
