"""Automated incremental-vs-full equivalence verification."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from .cache import atomic_write_json
from .config import Config, with_overrides
from .io_layer import discover_pngs
from .pipeline import Pipeline, semantic_digest
from .sample_generator import generate_samples


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_inputs(source: Path, target: Path) -> None:
    _reset_dir(target)
    for item in sorted(source.iterdir()):
        if item.is_file() and item.suffix.lower() in {".png", ".json"}:
            shutil.copy2(item, target / item.name)


def _replace_image(input_dir: Path, target_name: str, replacement_name: str) -> None:
    target_png = input_dir / target_name
    replacement_png = input_dir / replacement_name
    if not target_png.exists() or not replacement_png.exists():
        raise FileNotFoundError("替换图像不存在")
    shutil.copy2(replacement_png, target_png)
    # Mutate one light background pixel so the replacement has a genuinely new
    # content fingerprint while remaining visually and topologically equivalent.
    with Image.open(target_png) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    if pixels.size:
        pixels[0, 0] = 244 if int(pixels[0, 0]) != 244 else 245
    Image.fromarray(pixels, mode="L").save(target_png, format="PNG", optimize=False)
    target_truth = input_dir / (Path(target_name).stem + ".truth.json")
    replacement_truth = input_dir / (Path(replacement_name).stem + ".truth.json")
    raw = json.loads(replacement_truth.read_text(encoding="utf-8"))
    raw["image"] = target_name
    target_truth.write_text(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: 类型不同 {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: 键集合不同 {sorted(left)} != {sorted(right)}"
        for key in left:
            diff = _first_difference(left[key], right[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: 数组长度不同 {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            diff = _first_difference(a, b, f"{path}[{index}]")
            if diff:
                return diff
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def _trace_actions(report: dict[str, Any]) -> dict[str, Any]:
    trace = report["cache"]["reuse_recompute"]
    return {
        "columns": [
            (entry["image"], entry["column"], entry["action"])
            for entry in trace["columns"]
            if entry["stage"] == "columns"
        ],
        "segments": [
            (entry["image"], entry["column"], entry["action"])
            for entry in trace["columns"]
            if entry["stage"] == "segments"
        ],
        "regression": [
            (entry["image"], entry["column"], entry["char"], entry["action"])
            for entry in trace["characters"]
            if entry["stage"] == "regression"
        ],
        "match": [
            (entry["image"], entry["column"], entry["char"], entry["action"])
            for entry in trace["characters"]
            if entry["stage"] == "template_match"
        ],
    }


def verify_incremental_equivalence(
    samples_dir: str | Path,
    config: Config,
    workdir: str | Path,
    *,
    regenerate_samples: bool = False,
) -> dict[str, Any]:
    source = Path(samples_dir)
    if regenerate_samples:
        generate_samples(source)
    work = Path(workdir)
    _reset_dir(work)
    inputs = work / "inputs"
    _copy_inputs(source, inputs)
    incremental_state = work / "state_incremental"
    _reset_dir(incremental_state)

    initial = with_overrides(
        config,
        binarization_threshold=150,
        char_min_height=14,
        char_max_height=31,
        split_penalty=3,
        match_metric="l1",
    )
    steps: list[dict[str, Any]] = [
        {
            "name": "initial",
            "config": initial,
            "description": "初始非最终配置，建立增量缓存",
        },
        {
            "name": "threshold_edit",
            "config": with_overrides(initial, binarization_threshold=config.binarization_threshold),
            "description": "仅调整二值化阈值",
        },
        {
            "name": "height_edit",
            "config": with_overrides(
                with_overrides(initial, binarization_threshold=config.binarization_threshold),
                char_min_height=config.char_min_height,
                char_max_height=config.char_max_height,
            ),
            "description": "仅恢复最小/最大字符高度约束",
        },
        {
            "name": "penalty_edit",
            "config": with_overrides(
                with_overrides(
                    with_overrides(initial, binarization_threshold=config.binarization_threshold),
                    char_min_height=config.char_min_height,
                    char_max_height=config.char_max_height,
                ),
                split_penalty=config.split_penalty,
            ),
            "description": "仅调整粘连块分割罚项",
        },
        {
            "name": "metric_edit",
            "config": with_overrides(
                with_overrides(
                    with_overrides(
                        with_overrides(initial, binarization_threshold=config.binarization_threshold),
                        char_min_height=config.char_min_height,
                        char_max_height=config.char_max_height,
                    ),
                    split_penalty=config.split_penalty,
                ),
                match_metric=config.match_metric,
            ),
            "description": "仅切换模板匹配度量，达到最终配置",
        },
        {
            "name": "image_replace",
            "config": config,
            "description": "用另一张同分辨率样本整体替换 page_001 图像与真值",
            "replace": ("page_001.png", "page_003.png"),
        },
    ]

    results: list[dict[str, Any]] = []
    incremental_report: dict[str, Any] | None = None
    for index, step in enumerate(steps, start=1):
        if step.get("replace"):
            target, replacement = step["replace"]
            _replace_image(inputs, target, replacement)
        inc_pipeline = Pipeline(inputs, step["config"], incremental_state)
        incremental_report = inc_pipeline.run()
        full_state = work / "state_full" / f"step_{index:02d}"
        _reset_dir(full_state)
        full_report = Pipeline(inputs, step["config"], full_state).run(full_rebuild=True)
        inc_semantic = incremental_report.get("images", [])
        full_semantic = full_report.get("images", [])
        # Compare the canonical semantic digest and pinpoint the first difference if any.
        digest_equal = incremental_report["semantic_digest"] == full_report["semantic_digest"]
        diff = None if digest_equal else _first_difference(inc_semantic, full_semantic)
        actions = _trace_actions(incremental_report)
        results.append(
            {
                "step": index,
                "name": step["name"],
                "description": step["description"],
                "config_hash": step["config"].fingerprint(),
                "incremental_semantic_digest": incremental_report["semantic_digest"],
                "full_semantic_digest": full_report["semantic_digest"],
                "semantic_equal": digest_equal,
                "first_difference": diff,
                "incremental_global_version": incremental_report["run"]["global_version"],
                "snapshot_stale_reason": incremental_report["run"]["snapshot_stale_reason"],
                "cache_actions": {
                    key: {
                        action: sum(1 for entry in entries if entry[-1] == action)
                        for action in ("reuse", "recompute", "pending")
                    }
                    for key, entries in actions.items()
                },
            }
        )

    # A no-op rerun must reuse every stage product and preserve semantics.
    noop_report = Pipeline(inputs, config, incremental_state).run()
    noop_actions = _trace_actions(noop_report)
    noop_all_reused = all(entry[-1] == "reuse" for entries in noop_actions.values() for entry in entries)
    if incremental_report is None:
        raise RuntimeError("验证序列未产生运行报告")

    checks = {
        "all_step_semantics_equal": all(item["semantic_equal"] for item in results),
        "noop_all_cache_reused": noop_all_reused,
        "noop_semantics_equal": noop_report["semantic_digest"]
        == results[-1]["incremental_semantic_digest"],
        "final_config_matches": config.fingerprint() == results[-1]["config_hash"],
    }
    # Check the targeted invalidation contract on the isolated edit steps.
    penalty_step = next(item for item in results if item["name"] == "penalty_edit")
    metric_step = next(item for item in results if item["name"] == "metric_edit")
    height_step = next(item for item in results if item["name"] == "height_edit")
    checks["penalty_reuses_columns_and_segments"] = (
        penalty_step["cache_actions"]["columns"].get("recompute", 0) == 0
        and penalty_step["cache_actions"]["segments"].get("recompute", 0) == 0
    )
    checks["height_reuses_columns_and_segments"] = (
        height_step["cache_actions"]["columns"].get("recompute", 0) == 0
        and height_step["cache_actions"]["segments"].get("recompute", 0) == 0
    )
    checks["metric_reuses_regression"] = metric_step["cache_actions"]["regression"].get("recompute", 0) == 0
    checks["metric_recomputes_matches"] = metric_step["cache_actions"]["match"].get("recompute", 0) > 0
    checks["penalty_only_recomputes_oversize_regressions"] = (
        0
        < penalty_step["cache_actions"]["regression"].get("recompute", 0)
        < penalty_step["cache_actions"]["regression"].get("reuse", 0)
    )
    checks["noop_snapshot_stale"] = noop_report["snapshot"]["stale"] is False
    replacement_step = next(item for item in results if item["name"] == "image_replace")
    replacement_actions = replacement_step["cache_actions"]
    checks["replacement_invalidates_changed_image"] = (
        replacement_actions["columns"].get("recompute", 0) > 0
    )
    checks["replacement_reuses_unaffected_images"] = (
        replacement_actions["columns"].get("reuse", 0) > 0
    )
    passed = all(bool(value) for value in checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "steps": results,
        "final_semantic_digest": results[-1]["incremental_semantic_digest"],
        "noop": {
            "semantic_digest": noop_report["semantic_digest"],
            "all_cache_reused": noop_all_reused,
            "cache_summary": noop_report["cache"]["summary"],
        },
        "workdir": str(work),
        "inputs": str(inputs),
    }
