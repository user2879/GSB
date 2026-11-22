from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from .config import WorkbenchConfig
from .diagnostics import environment_info
from .models import ImageInfo
from .runner import ScenarioRun
from .util import atomic_write_json


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _preview_sheet(
    *,
    images: Mapping[str, ImageInfo],
    sample_dir: Path,
    output_dir: Path,
    regions: list[Any],
    target: Path,
) -> None:
    image_dir = sample_dir / "images" if (sample_dir / "images").is_dir() else sample_dir
    tile_size = 256
    label_height = 24
    padding = 10
    sheet = Image.new(
        "RGB",
        (tile_size * 2 + padding * 3, (tile_size + label_height + padding) * len(images) + padding),
        (26, 30, 36),
    )
    draw = ImageDraw.Draw(sheet)
    for row, image_name in enumerate(sorted(images)):
        source = Image.fromarray(_read_rgb(image_dir / image_name), mode="RGB").resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
        output = Image.fromarray(_read_rgb(output_dir / image_name), mode="RGB").resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
        y = padding + row * (tile_size + label_height + padding)
        draw.text((padding, y), f"{image_name} source", fill=(230, 234, 240))
        draw.text((padding * 2 + tile_size, y), f"{image_name} edited", fill=(230, 234, 240))
        source_image = source.copy()
        output_image = output.copy()
        for canvas in (source_image, output_image):
            canvas_draw = ImageDraw.Draw(canvas)
            for region in (item for item in regions if item.image_name == image_name):
                x0 = int(region.rect.x)
                y0 = int(region.rect.y)
                x1 = int(region.rect.x + region.rect.w)
                y1 = int(region.rect.y + region.rect.h)
                canvas_draw.rectangle((x0, y0, x1, y1), outline=(255, 230, 80), width=2)
                canvas_draw.text((x0 + 3, y0 + 3), f"#{region.region_index}", fill=(255, 255, 120))
        sheet.paste(source_image, (padding, y + 18))
        sheet.paste(output_image, (padding * 2 + tile_size, y + 18))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG", compress_level=9, optimize=False)


def _cross_preview(
    *,
    images: Mapping[str, ImageInfo],
    cross: dict[str, Any],
    target: Path,
) -> None:
    v1_dir = Path(cross["output_directories"]["v1"])
    v2_dir = Path(cross["output_directories"]["v2"])
    tile_size = 256
    label_height = 24
    padding = 10
    sheet = Image.new(
        "RGB",
        (tile_size * 3 + padding * 4, (tile_size + label_height + padding) * len(images) + padding),
        (26, 30, 36),
    )
    draw = ImageDraw.Draw(sheet)
    for row, image_name in enumerate(sorted(images)):
        v2 = _read_rgb(v2_dir / image_name)
        v1 = _read_rgb(v1_dir / image_name)
        delta = np.abs(v2.astype(np.int16) - v1.astype(np.int16))
        amplified = np.clip(delta * 4, 0, 255).astype(np.uint8)
        y = padding + row * (tile_size + label_height + padding)
        labels = (f"{image_name} v2", f"{image_name} v1", f"{image_name} diff x4")
        arrays = (v2, v1, amplified)
        for column, (label, array) in enumerate(zip(labels, arrays)):
            x = padding + column * (tile_size + padding)
            draw.text((x, y), label, fill=(230, 234, 240))
            tile = Image.fromarray(array, mode="RGB").resize((tile_size, tile_size), Image.Resampling.NEAREST)
            sheet.paste(tile, (x, y + 18))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG", compress_level=9, optimize=False)


def build_diagnostics(
    *,
    scenario: ScenarioRun,
    cross_version: dict[str, Any],
    injections: list[dict[str, Any]],
    config: WorkbenchConfig,
    images: Mapping[str, ImageInfo],
    sample_dir: Path,
    state_dir: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    command_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_failures: set[tuple[Any, ...]] = set()
    budget_sequences: list[dict[str, Any]] = []
    budget_pending_events: list[dict[str, Any]] = []
    overall_max_active_tasks = 0
    for command in scenario.commands:
        verification = command.verification or {}
        command_rows.append(
            {
                "command_id": command.command_id,
                "op": command.op,
                "exchange_version": command.exchange.version,
                "exchange_fingerprint": command.exchange.exchange_fingerprint,
                "snapshot": command.snapshot,
                "batches": [batch.to_dict() for batch in command.batches],
                "verification": verification,
            }
        )
        for batch in command.batches:
            overall_max_active_tasks = max(overall_max_active_tasks, batch.max_active_tasks)
            if batch.budget_log:
                budget_sequences.append({"command_id": command.command_id, "entries": batch.budget_log})
                for entry in batch.budget_log:
                    if entry.get("action") == "pending":
                        budget_pending_events.append({"command_id": command.command_id, **entry})
        for failure in command.result.failures:
            key = (failure.code, failure.region_key, failure.source, failure.message)
            if key in seen_failures:
                continue
            seen_failures.add(key)
            failures.append({**failure.to_dict(), "command_id": command.command_id})
    for event in budget_pending_events:
        key = (
            event.get("business_code"),
            event.get("region_key"),
            event.get("command_id"),
            event.get("required_pixels"),
            event.get("seq"),
        )
        if key in seen_failures:
            continue
        seen_failures.add(key)
        failures.append(
            {
                "code": event.get("business_code"),
                "stage": "schedule",
                "source": event.get("command_id"),
                "message": (
                    f"pixel budget pending: required={event.get('required_pixels')}, "
                    f"remaining={event.get('remaining_before')}"
                ),
                "region_key": event.get("region_key"),
                "injection_id": None,
                "command_id": event.get("command_id"),
            }
        )
    for failure in scenario.final_result.failures:
        key = (failure.code, failure.region_key, failure.source, failure.message)
        if key not in seen_failures:
            seen_failures.add(key)
            failures.append({**failure.to_dict(), "command_id": scenario.commands[-1].command_id})

    injection_hits: list[dict[str, Any]] = []
    for result in injections:
        injection_hits.append(
            {
                "injection_id": result["injection_id"],
                "source": "injection_config",
                "hit": result["hit"],
                "expected_code": result["expected_code"],
                "observed_codes": result["observed_codes"],
                "valid_region_count": result["valid_region_count"],
                "failure_count": result["failure_count"],
            }
        )
    for command in scenario.commands:
        verification = command.verification or {}
        if verification.get("injection_id"):
            injection_hits.append(
                {
                    "injection_id": verification["injection_id"],
                    "source": command.command_id,
                    "hit": bool(verification.get("injection_hit")),
                    "expected_code": verification.get("expected_code"),
                    "observed_codes": sorted(
                        {failure.code for failure in command.result.failures}
                    ),
                    "valid_region_count": len(command.result.region_records),
                    "failure_count": len(command.result.failures),
                }
            )

    all_semantic = all(
        bool((command.verification or {}).get("semantic_equivalence", False))
        for command in scenario.commands
        if command.verification is not None
    )
    business_codes = {failure["code"] for failure in failures}
    business_codes.update(event["business_code"] for event in budget_pending_events if event.get("business_code"))
    business_codes.update(cross_version.get("boundary_business_codes", []))
    business_codes.update(
        item.get("expected_code", "") for item in injection_hits if not item.get("hit")
    )
    business_codes.discard("")
    latest_manifest_path = state_dir / "CURRENT.json"
    latest_snapshot = None
    if latest_manifest_path.exists():
        try:
            latest_snapshot = json.loads(latest_manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            latest_snapshot = None
    final_stage_metadata = {
        region_key: {
            stage: record.get("stages", {}).get(stage, {}).get("metadata", {})
            for stage in ("parse", "mask", "repair", "attribute", "style", "compose")
            if isinstance(record.get("stages", {}).get(stage), dict)
        }
        for region_key, record in sorted(scenario.final_result.region_records.items())
    }
    return {
        "schema": "image-edit-workbench-diagnostics/v1",
        "environment": environment_info(),
        "sample": {
            "sample_dir": str(sample_dir),
            "image_count": len(images),
            "images": sorted(images),
            "region_count": len(scenario.final_state.regions),
        },
        "config": config.to_dict(),
        "config_hash": config.config_hash,
        "render_config_hash": config.render_config_hash,
        "exchange": {
            "final_version": scenario.final_exchange.version,
            "fingerprint": scenario.final_exchange.exchange_fingerprint,
            "sidecar_hash": scenario.final_exchange.sidecar_hash,
        },
        "run_summary": {
            "command_count": len(scenario.commands),
            "final_content_fingerprint": scenario.final_result.content_fingerprint,
            "final_image_fingerprints": scenario.final_result.image_fingerprints,
            "reused_regions": scenario.final_result.reused_regions,
            "recomputed_regions": scenario.final_result.recomputed_regions,
            "pending_queue": scenario.final_result.pending_queue,
            "max_active_tasks": overall_max_active_tasks,
            "final_batch_active_tasks": scenario.final_result.max_active_tasks,
            "worker_limit": config.thread_pool_workers,
            "affected_pixels": scenario.final_result.affected_pixels,
            "sidecar_restore_hits": cross_version.get("sidecar_restore_hits"),
            "sidecar_restore_misses": cross_version.get("sidecar_restore_misses"),
        },
        "resume": scenario.resume,
        "command_trajectory": command_rows,
        "final_region_trajectory": [trace.to_dict() for trace in scenario.final_result.traces],
        "final_stage_summaries": scenario.final_result.stage_summaries,
        "final_stage_metadata": final_stage_metadata,
        "budget_sequences": budget_sequences,
        "budget_pending_events": budget_pending_events,
        "pending_queue": scenario.final_result.pending_queue,
        "failures": failures,
        "cross_version": cross_version,
        "boundary_samples": cross_version.get("boundary_samples", []),
        "injection_hits": injection_hits,
        "equivalence": {
            "all_commands_semantic_equal": all_semantic,
            "per_command": [
                {
                    "command_id": command.command_id,
                    "semantic_equivalence": bool((command.verification or {}).get("semantic_equivalence", False)),
                    "pixels_equal": bool((command.verification or {}).get("pixels_equal", False)),
                    "stage_summaries_equal": bool((command.verification or {}).get("stage_summaries_equal", False)),
                    "content_fingerprint_equal": bool((command.verification or {}).get("content_fingerprint_equal", False)),
                    "pending_queue_equal": bool((command.verification or {}).get("pending_queue_equal", False)),
                    "raw_reuse_lists_equal": bool((command.verification or {}).get("raw_reuse_lists_equal", False)),
                }
                for command in scenario.commands
            ],
        },
        "snapshots": {
            "history": scenario.snapshot_history,
            "latest_pointer": latest_snapshot,
            "commit_interval_commands": config.snapshot_commit_interval_commands,
        },
        "business_codes": sorted(business_codes),
        "artifacts": {
            "output_dir": str(output_dir),
            "state_dir": str(state_dir),
            "report_dir": str(report_dir),
            "preview": str(report_dir / "preview.png"),
            "cross_version_preview": str(report_dir / "cross_version_preview.png"),
            "diagnostics_json": str(report_dir / "diagnostics.json"),
        },
    }


def _status_class(value: bool) -> str:
    return "ok" if value else "bad"


def write_report(
    *,
    report_dir: Path,
    diagnostics: dict[str, Any],
    scenario: ScenarioRun,
    cross_version: dict[str, Any],
    images: Mapping[str, ImageInfo],
    sample_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    assets = report_dir / "assets"
    source_assets = assets / "source"
    output_assets = assets / "output"
    cross_v1_assets = assets / "cross_v1"
    cross_v2_assets = assets / "cross_v2"
    image_dir = sample_dir / "images" if (sample_dir / "images").is_dir() else sample_dir
    for image_name in sorted(images):
        _copy(image_dir / image_name, source_assets / image_name)
        _copy(output_dir / image_name, output_assets / image_name)
        _copy(Path(cross_version["output_directories"]["v1"]) / image_name, cross_v1_assets / image_name)
        _copy(Path(cross_version["output_directories"]["v2"]) / image_name, cross_v2_assets / image_name)

    _preview_sheet(
        images=images,
        sample_dir=sample_dir,
        output_dir=output_dir,
        regions=scenario.final_state.regions,
        target=report_dir / "preview.png",
    )
    _cross_preview(images=images, cross=cross_version, target=report_dir / "cross_version_preview.png")
    atomic_write_json(report_dir / "diagnostics.json", diagnostics)

    run = diagnostics["run_summary"]
    equivalence = diagnostics["equivalence"]
    cross = diagnostics["cross_version"]
    command_rows = "".join(
        f"<tr><td><code>{html.escape(row['command_id'])}</code></td>"
        f"<td>{html.escape(row['op'])}</td><td>{row['exchange_version']}</td>"
        f"<td class='{_status_class(bool(row['verification'].get('semantic_equivalence', False)))}'>"
        f"{html.escape(str(bool(row['verification'].get('semantic_equivalence', False))))}</td>"
        f"<td>{html.escape(str(row['snapshot']))}</td></tr>"
        for row in diagnostics["command_trajectory"]
    )
    trace_rows = "".join(
        f"<tr><td><code>{html.escape(trace['region_key'])}</code></td>"
        f"<td>{html.escape(trace['status'])}</td><td>{trace['affected_pixels']}</td>"
        f"<td>{html.escape(', '.join(stage['stage'] + ':' + stage['status'] for stage in trace['stages']))}</td>"
        f"<td><code>{html.escape(trace['content_fingerprint'][:16])}</code></td></tr>"
        for trace in diagnostics["final_region_trajectory"]
    )
    cross_rows = "".join(
        f"<tr><td><code>{html.escape(row['region_key'])}</code></td>"
        f"<td>{html.escape(row['coordinate_max_abs_delta'])}</td>"
        f"<td>{html.escape(row['style_abs_delta'])}</td>"
        f"<td>{row['local_pixel_max_delta']}</td>"
        f"<td>{html.escape(', '.join(row['causes']) or '-')}</td>"
        f"<td class='{_status_class(not row['boundary_sensitive'])}'>{row['boundary_sensitive']}</td></tr>"
        for row in cross["regions"]
    )
    failure_rows = "".join(
        f"<tr><td><code>{html.escape(item['code'])}</code></td>"
        f"<td>{html.escape(item.get('command_id', ''))}</td>"
        f"<td>{html.escape(item.get('region_key') or '-')}</td>"
        f"<td>{html.escape(item.get('message', ''))}</td></tr>"
        for item in diagnostics["failures"]
    ) or "<tr><td colspan='4' class='ok'>No failures in the final trajectory.</td></tr>"
    injection_rows = "".join(
        f"<tr><td><code>{html.escape(item['injection_id'])}</code></td>"
        f"<td>{html.escape(item['source'])}</td><td>{html.escape(item['expected_code'])}</td>"
        f"<td class='{_status_class(item['hit'])}'>{item['hit']}</td></tr>"
        for item in diagnostics["injection_hits"]
    )
    image_cards = "".join(
        f"<section class='image-card'><h3>{html.escape(name)}</h3>"
        f"<div class='image-grid'>"
        f"<figure><img src='assets/source/{html.escape(name)}'><figcaption>source</figcaption></figure>"
        f"<figure><img src='assets/output/{html.escape(name)}'><figcaption>incremental output</figcaption></figure>"
        f"<figure><img src='assets/cross_v2/{html.escape(name)}'><figcaption>v2 render</figcaption></figure>"
        f"<figure><img src='assets/cross_v1/{html.escape(name)}'><figcaption>v1 degraded render</figcaption></figure>"
        f"</div></section>"
        for name in diagnostics["sample"]["images"]
    )
    page = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Image Edit Workbench Diagnostics</title>
<style>
:root {{ color-scheme: dark; --bg:#171b21; --panel:#222933; --line:#384250; --ink:#e8edf4; --muted:#9eabbb; --ok:#78d59a; --bad:#ff8b8b; --accent:#8bc8ff; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }}
main {{ max-width:1500px; margin:0 auto; padding:28px; }}
h1,h2,h3 {{ font-weight:650; letter-spacing:.01em; }}
h1 {{ margin:0 0 8px; }} h2 {{ margin-top:36px; border-bottom:1px solid var(--line); padding-bottom:8px; }}
.muted {{ color:var(--muted); }} .ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:22px 0; }}
.card,.image-card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }}
.card b {{ display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }}
th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
code {{ color:var(--accent); }}
.image-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }}
figure {{ margin:0; }} figure img {{ width:100%; image-rendering:pixelated; border:1px solid var(--line); border-radius:6px; }}
figcaption {{ color:var(--muted); margin-top:5px; }}
.wide img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; }}
a {{ color:var(--accent); }}
</style>
</head>
<body><main>
<h1>Deterministic Image Edit Interop Workbench</h1>
<p class='muted'>Offline v1/v2 exchange diagnostics, fixed-point rendering, region-level incremental reuse, snapshot recovery, and cross-version boundary attribution.</p>
<div class='cards'>
<div class='card'><b>Final content fingerprint</b><code>{html.escape(run['final_content_fingerprint'])}</code></div>
<div class='card'><b>Semantic equivalence</b><span class='{_status_class(equivalence['all_commands_semantic_equal'])}'>{equivalence['all_commands_semantic_equal']}</span></div>
<div class='card'><b>Recomputed / reused (last command)</b>{len(run['recomputed_regions'])} / {len(run['reused_regions'])}</div>
<div class='card'><b>Pending queue</b>{len(run['pending_queue'])}</div>
<div class='card'><b>Max active tasks / limit</b>{run['max_active_tasks']} / {run['worker_limit']}</div>
<div class='card'><b>Sidecar restore hits</b>{run['sidecar_restore_hits']}</div>
<div class='card'><b>Style max deviation</b>{html.escape(str(cross['style_strength_max_abs_delta']))}</div>
<div class='card'><b>Boundary samples</b>{len(diagnostics['boundary_samples'])}</div>
</div>
<h2>Visual Review</h2>
<div class='wide'><img src='preview.png' alt='source and output preview'></div>
<p><a href='diagnostics.json'>diagnostics.json</a> · <a href='cross_version_preview.png'>cross-version diff preview</a></p>
<div class='wide'><img src='cross_version_preview.png' alt='cross version preview'></div>
{image_cards}
<h2>Command Trajectory</h2>
<table><thead><tr><th>Command</th><th>Op</th><th>Version</th><th>Incremental = Full</th><th>Snapshot</th></tr></thead><tbody>{command_rows}</tbody></table>
<h2>Final Region Trace</h2>
<table><thead><tr><th>Region</th><th>Status</th><th>Affected px</th><th>Stages</th><th>Content fp</th></tr></thead><tbody>{trace_rows}</tbody></table>
<h2>Cross-Version Regions</h2>
<table><thead><tr><th>Region</th><th>Coord Δ max</th><th>Style Δ</th><th>Local pixel Δ</th><th>Attribution</th><th>Boundary sensitive</th></tr></thead><tbody>{cross_rows}</tbody></table>
<h2>Failures</h2>
<table><thead><tr><th>Code</th><th>Command</th><th>Region</th><th>Message</th></tr></thead><tbody>{failure_rows}</tbody></table>
<h2>Injection Hits</h2>
<table><thead><tr><th>Injection ID</th><th>Source</th><th>Expected code</th><th>Hit</th></tr></thead><tbody>{injection_rows}</tbody></table>
<h2>Environment</h2>
<p class='muted'>{html.escape(json.dumps(diagnostics['environment'], ensure_ascii=False, sort_keys=True))}</p>
</main></body></html>
"""
    (report_dir / "index.html").write_text(page, encoding="utf-8")
    return {
        "index_html": str(report_dir / "index.html"),
        "diagnostics_json": str(report_dir / "diagnostics.json"),
        "preview_png": str(report_dir / "preview.png"),
        "cross_version_preview_png": str(report_dir / "cross_version_preview.png"),
    }
