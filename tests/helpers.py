from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from streamparse.app import IncrementalApp
from streamparse.engine import full_decode, semantic_payload
from streamparse.loader import load_config
from streamparse.models import DecodeConfig
from streamparse.samplegen import write_sample


def make_sample(tmp_path: Path) -> Path:
    sample_dir = tmp_path / "sample_data"
    write_sample(sample_dir)
    return sample_dir


def make_config(
    sample_dir: Path,
    tmp_path: Path,
    **overrides: Any,
) -> DecodeConfig:
    config = load_config(sample_dir / "decode_config.json")
    values: Dict[str, Any] = {
        "snapshot_root": str(tmp_path / "snapshots"),
        "report_path": str(tmp_path / "diagnostic_report.json"),
        "response_budget_ms": 5000,
    }
    values.update(overrides)
    return replace(config, **values)


def make_app(tmp_path: Path, **overrides: Any) -> IncrementalApp:
    sample_dir = make_sample(tmp_path)
    config = make_config(sample_dir, tmp_path, **overrides)
    return IncrementalApp(sample_dir, workspace_root=tmp_path, config=config)


def canonical_full(app: IncrementalApp) -> Dict[str, Any]:
    return full_decode(
        app.dataset,
        app.config,
        overrides=app.overrides,
        injections=app.injections,
        version=app.version,
        initial_failures=app.load_failures,
    )


def assert_incremental_matches_full(testcase: Any, app: IncrementalApp) -> None:
    app.execute({"cmd": "wait"})
    full = canonical_full(app)
    testcase.assertEqual(semantic_payload(app.latest_report), semantic_payload(full))
    testcase.assertEqual(app.latest_version, app.version)
