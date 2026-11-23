from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from config_arbiter.engine import ArbiterEngine
from config_arbiter.ledger import Ledger
from config_arbiter.spec import Specs, load_specs
from config_arbiter.util import atomic_write_json, canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ENV = ROOT / "samples" / "environments.json"
SAMPLE_VIEWS = ROOT / "samples" / "views.json"


def copy_samples(destination: Path) -> tuple[Path, Path]:
    environment_path = destination / "environments.json"
    views_path = destination / "views.json"
    shutil.copyfile(SAMPLE_ENV, environment_path)
    shutil.copyfile(SAMPLE_VIEWS, views_path)
    return environment_path, views_path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def loaded(destination: Path, *, reverse_order: bool = False) -> tuple[Path, Path, Specs]:
    environment_path, views_path = copy_samples(destination)
    if reverse_order:
        environment_document = read_json(environment_path)
        environment_document["environments"].reverse()
        for environment in environment_document["environments"]:
            environment["available_sources"].reverse()
            environment["sources"].reverse()
            for source in environment["sources"]:
                source["observations"].reverse()
        view_document = read_json(views_path)
        view_document["views"].reverse()
        for view in view_document["views"]:
            view["environments"].reverse()
            view["key_policies"] = dict(reversed(list(view["key_policies"].items())))
        write_json(environment_path, environment_document)
        write_json(views_path, view_document)
    return environment_path, views_path, load_specs(environment_path, views_path)


def build_engine(destination: Path, *, reverse_order: bool = False) -> tuple[ArbiterEngine, Ledger, Specs, Path, Path]:
    environment_path, views_path, specs = loaded(destination, reverse_order=reverse_order)
    ledger = Ledger(destination / "ledger.sqlite")
    return ArbiterEngine(specs, ledger), ledger, specs, environment_path, views_path


def current_key_bytes(ledger: Ledger, view_id: str, key: str) -> bytes:
    current = ledger.current_view(view_id)
    assert current is not None
    return canonical_bytes(current["result"]["keys"][key])
