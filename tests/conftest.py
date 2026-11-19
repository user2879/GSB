from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audiofp.config import DEFAULT_CONFIG
from audiofp.fixtures import generate_fixtures, load_manifest
from audiofp.library import Library


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fixtures")
    generate_fixtures(root, seed=20261119, clean=True)
    return root


@pytest.fixture(scope="session")
def manifest(fixture_root: Path) -> dict:
    return load_manifest(fixture_root)


@pytest.fixture(scope="session")
def main_index(
    fixture_root: Path,
    manifest: dict,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    index = tmp_path_factory.mktemp("main-index") / "current.db"
    with Library(index, DEFAULT_CONFIG) as library:
        for track in manifest["tracks"]:
            for entry in track["formats"]:
                library.import_file(
                    fixture_root / entry["relative_path"],
                    root=fixture_root,
                    display_label=track["title"],
                )
    return index
