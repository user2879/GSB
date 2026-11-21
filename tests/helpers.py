from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

from replica_repair.manifest_repo import compute_state_digest
from replica_repair.model import Manifest
from replica_repair.sample import init_sample, sample_manifest_path, start_servers
from replica_repair.util import canonical_json_dumps


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_sample(root: Path) -> None:
    init_sample(root, force=True)


def load_v3(root: Path, group_id: str = "interop-v3") -> Manifest:
    path = sample_manifest_path(root, group_id)
    from replica_repair.manifest_store import ManifestStore

    return ManifestStore(root).load_current(path).manifest


@contextlib.contextmanager
def group_servers(root: Path, group_id: str = "interop-v3"):
    manifest = load_v3(root, group_id)
    path = sample_manifest_path(root, group_id)
    with start_servers(root, path, [node.node_id for node in manifest.nodes]) as (clients, servers):
        yield manifest, clients, servers


def run_cli(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(
        [sys.executable, "-m", "replica_repair", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def stable_json(value) -> str:
    return canonical_json_dumps(value)


def current_state(root: Path, group_id: str = "interop-v3") -> tuple[Manifest, str]:
    manifest = Manifest.from_file(sample_manifest_path(root, group_id))
    return manifest, compute_state_digest(root, manifest)
