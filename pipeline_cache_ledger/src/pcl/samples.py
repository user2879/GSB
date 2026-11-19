"""Deterministic sample workspace: artifact files + manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

from .digest import digest_file

MANIFEST_NAME = "pipeline_manifest.json"
ARTIFACT_DIR = "artifacts"

# content_id -> exact bytes; stable across platforms.
SAMPLE_CONTENT = {
    "base-lib": b"base-lib\ncontent-version=1\nline-a\nline-b\n",
    "codegen": b"codegen\ncontent-version=1\ndepends-on=base-lib\n",
    "config": b"config\ncontent-version=1\nregion=local\n",
    "core-bundle": (
        b"core-bundle\ncontent-version=1\ndepends-on=codegen\n"
        b"depends-on=config\n"
    ),
    "api-image": b"api-image\ncontent-version=1\ndepends-on=core-bundle\n",
    "web-image": b"web-image\ncontent-version=1\ndepends-on=core-bundle\n",
    "release-notes": b"release-notes\ncontent-version=1\ndepends-on=api-image\n",
}


def write_sample_workspace(root, *, manifest_version=3, pipeline_version=1,
                           content_overrides=None, version_overrides=None):
    root = Path(root)
    art_dir = root / ARTIFACT_DIR
    art_dir.mkdir(parents=True, exist_ok=True)
    content_overrides = content_overrides or {}
    version_overrides = version_overrides or {}

    paths = {}
    for key, data in SAMPLE_CONTENT.items():
        data = content_overrides.get(key, data)
        path = art_dir / f"{key}.txt"
        path.write_bytes(data)
        paths[key] = path

    def d(key):
        return digest_file(paths[key])

    stages = [
        {
            "id": "stage-foundation",
            "artifacts": [
                {"key": "base-lib", "version": version_overrides.get("base-lib", 1),
                 "file": f"{ARTIFACT_DIR}/base-lib.txt", "refs": []},
                {"key": "config", "version": version_overrides.get("config", 1),
                 "file": f"{ARTIFACT_DIR}/config.txt", "refs": []},
            ],
        },
        {
            "id": "stage-build",
            "artifacts": [
                {"key": "codegen", "version": version_overrides.get("codegen", 1),
                 "file": f"{ARTIFACT_DIR}/codegen.txt",
                 "refs": [{"key": "base-lib", "digest": d("base-lib"),
                           "version": version_overrides.get("base-lib", 1)}]},
                {"key": "core-bundle",
                 "version": version_overrides.get("core-bundle", 1),
                 "file": f"{ARTIFACT_DIR}/core-bundle.txt",
                 "refs": [
                     {"key": "codegen", "digest": d("codegen"),
                      "version": version_overrides.get("codegen", 1)},
                     {"key": "config", "digest": d("config"),
                      "version": version_overrides.get("config", 1)},
                 ]},
            ],
        },
        {
            "id": "stage-package",
            "artifacts": [
                {"key": "api-image", "version": version_overrides.get("api-image", 1),
                 "file": f"{ARTIFACT_DIR}/api-image.txt",
                 "refs": [{"key": "core-bundle", "digest": d("core-bundle"),
                           "version": version_overrides.get("core-bundle", 1)}]},
                {"key": "web-image", "version": version_overrides.get("web-image", 1),
                 "file": f"{ARTIFACT_DIR}/web-image.txt",
                 "refs": [{"key": "core-bundle", "digest": d("core-bundle"),
                           "version": version_overrides.get("core-bundle", 1)}]},
            ],
        },
        {
            "id": "stage-release",
            "artifacts": [
                {"key": "release-notes",
                 "version": version_overrides.get("release-notes", 1),
                 "file": f"{ARTIFACT_DIR}/release-notes.txt",
                 "refs": [{"key": "api-image", "digest": d("api-image"),
                           "version": version_overrides.get("api-image", 1)}]},
            ],
        },
    ]
    manifest = {
        "format_version": manifest_version,
        "pipeline_version": pipeline_version,
        "stages": stages,
    }
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    return manifest_path, paths
