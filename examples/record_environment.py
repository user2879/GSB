"""Record the local runtime baseline used by replay checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_environment(samples_dir: Path) -> dict[str, Any]:
    manifest_path = samples_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_lines: list[str] = []
    for name in sorted(manifest["files"]):
        item = manifest["files"][name]
        sample_lines.append(f"{item['sha256']}  {item['path']}")
    sample_set_digest = hashlib.sha256(
        ("\n".join(sample_lines) + "\n").encode("utf-8")
    ).hexdigest()
    try:
        linux_distribution = platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        linux_distribution = None
    return {
        "target_platform": "local Windows, Python >= 3.11",
        "validation_host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "linux_distribution": linux_distribution,
            "wsl": "microsoft-standard-wsl2" in platform.release().lower(),
        },
        "python": {
            "version": platform.python_version(),
            "full_version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "numpy_version": numpy.__version__,
        "sample_manifest_path": str(manifest_path),
        "sample_manifest_sha256": _sha256(manifest_path),
        "sample_set_digest": sample_set_digest,
        "samples": manifest["files"],
        "runtime_dependency_policy": "local filesystem only; no network or external services",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=Path("samples"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/environment.json"))
    args = parser.parse_args()
    payload = build_environment(args.samples_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
