from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from fingerprint_workbench.search import compare_retrieval_candidates, search
from fingerprint_workbench.workbench import Workbench


def test_forced_termination_recovers_only_committed_state(generated: dict, tmp_path: Path) -> None:
    root = tmp_path / "workbench"
    workbench = Workbench(root)
    workbench.import_track("track_01", generated["tracks"]["track_01"])
    workbench.import_track("track_02", generated["tracks"]["track_02"])
    committed_version = workbench.current.version
    committed_hash = workbench.current.manifest_hash
    workbench.close()
    control_root = tmp_path / "control"
    shutil.copytree(root, control_root)

    project_root = str(Path(__file__).resolve().parent.parent)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = project_root + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fingerprint_workbench",
            "_crash-import",
            "--root",
            str(root),
            "--track-id",
            "track_03",
            "--audio",
            generated["tracks"]["track_03"],
            "--crash-after-shards",
            "1",
        ],
        cwd=project_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode != 0
    recovered = Workbench(root)
    assert recovered.current.version == committed_version
    assert recovered.current.manifest_hash == committed_hash
    assert recovered.recovery_report["status"] == "recovered"
    assert recovered.recovery_report["clean_previous_exit"] is False
    assert recovered.recovery_report["orphan_temp_files"]
    resumed = recovered.import_track("track_03", generated["tracks"]["track_03"])
    query = generated["queries"]["q_01_exact_cross"]
    resumed_report = search(recovered, query, query_id="q_01_exact_cross")
    recovered.close()

    control = Workbench(control_root)
    controlled = control.import_track("track_03", generated["tracks"]["track_03"])
    control_report = search(control, query, query_id="q_01_exact_cross")
    control.close()
    assert resumed.manifest_hash == controlled.manifest_hash
    assert resumed.state_digest == controlled.state_digest
    compare_retrieval_candidates(control_report, resumed_report)
