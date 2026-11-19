from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from audiofp.config import DEFAULT_CONFIG
from audiofp.library import Library


def _track_paths(root: Path, manifest: dict, truth_id: str) -> list[Path]:
    return [
        root / entry["relative_path"]
        for track in manifest["tracks"]
        if track["truth_id"] == truth_id
        for entry in track["formats"]
    ]


def test_incremental_import_and_delete_preserve_other_scores(
    fixture_root: Path, manifest: dict, tmp_path: Path
) -> None:
    index = tmp_path / "incremental.db"
    query = fixture_root / manifest["query"]["formats"][1]["relative_path"]
    with Library(index, DEFAULT_CONFIG) as library:
        for path in _track_paths(fixture_root, manifest, "aurora"):
            library.import_file(path, root=fixture_root, display_label="Aurora Circuit")
        before = library.query_file(query, top_k=3, search_variants=False)
        before_aurora = next(item for item in before["results"] if item["display_label"] == "Aurora Circuit")
        for path in _track_paths(fixture_root, manifest, "cinder"):
            library.import_file(path, root=fixture_root, display_label="Cinder Atlas")
        after_import = library.query_file(query, top_k=3, search_variants=False)
        after_import_aurora = next(
            item for item in after_import["results"] if item["display_label"] == "Aurora Circuit"
        )
        assert after_import_aurora["votes"] == before_aurora["votes"]
        assert after_import_aurora["alignment_start_seconds"] == before_aurora["alignment_start_seconds"]
        cinder_group = next(
            group["group_id"]
            for group in library.store.list_groups()
            if group["display_label"] == "Cinder Atlas"
        )
        deletion = library.store.delete_group(cinder_group)
        after_delete = library.query_file(query, top_k=3, search_variants=False)
        after_delete_aurora = next(
            item for item in after_delete["results"] if item["display_label"] == "Aurora Circuit"
        )
        assert deletion["deleted_sources"] == 3
        assert after_delete_aurora["votes"] == before_aurora["votes"]
        assert library.store.connection.execute(
            "SELECT COUNT(*) FROM fingerprints WHERE group_id=?", (cinder_group,)
        ).fetchone()[0] == 0


def test_committed_state_survives_forced_process_termination_and_orphans_recover(
    main_index: Path, tmp_path: Path
) -> None:
    index = tmp_path / "crash.db"
    shutil.copy2(main_index, index)
    with Library(index, DEFAULT_CONFIG) as library:
        before = library.store.stats()
    orphan = Path(str(index) + ".tmp-orphan-test")
    orphan.write_bytes(b"orphan")
    child_script = r"""
import os, sqlite3, sys
conn=sqlite3.connect(sys.argv[1]); conn.execute("PRAGMA foreign_keys=ON")
conn.execute("BEGIN IMMEDIATE")
conn.execute("INSERT INTO work_groups(group_id, display_label, fingerprint_count, duration) VALUES ('grp_crash','crash',1,1.0)")
conn.execute("INSERT INTO sources(source_id, group_id, relative_path, content_sha256, fingerprint_content_hash, container, codec, sample_rate, bit_depth, channels, declared_frames, decoded_frames, duration, format_signature, fingerprint_count, import_sequence, format_json) VALUES ('src_crash','grp_crash','x.wav','x','x','WAV','PCM_16',44100,16,1,1,1,1.0,'wav:pcm_16:44100:16:1',1,999,'{}')")
conn.execute("INSERT INTO fingerprints(fingerprint_hash, source_id, group_id, anchor_frame, target_frame, band1, band2, delta_frames) VALUES ('crashed','src_crash','grp_crash',0,1,1,1,1)")
os._exit(37)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_script, str(index)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert child.returncode == 37
    with Library(index, DEFAULT_CONFIG) as recovered:
        after = recovered.store.stats()
        crashed = recovered.store.connection.execute(
            "SELECT COUNT(*) FROM fingerprints WHERE fingerprint_hash='crashed'"
        ).fetchone()[0]
        assert after["groups"] == before["groups"]
        assert after["sources"] == before["sources"]
        assert after["fingerprints"] == before["fingerprints"]
        assert after["content_hash"] == before["content_hash"]
        assert crashed == 0
        assert orphan.name in after["recovered_orphans"]
        assert not orphan.exists()
