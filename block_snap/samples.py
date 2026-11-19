"""Deterministic sample volume profiles used by the demo and test suites.

Three profiles deliberately stress different scale regimes:

* ``small_many_snapshots``  - tiny volume, many snapshots (chain depth)
* ``large_low_change``      - big volume with a sparse change ratio (bitmap path)
* ``large_high_change``     - big volume with a dense change ratio (full path)

All content derives from :func:`deterministic_block` so a regenerated image is
byte-identical across machines and runs.
"""
from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass, field
from typing import Tuple

from .service import BlockSnapService
from .volume import deterministic_block


@dataclass(frozen=True)
class SampleProfile:
    name: str
    block_size: int
    block_count: int
    seed: str
    description: str
    # change specs applied between snapshots (hex block no : hex pattern)
    rounds: Tuple[str, ...] = field(default_factory=tuple)
    snapshot_names: Tuple[str, ...] = field(default_factory=tuple)


PROFILES: dict[str, SampleProfile] = {
    "small_many_snapshots": SampleProfile(
        name="small_many_snapshots",
        block_size=512,
        block_count=16,
        seed="small",
        description="16 x 512B blocks; 5 snapshots exercising deep delta chains",
        rounds=(
            "00:11,03:22",
            "01:33,0a:44,0f:55",
            "00:66,07:77",
            "0b:88,0c:99,0d:aa",
        ),
        snapshot_names=("base", "snap-a", "snap-b", "snap-c", "snap-d"),
    ),
    "large_low_change": SampleProfile(
        name="large_low_change",
        block_size=4096,
        block_count=256,
        seed="low",
        description="256 x 4KiB blocks; ~3% change ratio drives the bitmap path",
        rounds=(
            "05:ab,40:cd,80:ef,c0:12",
            "10:34,90:56",
            "ff:77,02:88",
        ),
        snapshot_names=("base", "low-a", "low-b", "low-c"),
    ),
    "large_high_change": SampleProfile(
        name="large_high_change",
        block_size=4096,
        block_count=128,
        seed="high",
        description="128 x 4KiB blocks; ~45% change ratio drives the full path",
        rounds=(
            ",".join(f"{block_no:02x}:{(block_no * 7) & 0xff:02x}" for block_no in range(0, 60)),
            ",".join(f"{block_no:02x}:{(block_no * 13 + 5) & 0xff:02x}" for block_no in range(30, 90)),
        ),
        snapshot_names=("base", "high-a", "high-b"),
    ),
}


def build_profile(service: BlockSnapService, profile: SampleProfile) -> None:
    """Create the baseline volume then walk the deterministic change rounds."""
    service.create_volume(profile.name, profile.block_size, profile.block_count, profile.seed)
    service.create_snapshot(profile.name, profile.snapshot_names[0])
    for spec, snap_name in zip(profile.rounds, profile.snapshot_names[1:]):
        service.inject_changes(profile.name, spec)
        service.create_snapshot(profile.name, snap_name)


def reset_workspace(workspace: str) -> None:
    """Remove old images, staging area, backups and SQLite manifests."""
    def make_writable(func, path, _exc_info):
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)

    for child in ("volumes", "blocks", "tmp", "backups", "meta"):
        path = os.path.join(workspace, child)
        if os.path.exists(path):
            shutil.rmtree(path, onerror=make_writable)
    os.makedirs(workspace, exist_ok=True)


def expected_block(profile: SampleProfile, block_no: int) -> bytes:
    """Recompute the current expected content of one block from its specs."""
    pattern = 0
    for spec in profile.rounds:
        for changed, changed_pattern in (
            (int(item.split(":")[0], 16), int(item.split(":")[1], 16))
            for item in spec.split(",")
        ):
            if changed == block_no:
                pattern = changed_pattern
    return deterministic_block(profile.seed, block_no, profile.block_size, pattern)
