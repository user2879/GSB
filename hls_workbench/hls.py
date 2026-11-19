"""Deterministic HLS playlist emission."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .canonical import sha256_file
from .config import MediaConfig, ProfileConfig
from .transcode import TranscodeArtifact


def write_hls_playlists(
    *,
    root: Path,
    profiles: tuple[ProfileConfig, ...],
    media: MediaConfig,
    artifacts: list[TranscodeArtifact],
) -> dict[str, str]:
    ordered_profiles = sorted(profiles, key=lambda profile: profile.priority)
    target_duration = max(1, math.ceil(media.slice_seconds))
    by_key = {(artifact.profile, artifact.sequence): artifact for artifact in artifacts}
    hashes: dict[str, str] = {}

    master_lines = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-INDEPENDENT-SEGMENTS"]
    for profile in ordered_profiles:
        bandwidth = (profile.video_bitrate_kbps + profile.audio_bitrate_kbps) * 1000
        master_lines.append(
            f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},AVERAGE-BANDWIDTH={bandwidth},"
            f"RESOLUTION={profile.width}x{profile.height}"
        )
        master_lines.append(f"{profile.name}/index.m3u8")
    master_path = root / "hls" / "master.m3u8"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_text("\n".join(master_lines) + "\n", encoding="utf-8", newline="\n")
    hashes[master_path.relative_to(root).as_posix()] = sha256_file(master_path)

    for profile in ordered_profiles:
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{target_duration}",
            "#EXT-X-MEDIA-SEQUENCE:0",
            "#EXT-X-PLAYLIST-TYPE:VOD",
            "#EXT-X-INDEPENDENT-SEGMENTS",
        ]
        for sequence in range(media.slice_count):
            artifact = by_key[(profile.name, sequence)]
            duration_seconds = media.frame_count(sequence) / media.fps
            lines.append(f"#EXTINF:{duration_seconds:.3f},")
            lines.append(Path(artifact.relative_path).name)
        lines.append("#EXT-X-ENDLIST")
        playlist_path = root / "hls" / profile.name / "index.m3u8"
        playlist_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        hashes[playlist_path.relative_to(root).as_posix()] = sha256_file(playlist_path)
    return hashes
