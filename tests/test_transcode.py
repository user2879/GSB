from __future__ import annotations

from pathlib import Path

from hls_workbench.canonical import sha256_file
from hls_workbench.config import DEFAULT_PROFILES, EncoderConfig
from hls_workbench.transcode import inspect_ts_segment, transcode_segment


def test_profiles_have_continuous_absolute_pts(generated_media, media_config) -> None:
    record = generated_media.manifest["slices"][1]
    for profile in DEFAULT_PROFILES:
        output = generated_media.root / "hls" / profile.name / "segment_00001.ts"
        artifact = transcode_segment(
            input_video=generated_media.root / record["video_file"],
            input_audio=generated_media.root / record["audio_file"],
            output_path=output,
            profile=profile,
            sequence=1,
            media=media_config,
            encoder=EncoderConfig(),
        )
        assert artifact.decoded_frame_count == 30
        assert artifact.encoded_frame_count == 30
        check = inspect_ts_segment(output, profile=profile, sequence=1, media=media_config)
        assert check["video_start_pts_seconds"] == 1.0
        assert check["audio_start_pts_seconds"] == 1.0


def test_same_segment_transcode_is_byte_identical(generated_media, media_config, tmp_path: Path) -> None:
    record = generated_media.manifest["slices"][0]
    profile = DEFAULT_PROFILES[0]
    first = tmp_path / "first.ts"
    second = tmp_path / "second.ts"
    for output in (first, second):
        transcode_segment(
            input_video=generated_media.root / record["video_file"],
            input_audio=generated_media.root / record["audio_file"],
            output_path=output,
            profile=profile,
            sequence=0,
            media=media_config,
            encoder=EncoderConfig(),
        )
    assert sha256_file(first) == sha256_file(second)
