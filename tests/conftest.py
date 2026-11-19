from __future__ import annotations

from pathlib import Path

import pytest

from hls_workbench.config import EncoderConfig, MediaConfig
from hls_workbench.media import GeneratedMedia, SyntheticMediaGenerator


@pytest.fixture(scope="session")
def media_config() -> MediaConfig:
    return MediaConfig(duration_seconds=2.0, slice_seconds=1.0, width=320, height=180)


@pytest.fixture(scope="session")
def generated_media(tmp_path_factory: pytest.TempPathFactory, media_config: MediaConfig) -> GeneratedMedia:
    root = tmp_path_factory.mktemp("media")
    generator = SyntheticMediaGenerator(media_config, EncoderConfig(), root)
    return generator.generate()
