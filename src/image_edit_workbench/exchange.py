from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .codec_v1 import parse_v1
from .codec_v2 import encode_v2, parse_v2
from .convert import downgrade_v2
from .models import ImageInfo, ParseOutcome, RegionEdit
from .util import sha256_bytes


@dataclass(frozen=True)
class MaterializedExchange:
    version: int
    text: str
    parsed: ParseOutcome
    exchange_fingerprint: str
    sidecar_hash: str
    sidecar: dict | None

    @property
    def regions(self) -> list[RegionEdit]:
        return self.parsed.regions


def materialize_logical_state(
    regions: list[RegionEdit],
    *,
    version: int,
    images: Mapping[str, ImageInfo],
    manifest_keys: set[str],
) -> MaterializedExchange:
    downgraded = downgrade_v2(regions)
    sidecar = downgraded.sidecar
    sidecar_hash = downgraded.sidecar_hash
    if version == 2:
        text = encode_v2(regions)
        parsed = parse_v2(text, images=images, manifest_keys=manifest_keys, source="generated-v2")
    elif version == 1:
        text = downgraded.text
        parsed = parse_v1(text, images=images, manifest_keys=manifest_keys, source="generated-v1")
    else:
        raise ValueError(f"unsupported exchange version {version}")
    return MaterializedExchange(
        version=version,
        text=text,
        parsed=parsed,
        exchange_fingerprint=sha256_bytes(text.encode("utf-8")),
        sidecar_hash=sidecar_hash,
        sidecar=sidecar,
    )


def parse_exchange_bytes(
    data: bytes,
    *,
    version: int,
    images: Mapping[str, ImageInfo],
    manifest_keys: set[str],
    source: str,
) -> MaterializedExchange:
    if version == 1:
        parsed = parse_v1(data, images=images, manifest_keys=manifest_keys, source=source)
    elif version == 2:
        parsed = parse_v2(data, images=images, manifest_keys=manifest_keys, source=source)
    else:
        raise ValueError(f"unsupported exchange version {version}")
    return MaterializedExchange(
        version=version,
        text=data.decode("utf-8", errors="replace"),
        parsed=parsed,
        exchange_fingerprint=sha256_bytes(data),
        sidecar_hash="",
        sidecar=None,
    )


def load_exchange_file(
    path: Path,
    *,
    version: int,
    images: Mapping[str, ImageInfo],
    manifest_keys: set[str],
) -> MaterializedExchange:
    return parse_exchange_bytes(
        path.read_bytes(),
        version=version,
        images=images,
        manifest_keys=manifest_keys,
        source=str(path),
    )

