"""Compile a manifest version into content-addressed, byte-stable artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import hashlib

from ..util.canonical import canonical_bytes, digest
from . import fixedpoint as fp
from .grid import Grid
from .manifest import Item, ManifestVersion, topo_order


@dataclass(frozen=True)
class CompiledItem:
    key: str
    version: int
    artifact_id: str
    fixed: fp.Fixed
    value_exact: str
    targets: tuple[str, ...]
    derivation_fingerprint: str | None
    error: fp.ErrorResult
    artifact_bytes: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "version": self.version,
            "artifact_id": self.artifact_id,
            "fixed": self.fixed.to_dict(),
            "value_exact": self.value_exact,
            "targets": list(self.targets),
            "derivation_fingerprint": self.derivation_fingerprint,
            "error": self.error.to_dict(),
        }


@dataclass
class CompiledVersion:
    version: int
    strategy: str
    order: list[str]
    items: dict[str, CompiledItem]
    values: dict[str, fp.Fixed]
    errors: dict[str, fp.ErrorResult]
    content_digest: str


def _derivation_fingerprint(item: Item) -> str | None:
    if item.derivation is None:
        return None
    payload = item.derivation.to_dict()
    return digest(payload)


def compile_version(version: ManifestVersion, grid: Grid) -> CompiledVersion:
    ordered = topo_order(version)
    values: dict[str, fp.Fixed] = {}
    errors: dict[str, fp.ErrorResult] = {}
    leaf_errors: dict[str, Fraction] = {}
    items: dict[str, CompiledItem] = {}

    for key in ordered:
        item = version.items[key]
        if item.derivation is None:
            assert item.value is not None
            quantized = fp.quantize_value(item.value, grid)
            values[key] = quantized.fixed
            leaf_errors[key] = quantized.trunc_error
            if quantized.trunc_error != 0:
                leaf_error = fp.ErrorResult(
                    key=key,
                    total=quantized.trunc_error,
                    immediate=[
                        fp.Contribution(f"{key}#quant", quantized.trunc_error, "local-trunc")
                    ],
                    paths=[
                        fp.PathContribution(
                            path=[f"{key}#quant", key],
                            leaf=f"{key}#quant",
                            amount=quantized.trunc_error,
                            level=0,
                            share_pct=Fraction(0),
                        )
                    ],
                )
            else:
                leaf_error = fp.ErrorResult(key=key, total=Fraction(0))
            errors[key] = leaf_error
        else:
            derivation = item.derivation
            ordered_upstream = sorted(derivation.coefficients)
            coefficients = {
                upstream: fp.quantize_coefficient(derivation.coefficients[upstream], grid)
                for upstream in ordered_upstream
            }
            derived_fixed, local = fp.derive_with_values(
                key=key,
                ordered_upstream=ordered_upstream,
                values=values,
                coefficients=coefficients,
                offset=derivation.offset,
                grid=grid,
            )
            values[key] = derived_fixed
            errors[key] = fp.synthesize_error(
                key=key,
                ordered_upstream=ordered_upstream,
                coefficients=coefficients,
                upstream_results=errors,
                leaf_errors={},
                local=local,
            )

    for key in ordered:
        item = version.items[key]
        fingerprint = _derivation_fingerprint(item)
        # The artifact is content-addressed: it must NOT embed the manifest
        # version, otherwise an unchanged item would get different bytes across
        # versions.  The version is tracked separately on CompiledItem.
        payload = {
            "key": key,
            "grid": grid.to_dict(),
            "fixed": values[key].to_dict(),
            "value_exact": f"{item.value}" if item.derivation is None else None,
            "targets": sorted(item.targets),
            "derivation_fingerprint": fingerprint,
            "error": errors[key].to_dict(),
        }
        blob = canonical_bytes(payload)
        artifact_id = hashlib.sha256(blob).hexdigest()
        items[key] = CompiledItem(
            key=key,
            version=version.version,
            artifact_id=artifact_id,
            fixed=values[key],
            value_exact=payload["value_exact"] or f"{values[key].real}",
            targets=tuple(sorted(item.targets)),
            derivation_fingerprint=fingerprint,
            error=errors[key],
            artifact_bytes=blob,
        )

    version_digest = digest(
        {
            "version": version.version,
            "strategy": version.strategy,
            "order": ordered,
            "artifacts": {key: items[key].artifact_id for key in ordered},
        }
    )
    return CompiledVersion(
        version=version.version,
        strategy=version.strategy,
        order=ordered,
        items=items,
        values=values,
        errors=errors,
        content_digest=version_digest,
    )
