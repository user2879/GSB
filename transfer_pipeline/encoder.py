"""Deterministic shared-encoder forward representation stage."""

from __future__ import annotations

from typing import Any

import numpy as np

from .canonical import sha256_json
from .inputs import Encoder, feature_library_fingerprint


def _activate(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "identity":
        return np.array(values, dtype=np.float64, copy=True)
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "tanh":
        return np.tanh(values)
    raise ValueError(f"unsupported activation: {activation}")


def forward_fingerprint_inputs(encoder: Encoder, features: dict[str, dict[str, Any]], activation: str) -> str:
    return sha256_json(
        {
            "encoder_fingerprint": encoder.fingerprint,
            "feature_library_fingerprint": feature_library_fingerprint(features),
            "activation": activation,
        }
    )


def run_shared_forward(
    encoder: Encoder,
    features: dict[str, dict[str, Any]],
    activation: str,
) -> dict[str, Any]:
    """Run all encoder layers in layer order and sample-ID order.

    The encoder parameter arrays are marked read-only during loading. Every
    operation that could alias an input creates a copy, so representations can
    never write through to the base weights.
    """
    sample_content = feature_library_fingerprint(features)
    samples: dict[str, Any] = {}
    for sample_id in sorted(features):
        record = features[sample_id]
        raw = np.asarray(record["features"], dtype=np.float64)
        current = np.array((raw - encoder.mean) / encoder.std, dtype=np.float64, copy=True)
        normalized_fingerprint = sha256_json(current.tolist())
        layer_representations: dict[str, Any] = {}
        for layer in encoder.layers:
            # numpy.matmul creates a fresh output; bias addition is also out-of-place.
            current = np.matmul(current, layer.weights) + layer.bias
            current = _activate(current, activation)
            layer_representations[layer.name] = {
                "version": layer.version,
                "values": current.tolist(),
                "fingerprint": sha256_json(
                    {
                        "layer": layer.name,
                        "layer_version": layer.version,
                        "sample_id": sample_id,
                        "values": current.tolist(),
                    }
                ),
            }
        samples[sample_id] = {
            "input_fingerprint": normalized_fingerprint,
            "layers": layer_representations,
        }

    input_fingerprint = forward_fingerprint_inputs(encoder, features, activation)
    representation_fingerprint = sha256_json(
        {
            "input_fingerprint": input_fingerprint,
            "layers": {
                layer.name: {
                    "version": layer.version,
                    "fingerprint": layer.fingerprint,
                    "output_dim": layer.output_dim,
                }
                for layer in encoder.layers
            },
            "samples": {
                sample_id: {
                    layer_name: representation["fingerprint"]
                    for layer_name, representation in sorted(sample["layers"].items())
                }
                for sample_id, sample in sorted(samples.items())
            },
        }
    )
    return {
        "schema_version": 1,
        "encoder_fingerprint": encoder.fingerprint,
        "encoder_version_manifest": encoder.layer_manifests(),
        "normalization_fingerprint": encoder.normalization_fingerprint,
        "sample_content_fingerprint": sample_content,
        "input_fingerprint": input_fingerprint,
        "representation_fingerprint": representation_fingerprint,
        "activation": activation,
        "sample_count": len(samples),
        "samples": samples,
    }


def target_representation(artifact: dict[str, Any], sample_id: str, target_layer: str) -> np.ndarray:
    try:
        values = artifact["samples"][sample_id]["layers"][target_layer]["values"]
    except KeyError as exc:
        raise ValueError(
            f"missing representation for sample {sample_id!r} at layer {target_layer!r}"
        ) from exc
    return np.asarray(values, dtype=np.float64)
