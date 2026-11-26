"""Deterministic fixed-point quantization, integer MAC and dequantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import LayerSpec
from .errors import E_BAD_DIMENSION, E_DIMENSION, E_NONFINITE_INPUT, E_QUANT_LEVEL, QLabError
from .util import digest_array, digest_int_array, digest_json, rounded_list


@dataclass
class QuantizedTensor:
    values: np.ndarray
    scale_units: int
    frac_bits: int
    quant_level: int

    @property
    def unit(self) -> float:
        return 1.0 / float(1 << self.frac_bits)

    @property
    def scale(self) -> float:
        return float(self.scale_units) * self.unit

    def dequantize(self) -> np.ndarray:
        return self.values.astype(np.float64) * self.scale

    def digest(self) -> str:
        return digest_int_array(self.values)

    def to_dict(self, *, include_values: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "shape": list(self.values.shape),
            "scale_units": int(self.scale_units),
            "scale": self.scale,
            "frac_bits": self.frac_bits,
            "quant_level": self.quant_level,
            "digest": self.digest(),
        }
        if include_values:
            payload["values"] = self.values.astype(int).tolist()
        return payload


def quantize_fixed(array: Any, *, frac_bits: int, quant_level: int) -> QuantizedTensor:
    """Symmetric fixed-point quantization with a power-of-two fractional unit.

    The group/tensor scale is itself represented as an integer number of
    ``2^-frac_bits`` units. Integer matmul is therefore exact and the
    dequantized result only depends on the integer MAC and the two scale units.
    """

    if quant_level not in (4, 8, 16):
        raise QLabError(E_QUANT_LEVEL, f"unsupported quant level {quant_level}", quant_level=quant_level)
    if frac_bits not in (4, 6, 8, 10, 12):
        raise QLabError("E_BAD_FRAC_BITS", f"unsupported fractional width {frac_bits}", frac_bits=frac_bits)
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise QLabError(E_NONFINITE_INPUT, "tensor contains non-finite values")
    qmax = (1 << (int(quant_level) - 1)) - 1
    qmin = -(1 << (int(quant_level) - 1))
    unit_denominator = 1 << int(frac_bits)
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    if maximum == 0.0:
        scale_units = 1
        quantized = np.zeros(values.shape, dtype=np.int64)
    else:
        raw_units = np.ceil(maximum * unit_denominator / qmax)
        scale_units = max(1, int(raw_units))
        scale = scale_units / unit_denominator
        quantized = np.rint(values / scale).astype(np.int64)
        np.clip(quantized, qmin, qmax, out=quantized)
    return QuantizedTensor(
        values=quantized,
        scale_units=scale_units,
        frac_bits=int(frac_bits),
        quant_level=int(quant_level),
    )


def integer_matmul(left: QuantizedTensor, right: QuantizedTensor) -> np.ndarray:
    if left.values.shape[-1] != right.values.shape[1]:
        raise QLabError(
            E_BAD_DIMENSION,
            f"matmul inner dimensions differ: {left.values.shape} and {right.values.shape}",
            code_detail=E_DIMENSION,
        )
    # int64 is sufficient for the small deterministic acceptance tensors and
    # makes overflow detection explicit rather than silently wrapping.
    left_i = left.values.astype(np.int64, copy=False)
    right_i = right.values.astype(np.int64, copy=False)
    with np.errstate(over="raise"):
        try:
            return left_i @ right_i.T
        except FloatingPointError as exc:  # pragma: no cover - NumPy integer overflow policy varies
            raise QLabError(E_BAD_DIMENSION, "integer matrix multiply overflow") from exc


def weight_group_fingerprints(layer: LayerSpec) -> list[dict[str, Any]]:
    flat = np.ascontiguousarray(layer.weight, dtype=np.float64).reshape(-1)
    groups: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, flat.size, layer.group_size)):
        group = flat[start : start + layer.group_size]
        if group.size != layer.group_size:
            raise QLabError(
                "E_GROUP_NOT_DIVISIBLE",
                f"layer {layer.name} group {index} is incomplete",
                layer=layer.name,
            )
        groups.append(
            {
                "index": index,
                "start": start,
                "end": start + layer.group_size,
                "digest": digest_array(group),
            }
        )
    return groups


def layer_content_fingerprint(layer: LayerSpec) -> str:
    return digest_json(
        {
            "name": layer.name,
            "shape": [layer.rows, layer.cols],
            "group_size": layer.group_size,
            "weight": digest_array(layer.weight),
            "bias": "__NONE__" if layer.bias is None else digest_array(layer.bias),
        }
    )


def layer_config_fingerprint(layer: LayerSpec) -> str:
    return digest_json(
        {
            "name": layer.name,
            "rows": layer.rows,
            "cols": layer.cols,
            "group_size": layer.group_size,
            "bias": "__NONE__" if layer.bias is None else digest_array(layer.bias),
        }
    )


def layer_cache_keys(
    layer: LayerSpec,
    *,
    activation: np.ndarray,
    revision: int,
    frac_bits: int,
    quant_level: int,
    batch_ids: list[str],
) -> dict[str, str]:
    base = {
        "layer_content": layer_content_fingerprint(layer),
        "layer_config": layer_config_fingerprint(layer),
        "group_fingerprints": [group["digest"] for group in weight_group_fingerprints(layer)],
        "activation": digest_array(activation),
        "revision": revision,
        "frac_bits": frac_bits,
        "quant_level": quant_level,
        "batch_ids": sorted(batch_ids),
    }
    return {
        "quantize": digest_json({"role": "quantize", **base}),
        "matmul": digest_json({"role": "matmul", **base}),
        "dequant": digest_json({"role": "dequant", **base}),
    }


def clean_float_output(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64).copy()
    result[np.abs(result) < 1e-15] = 0.0
    return result


def evaluate_layer(
    layer: LayerSpec,
    activation: np.ndarray,
    *,
    revision: int,
    frac_bits: int,
    quant_level: int,
    batch_ids: list[str],
    layer_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    activation = np.asarray(activation, dtype=np.float64)
    if activation.ndim != 2 or activation.shape[1] != layer.cols:
        raise QLabError(
            E_BAD_DIMENSION,
            f"activation shape {list(activation.shape)} does not match layer {layer.name} input width {layer.cols}",
            layer=layer.name,
        )
    keys = layer_cache_keys(
        layer,
        activation=activation,
        revision=revision,
        frac_bits=frac_bits,
        quant_level=quant_level,
        batch_ids=batch_ids,
    )
    quantized_input = quantize_fixed(activation, frac_bits=frac_bits, quant_level=quant_level)
    quantized_weight = quantize_fixed(layer.weight, frac_bits=frac_bits, quant_level=quant_level)
    accumulated = integer_matmul(quantized_input, quantized_weight)
    dequantized = accumulated.astype(np.float64) * quantized_input.scale * quantized_weight.scale
    if layer.bias is not None:
        dequantized = dequantized + layer.bias.reshape(1, -1)
    output = clean_float_output(dequantized)
    groups = weight_group_fingerprints(layer)
    record = {
        "layer_index": layer_index,
        "layer": layer.name,
        "revision": revision,
        "frac_bits": frac_bits,
        "quant_level": quant_level,
        "input_shape": list(activation.shape),
        "output_shape": list(output.shape),
        "layer_content_fingerprint": layer_content_fingerprint(layer),
        "layer_config_fingerprint": layer_config_fingerprint(layer),
        "group_fingerprints": groups,
        "activation_fingerprint": digest_array(activation),
        "cache_keys": keys,
        "quantized_input": quantized_input.to_dict(),
        "quantized_weight": quantized_weight.to_dict(include_values=False),
        "integer_accumulator_digest": digest_int_array(accumulated),
        "integer_accumulator": accumulated.astype(int).tolist(),
        "dequantized_digest": digest_array(output),
        "dequantized_output": output.astype(float).tolist(),
        "referenced_batch_ids": sorted(set(batch_ids)),
    }
    return output, record


def evaluate_chain(
    layers: list[LayerSpec],
    samples: np.ndarray,
    *,
    revision: int,
    frac_bits: int,
    quant_level: int,
    batch_id: str,
    reused_layer_records: dict[int, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a chain while reusing only cache entries whose keys match."""

    activation = np.asarray(samples, dtype=np.float64)
    records: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        expected_keys = layer_cache_keys(
            layer,
            activation=activation,
            revision=revision,
            frac_bits=frac_bits,
            quant_level=quant_level,
            batch_ids=[batch_id],
        )
        cached = (reused_layer_records or {}).get(index)
        if cached is not None and cached.get("cache_keys", {}).get("dequant") == expected_keys["dequant"]:
            output = np.asarray(cached["dequantized_output"], dtype=np.float64)
            record = dict(cached)
            record["referenced_batch_ids"] = sorted(set(record.get("referenced_batch_ids", [])) | {batch_id})
            records.append(record)
            reused.append(
                {
                    "layer_index": index,
                    "layer": layer.name,
                    "batch_id": batch_id,
                    "cache_key": expected_keys["dequant"],
                    "reason": "matching_layer_group_config_fingerprint",
                }
            )
        else:
            output, record = evaluate_layer(
                layer,
                activation,
                revision=revision,
                frac_bits=frac_bits,
                quant_level=quant_level,
                batch_ids=[batch_id],
                layer_index=index,
            )
            records.append(record)
        activation = output
    ledger_key = digest_json(
        {
            "batch_id": batch_id,
            "revision": revision,
            "frac_bits": frac_bits,
            "quant_level": quant_level,
            "layer_cache_keys": [record["cache_keys"]["dequant"] for record in records],
            "output": digest_array(activation),
        }
    )
    result_meta = {
        "batch_id": batch_id,
        "ledger_cache_key": ledger_key,
        "output_digest": digest_array(activation),
        "output": activation.astype(float).tolist(),
        "layer_cache_keys": [record["cache_keys"]["dequant"] for record in records],
    }
    return activation, records, reused, result_meta


def quant_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "quant_levels": sorted({int(record["quant_level"]) for record in records}),
        "frac_bits": sorted({int(record["frac_bits"]) for record in records}),
        "layer_digests": [record["dequantized_digest"] for record in records],
        "digest": digest_json(
            {
                "levels": [int(record["quant_level"]) for record in records],
                "frac_bits": [int(record["frac_bits"]) for record in records],
                "dequantized": [record["dequantized_digest"] for record in records],
            }
        ),
    }


def output_summary(output: np.ndarray) -> str:
    return digest_json({"output": rounded_list(output, 15)})

