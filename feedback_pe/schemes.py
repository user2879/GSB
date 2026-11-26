from __future__ import annotations

import math
from typing import Any

import numpy as np

from .quant import quantize_vector


class PositionOperator:
    """Common interface for every positional-angle growth rule."""

    kind = "base"

    def __init__(self, scheme: dict[str, Any]):
        self.name = scheme["name"]
        self.params = dict(scheme)
        self.base_freq = float(scheme["base_freq"])
        self.scale = float(scheme["scale"])
        self.block_size = int(scheme["block_size"])

    def angle_for_dimension(self, position: int, dimension_index: int,
                            dimension: int) -> np.float64:
        raise NotImplementedError

    def build_float(self, position: int, dimension: int) -> dict[str, list[float]]:
        angles: list[float] = []
        vector: list[float] = []
        for index in range(dimension):
            angle = np.float64(self.angle_for_dimension(int(position), index, dimension))
            angles.append(float(angle))
            if index % 2 == 0:
                value = np.sin(angle, dtype=np.float64)
            else:
                value = np.cos(angle, dtype=np.float64)
            vector.append(float(value))
        return {"angles": angles, "pe": vector}

    def float_encoding(self, sample: dict[str, Any]) -> dict[str, Any]:
        dimension = int(sample["dimension"])
        query_position = int(sample["query_position"])
        query_position_vector = self.build_float(query_position, dimension)
        query_encoded = [
            float(np.float64(sample["query_vector"][index])
                  + np.float64(query_position_vector["pe"][index]))
            for index in range(dimension)
        ]
        keys = []
        for key_index, position in enumerate(sample["key_positions"]):
            position_vector = self.build_float(int(position), dimension)
            encoded = [
                float(np.float64(sample["key_vectors"][key_index][index])
                      + np.float64(position_vector["pe"][index]))
                for index in range(dimension)
            ]
            keys.append({
                "key_index": key_index,
                "position": int(position),
                **position_vector,
                "encoded_vector": encoded,
            })
        return {
            "scheme": self.name,
            "kind": self.kind,
            "sample_id": sample["id"],
            "query": {
                "position": query_position,
                **query_position_vector,
                "encoded_vector": query_encoded,
            },
            "keys": keys,
        }

    def fixed_encoding(self, sample: dict[str, Any], frac_bits: int,
                       quant_tier: str) -> dict[str, Any]:
        dimension = int(sample["dimension"])
        float_encoding = self.float_encoding(sample)

        query_feature_quant = quantize_vector(
            [float(x) for x in sample["query_vector"]], frac_bits, quant_tier,
        )
        query_pe_quant = quantize_vector(
            float_encoding["query"]["pe"], frac_bits, quant_tier,
        )
        query_feature_int = [item["int_value"] for item in query_feature_quant]
        query_pe_int = [item["int_value"] for item in query_pe_quant]
        query_encoded_int = [
            query_feature_int[index] + query_pe_int[index]
            for index in range(dimension)
        ]

        keys = []
        for key_index, position in enumerate(sample["key_positions"]):
            feature_quant = quantize_vector(
                [float(x) for x in sample["key_vectors"][key_index]],
                frac_bits, quant_tier,
            )
            pe_quant = quantize_vector(
                float_encoding["keys"][key_index]["pe"], frac_bits, quant_tier,
            )
            feature_int = [item["int_value"] for item in feature_quant]
            pe_int = [item["int_value"] for item in pe_quant]
            encoded_int = [
                feature_int[index] + pe_int[index]
                for index in range(dimension)
            ]
            keys.append({
                "key_index": key_index,
                "position": int(position),
                "angles": float_encoding["keys"][key_index]["angles"],
                "pe_float": float_encoding["keys"][key_index]["pe"],
                "pe_quantization": pe_quant,
                "pe_int": pe_int,
                "feature_int": feature_int,
                "feature_residual": [item["residual"] for item in feature_quant],
                "encoded_int": encoded_int,
            })

        return {
            "scheme": self.name,
            "kind": self.kind,
            "sample_id": sample["id"],
            "frac_bits": frac_bits,
            "quant_tier": quant_tier,
            "query": {
                "position": int(sample["query_position"]),
                "angles": float_encoding["query"]["angles"],
                "pe_float": float_encoding["query"]["pe"],
                "pe_quantization": query_pe_quant,
                "pe_int": query_pe_int,
                "feature_int": query_feature_int,
                "feature_residual": [item["residual"] for item in query_feature_quant],
                "encoded_int": query_encoded_int,
            },
            "keys": keys,
        }


class SinusoidalOperator(PositionOperator):
    kind = "sinusoidal"

    def angle_for_dimension(self, position: int, dimension_index: int,
                            dimension: int) -> np.float64:
        block = dimension_index // self.block_size
        exponent = np.float64(2.0 * block / dimension)
        inverse_frequency = np.power(np.float64(self.base_freq), -exponent,
                                     dtype=np.float64)
        return np.float64(position) * np.float64(self.scale) * inverse_frequency


class LinearRampOperator(PositionOperator):
    kind = "linear_ramp"

    def angle_for_dimension(self, position: int, dimension_index: int,
                            dimension: int) -> np.float64:
        denominator = np.sqrt(np.float64(self.base_freq), dtype=np.float64)
        growth = np.float64(dimension_index + 1) / np.float64(dimension)
        return (np.float64(position) * np.float64(self.scale)
                * growth * 10.0 / denominator)


class BlockwaveOperator(PositionOperator):
    kind = "blockwave"

    def angle_for_dimension(self, position: int, dimension_index: int,
                            dimension: int) -> np.float64:
        block = dimension_index // self.block_size
        block_count = max(1, dimension // self.block_size)
        root = np.power(np.float64(self.base_freq),
                        np.float64(1.0 / block_count), dtype=np.float64)
        frequency = np.power(root, block, dtype=np.float64) / np.sqrt(
            np.float64(self.base_freq), dtype=np.float64)
        return np.float64(position) * np.float64(self.scale) * frequency * 4.0


OPERATOR_CLASSES = {
    item.kind: item
    for item in (SinusoidalOperator, LinearRampOperator, BlockwaveOperator)
}


def make_operator(scheme: dict[str, Any]) -> PositionOperator:
    return OPERATOR_CLASSES[scheme["kind"]](scheme)
