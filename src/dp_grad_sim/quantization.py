from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .errors import SimError


PACKET_MAGIC = b"DPQ1"
PACKET_VERSION = 1
SCHEME_TO_CODE = {"symmetric": 0, "asymmetric": 1}
CODE_TO_SCHEME = {value: key for key, value in SCHEME_TO_CODE.items()}
PACKET_BASE_STRUCT = struct.Struct(">4sBBBBidQQH")
DIMENSION_STRUCT = struct.Struct(">I")
CRC_STRUCT = struct.Struct(">I")
PACKET_BASE_SIZE = PACKET_BASE_STRUCT.size


@dataclass(frozen=True)
class QuantParams:
    tensor_name: str
    scheme: str
    bits: int
    scale: float
    zero_point: int
    shape: tuple[int, ...]

    @property
    def qmax(self) -> int:
        if self.scheme == "symmetric":
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1

    def as_dict(self) -> dict[str, object]:
        return {
            "tensor_name": self.tensor_name,
            "scheme": self.scheme,
            "bits": self.bits,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class PacketView:
    header: QuantParams
    payload: bytes

    @property
    def packet_size(self) -> int:
        return packet_size(self.header.tensor_name, self.header.shape, self.header.bits)


@dataclass(frozen=True)
class DecodedPacket:
    header: QuantParams
    q: np.ndarray
    values: np.ndarray
    payload: bytes


def payload_size_for(numel: int, bits: int) -> int:
    if numel < 0:
        raise ValueError("numel must be non-negative")
    return (numel * bits + 7) // 8


def packet_size(name: str, shape: Iterable[int], bits: int) -> int:
    shape_tuple = tuple(int(dimension) for dimension in shape)
    name_bytes = name.encode("utf-8")
    rank = len(shape_tuple)
    numel = math.prod(shape_tuple)
    return (
        PACKET_BASE_SIZE
        + DIMENSION_STRUCT.size * rank
        + len(name_bytes)
        + CRC_STRUCT.size
        + payload_size_for(numel, bits)
        + CRC_STRUCT.size
    )


def _validate_values(values: np.ndarray, tensor_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise SimError(
            "E_PARAM_NONFINITE",
            f"non-finite values in tensor {tensor_name}",
            tensor=tensor_name,
        )
    return array


def compute_common_quant_params(
    tensor_name: str,
    arrays: Iterable[np.ndarray],
    bits: int,
    scheme: str,
) -> QuantParams:
    if scheme not in SCHEME_TO_CODE:
        raise SimError("E_CONFIG_INVALID", f"unsupported quantization scheme {scheme!r}")
    if bits < 2 or bits > 16:
        raise SimError("E_CONFIG_INVALID", "quantization bits must be in [2, 16]")
    ordered_arrays = [_validate_values(array, tensor_name) for array in arrays]
    if not ordered_arrays:
        raise SimError("E_DATA_INVALID", f"cannot quantize tensor {tensor_name} with no source arrays")
    shape = ordered_arrays[0].shape
    if any(array.shape != shape for array in ordered_arrays):
        raise SimError("E_DATA_INVALID", f"shape mismatch while quantizing tensor {tensor_name}")
    numel = int(shape[0]) if len(shape) == 1 else int(math.prod(shape))

    if numel == 0:
        return QuantParams(tensor_name, scheme, bits, 1.0, 0, tuple(shape))

    if scheme == "symmetric":
        qmax = (1 << (bits - 1)) - 1
        max_abs = max(float(np.max(np.abs(array))) for array in ordered_arrays)
        scale = max_abs / qmax if max_abs > 0 else 1.0
        return QuantParams(tensor_name, scheme, bits, float(scale), 0, tuple(shape))

    qmax = (1 << bits) - 1
    low = min(float(np.min(array)) for array in ordered_arrays)
    high = max(float(np.max(array)) for array in ordered_arrays)
    span = high - low
    if span > 0:
        scale = span / qmax
    else:
        scale = max(abs(low), 1.0) / qmax
    if not math.isfinite(scale) or scale <= 0:
        raise SimError("E_PARAM_NONFINITE", f"invalid dynamic scale for tensor {tensor_name}")
    zero_point = int(round(-low / scale))
    zero_point = min(qmax, max(0, zero_point))
    return QuantParams(tensor_name, scheme, bits, float(scale), zero_point, tuple(shape))


def quantize(values: np.ndarray, params: QuantParams) -> np.ndarray:
    array = _validate_values(values, params.tensor_name)
    if tuple(array.shape) != params.shape:
        raise SimError(
            "E_QUANT_SHAPE_MISMATCH",
            f"tensor {params.tensor_name} shape {array.shape} does not match {params.shape}",
        )
    if not math.isfinite(params.scale) or params.scale <= 0:
        raise SimError("E_QUANT_HEADER_INVALID", "quantization scale must be finite and positive")
    qmax = params.qmax
    if params.scheme == "symmetric":
        rounded = np.rint(array.astype(np.float64) / params.scale).astype(np.int64)
        return np.clip(rounded, -qmax, qmax).astype(np.int64)
    rounded = np.rint(array.astype(np.float64) / params.scale).astype(np.int64) + params.zero_point
    return np.clip(rounded, 0, qmax).astype(np.int64)


def dequantize(q: np.ndarray, params: QuantParams) -> np.ndarray:
    quantized = np.asarray(q, dtype=np.int64)
    if tuple(quantized.shape) != params.shape:
        raise SimError(
            "E_QUANT_SHAPE_MISMATCH",
            f"quantized tensor {params.tensor_name} shape {quantized.shape} does not match {params.shape}",
        )
    if params.scheme == "symmetric":
        values = quantized.astype(np.float64) * params.scale
    else:
        values = (quantized.astype(np.float64) - params.zero_point) * params.scale
    return values.astype(np.float32)


def _pack_unsigned(unsigned: np.ndarray, bits: int) -> bytes:
    if unsigned.size == 0:
        return b""
    flat = np.asarray(unsigned, dtype=np.uint64).reshape(-1)
    bit_positions = np.arange(bits, dtype=np.uint64)
    planes = ((flat[:, None] >> bit_positions[None, :]) & np.uint64(1)).astype(np.uint8)
    return np.packbits(planes.reshape(-1), bitorder="little").tobytes()


def _unpack_unsigned(payload: bytes, numel: int, bits: int) -> np.ndarray:
    if numel == 0:
        return np.zeros((0,), dtype=np.uint64)
    expected = payload_size_for(numel, bits)
    if len(payload) != expected:
        raise SimError(
            "E_QUANT_HEADER_INVALID",
            f"payload length {len(payload)} does not match expected {expected}",
        )
    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    needed = numel * bits
    if unpacked.size < needed:
        raise SimError("E_QUANT_HEADER_INVALID", "truncated packed payload")
    matrix = unpacked[:needed].reshape(numel, bits).astype(np.uint64)
    weights = np.left_shift(np.uint64(1), np.arange(bits, dtype=np.uint64))
    return np.sum(matrix * weights[None, :], axis=1, dtype=np.uint64)


def _encode_unsigned(q: np.ndarray, bits: int) -> np.ndarray:
    mask = np.uint64((1 << bits) - 1)
    return np.asarray(q, dtype=np.int64).astype(np.uint64) & mask


def _decode_unsigned(unsigned: np.ndarray, bits: int, scheme: str) -> np.ndarray:
    signed = unsigned.astype(np.int64)
    if scheme == "symmetric":
        sign_bit = np.int64(1 << (bits - 1))
        signed = np.where(signed >= sign_bit, signed - np.int64(1 << bits), signed)
    return signed.astype(np.int64)


def build_packet(values: np.ndarray, params: QuantParams) -> bytes:
    array = _validate_values(values, params.tensor_name)
    if tuple(array.shape) != params.shape:
        raise SimError(
            "E_QUANT_SHAPE_MISMATCH",
            f"tensor {params.tensor_name} shape {array.shape} does not match {params.shape}",
        )
    q = quantize(array, params)
    payload = _pack_unsigned(_encode_unsigned(q, params.bits), params.bits)
    name_bytes = params.tensor_name.encode("utf-8")
    if len(name_bytes) > 65535:
        raise SimError("E_QUANT_HEADER_INVALID", "tensor name is too long for packet header")
    if len(params.shape) > 32:
        raise SimError("E_QUANT_HEADER_INVALID", "tensor rank is too large for packet header")
    header_without_crc = bytearray()
    header_without_crc.extend(
        PACKET_BASE_STRUCT.pack(
            PACKET_MAGIC,
            PACKET_VERSION,
            SCHEME_TO_CODE[params.scheme],
            params.bits,
            len(params.shape),
            params.zero_point,
            params.scale,
            int(math.prod(params.shape)),
            len(payload),
            len(name_bytes),
        )
    )
    for dimension in params.shape:
        if dimension < 0 or dimension > 0xFFFFFFFF:
            raise SimError("E_QUANT_HEADER_INVALID", "tensor dimension cannot be encoded")
        header_without_crc.extend(DIMENSION_STRUCT.pack(dimension))
    header_without_crc.extend(name_bytes)
    header_crc = zlib.crc32(header_without_crc) & 0xFFFFFFFF
    packet = bytes(header_without_crc) + CRC_STRUCT.pack(header_crc) + payload
    packet += CRC_STRUCT.pack(zlib.crc32(payload) & 0xFFFFFFFF)
    return packet


def parse_packet(
    packet: bytes,
    *,
    expected_name: str | None = None,
    expected_shape: tuple[int, ...] | None = None,
    expected_bits: int | None = None,
    expected_scheme: str | None = None,
) -> PacketView:
    minimum = PACKET_BASE_SIZE + CRC_STRUCT.size + CRC_STRUCT.size
    if len(packet) < minimum:
        raise SimError("E_QUANT_HEADER_INVALID", "packet is shorter than the minimum header")
    (
        magic,
        version,
        scheme_code,
        bits,
        rank,
        zero_point,
        scale,
        numel,
        payload_bytes,
        name_len,
    ) = PACKET_BASE_STRUCT.unpack_from(packet, 0)
    if magic != PACKET_MAGIC:
        raise SimError("E_QUANT_HEADER_INVALID", "invalid packet magic")
    if version != PACKET_VERSION:
        raise SimError("E_QUANT_HEADER_INVALID", f"unsupported packet version {version}")
    if scheme_code not in CODE_TO_SCHEME:
        raise SimError("E_QUANT_HEADER_INVALID", f"unknown scheme code {scheme_code}")
    if bits < 2 or bits > 16:
        raise SimError("E_QUANT_HEADER_INVALID", f"invalid bit width {bits}")
    if rank > 32:
        raise SimError("E_QUANT_HEADER_INVALID", f"invalid tensor rank {rank}")
    if not math.isfinite(scale) or scale <= 0:
        raise SimError("E_QUANT_HEADER_INVALID", "invalid quantization scale")
    scheme = CODE_TO_SCHEME[scheme_code]
    qmax = (1 << (bits - 1)) - 1 if scheme == "symmetric" else (1 << bits) - 1
    if zero_point < 0 or zero_point > qmax:
        raise SimError("E_QUANT_HEADER_INVALID", "zero point is outside the representable range")
    if scheme == "symmetric" and zero_point != 0:
        raise SimError("E_QUANT_HEADER_INVALID", "symmetric packets must use zero point 0")

    offset = PACKET_BASE_SIZE
    dims: list[int] = []
    for _ in range(rank):
        if offset + DIMENSION_STRUCT.size > len(packet):
            raise SimError("E_QUANT_HEADER_INVALID", "truncated tensor shape")
        dims.append(DIMENSION_STRUCT.unpack_from(packet, offset)[0])
        offset += DIMENSION_STRUCT.size
    if offset + name_len > len(packet):
        raise SimError("E_QUANT_HEADER_INVALID", "truncated tensor name")
    name_bytes = packet[offset : offset + name_len]
    offset += name_len
    try:
        tensor_name = name_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SimError("E_QUANT_HEADER_INVALID", "tensor name is not valid UTF-8") from exc
    if not tensor_name:
        raise SimError("E_QUANT_HEADER_INVALID", "tensor name is empty")
    if offset + CRC_STRUCT.size > len(packet):
        raise SimError("E_QUANT_HEADER_INVALID", "truncated header checksum")
    expected_header_crc = CRC_STRUCT.unpack_from(packet, offset)[0]
    actual_header_crc = zlib.crc32(packet[:offset]) & 0xFFFFFFFF
    offset += CRC_STRUCT.size
    if expected_header_crc != actual_header_crc:
        raise SimError("E_QUANT_HEADER_INVALID", "header checksum mismatch")
    shape = tuple(int(dimension) for dimension in dims)
    expected_numel = math.prod(shape)
    if numel != expected_numel:
        raise SimError(
            "E_QUANT_HEADER_INVALID", f"numel {numel} does not match shape product {expected_numel}"
        )
    expected_payload_bytes = payload_size_for(numel, bits)
    if payload_bytes != expected_payload_bytes:
        raise SimError(
            "E_QUANT_HEADER_INVALID",
            f"payload byte count {payload_bytes} does not match bit-packed size {expected_payload_bytes}",
        )
    if offset + payload_bytes + CRC_STRUCT.size != len(packet):
        raise SimError("E_QUANT_HEADER_INVALID", "packet length does not match header")
    payload = packet[offset : offset + payload_bytes]
    payload_crc = CRC_STRUCT.unpack_from(packet, offset + payload_bytes)[0]
    if payload_crc != (zlib.crc32(payload) & 0xFFFFFFFF):
        raise SimError("E_QUANT_HEADER_INVALID", "payload checksum mismatch")

    if expected_name is not None and tensor_name != expected_name:
        raise SimError(
            "E_QUANT_HEADER_INVALID",
            f"unexpected tensor name {tensor_name!r}; expected {expected_name!r}",
        )
    if expected_shape is not None and shape != tuple(expected_shape):
        raise SimError(
            "E_QUANT_HEADER_INVALID",
            f"unexpected shape {shape}; expected {tuple(expected_shape)}",
        )
    if expected_bits is not None and bits != expected_bits:
        raise SimError(
            "E_QUANT_HEADER_INVALID", f"unexpected bit width {bits}; expected {expected_bits}"
        )
    if expected_scheme is not None and scheme != expected_scheme:
        raise SimError(
            "E_QUANT_HEADER_INVALID", f"unexpected scheme {scheme!r}; expected {expected_scheme!r}"
        )

    header = QuantParams(tensor_name, scheme, bits, float(scale), int(zero_point), shape)
    return PacketView(header=header, payload=payload)


def decode_packet(
    packet: bytes,
    *,
    expected_name: str | None = None,
    expected_shape: tuple[int, ...] | None = None,
    expected_bits: int | None = None,
    expected_scheme: str | None = None,
) -> DecodedPacket:
    view = parse_packet(
        packet,
        expected_name=expected_name,
        expected_shape=expected_shape,
        expected_bits=expected_bits,
        expected_scheme=expected_scheme,
    )
    header = view.header
    unsigned = _unpack_unsigned(view.payload, int(math.prod(header.shape)), header.bits)
    q = _decode_unsigned(unsigned, header.bits, header.scheme).reshape(header.shape)
    values = dequantize(q, header)
    return DecodedPacket(header=header, q=q, values=values, payload=view.payload)


def compress_tensor(
    values: np.ndarray,
    tensor_name: str,
    bits: int,
    scheme: str,
    params: QuantParams | None = None,
) -> bytes:
    array = _validate_values(values, tensor_name)
    actual_params = params or compute_common_quant_params(tensor_name, [array], bits, scheme)
    if actual_params.tensor_name != tensor_name:
        raise SimError("E_QUANT_HEADER_INVALID", "quantization parameters use a different tensor name")
    if actual_params.bits != bits or actual_params.scheme != scheme:
        raise SimError("E_QUANT_HEADER_INVALID", "quantization parameters do not match requested format")
    return build_packet(array, actual_params)


def quant_params_from_dict(value: Mapping[str, object]) -> QuantParams:
    shape = value.get("shape")
    if not isinstance(shape, list):
        raise SimError("E_QUANT_HEADER_INVALID", "quantization shape must be a list")
    return QuantParams(
        tensor_name=str(value["tensor_name"]),
        scheme=str(value["scheme"]),
        bits=int(value["bits"]),
        scale=float(value["scale"]),
        zero_point=int(value["zero_point"]),
        shape=tuple(int(item) for item in shape),
    )
