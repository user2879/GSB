"""Self-describing binary trajectory format and conversion helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .errors import ErrorCode, PipelineError
from .hashing import animation_semantic_sha256, sha256_bytes
from .json_format import LATEST_SCHEMA_VERSION, animation_from_document, animation_to_document
from .math3d import format_quaternion, normalize_quaternion
from .models import Animation

MAGIC = b"BTRJ"
HEADER_SIZE = 24
FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, 2)
ENDIAN_MARKER_VALUE = 0x01020304
QUAT_ORDER_TO_CODE = {"wxyz": 0, "xyzw": 1}
CODE_TO_QUAT_ORDER = {value: key for key, value in QUAT_ORDER_TO_CODE.items()}
INTERPOLATION_TO_CODE = {"slerp": 0, "linear": 1, "step": 2}
CODE_TO_INTERPOLATION = {value: key for key, value in INTERPOLATION_TO_CODE.items()}
ENDIAN_TO_CODE = {"little": ord("L"), "big": ord("B")}
CODE_TO_ENDIAN = {value: key for key, value in ENDIAN_TO_CODE.items()}


@dataclass
class BinaryMetadata:
    format_version: int
    endian: str
    quat_order: str
    index_width: int
    joint_count: int
    frame_count: int
    interpolation: str
    has_root_translation: bool
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "endian": self.endian,
            "quat_order": self.quat_order,
            "index_width": self.index_width,
            "joint_count": self.joint_count,
            "frame_count": self.frame_count,
            "interpolation": self.interpolation,
            "has_root_translation": self.has_root_translation,
            "sha256": self.sha256,
        }


@dataclass
class ParsedBinary:
    animation: Animation
    metadata: BinaryMetadata


@dataclass
class ConversionReport:
    direction: str
    source_semantic_sha256: str
    destination_semantic_sha256: str
    source_artifact_sha256: str | None
    destination_artifact_sha256: str
    joint_count: int
    frame_count: int
    equal_within_tolerance: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "source_semantic_sha256": self.source_semantic_sha256,
            "destination_semantic_sha256": self.destination_semantic_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "destination_artifact_sha256": self.destination_artifact_sha256,
            "joint_count": self.joint_count,
            "frame_count": self.frame_count,
            "equal_within_tolerance": self.equal_within_tolerance,
            "metadata": self.metadata,
        }


def _endian_prefix(endian: str) -> str:
    if endian == "little":
        return "<"
    if endian == "big":
        return ">"
    raise PipelineError(
        ErrorCode.E_FORMAT_ENDIAN,
        f"unsupported endian declaration: {endian}",
        field="endian",
        details={"allowed": ["little", "big"]},
    )


def _validate_channel_order(joint_count: int, channel_joint_order: Sequence[int] | None) -> tuple[int, ...]:
    if channel_joint_order is None:
        return tuple(range(joint_count))
    order = tuple(int(value) for value in channel_joint_order)
    if len(order) != joint_count:
        raise PipelineError(
            ErrorCode.E_INDEX_OVERFLOW,
            "channel order length does not match joint count",
            field="channel_joint_order",
            details={"joint_count": joint_count, "channel_count": len(order)},
        )
    if set(order) != set(range(joint_count)):
        raise PipelineError(
            ErrorCode.E_INDEX_OVERFLOW,
            "channel order must be a permutation of joint indices",
            field="channel_joint_order",
        )
    return order


def animation_to_binary(
    animation: Animation,
    *,
    endian: str = "little",
    index_width: int = 4,
    quat_order: str = "wxyz",
    format_version: int = FORMAT_VERSION,
    channel_joint_order: Sequence[int] | None = None,
) -> tuple[bytes, BinaryMetadata]:
    endian = endian.lower()
    quat_order = quat_order.lower()
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise PipelineError(
            ErrorCode.E_SCHEMA_VERSION,
            f"unsupported binary format version {format_version}",
            field="format_version",
            details={"supported_versions": list(SUPPORTED_FORMAT_VERSIONS)},
        )
    if index_width not in (2, 4):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "index_width must be 2 or 4",
            field="index_width",
        )
    if animation.joint_count > (1 << (8 * index_width)):
        raise PipelineError(
            ErrorCode.E_INDEX_OVERFLOW,
            "joint count cannot be represented by the chosen index width",
            field="index_width",
            details={"joint_count": animation.joint_count, "index_width": index_width},
        )
    if quat_order not in QUAT_ORDER_TO_CODE:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            f"unsupported quat_order: {quat_order}",
            field="quat_order",
        )
    prefix = _endian_prefix(endian)
    channel_order = _validate_channel_order(animation.joint_count, channel_joint_order)
    root_present = format_version >= 2
    declared_endian = ENDIAN_TO_CODE[endian]
    marker = struct.pack(prefix + "I", ENDIAN_MARKER_VALUE)
    flags = 1 if root_present else 0

    header = bytearray()
    header.extend(MAGIC)
    header.extend(struct.pack("B", format_version))
    header.extend(struct.pack("B", declared_endian))
    header.extend(marker)
    header.extend(struct.pack("B", index_width))
    header.extend(struct.pack("B", QUAT_ORDER_TO_CODE[quat_order]))
    header.extend(struct.pack("B", INTERPOLATION_TO_CODE[animation.interpolation]))
    header.extend(struct.pack("B", flags))
    header.extend(b"\x00\x00")
    header.extend(struct.pack(prefix + "I", animation.joint_count))
    header.extend(struct.pack(prefix + "I", animation.frame_count))
    if len(header) != HEADER_SIZE:
        raise AssertionError(f"binary header size bug: {len(header)}")

    payload = bytearray()
    index_format = prefix + ("H" if index_width == 2 else "I")
    for joint_index in channel_order:
        payload.extend(struct.pack(index_format, joint_index))

    for name in animation.joint_names:
        encoded = name.encode("utf-8")
        if len(encoded) > 0xFFFF:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "joint name is too long for binary name table",
                joint=name,
                field="names",
            )
        payload.extend(struct.pack(prefix + "H", len(encoded)))
        payload.extend(encoded)

    for value in animation.times:
        payload.extend(struct.pack(prefix + "d", float(value)))
    if root_present:
        for translation in animation.root_translation:
            for value in translation:
                payload.extend(struct.pack(prefix + "d", float(value)))
    for frame_index in range(animation.frame_count):
        for joint_index in channel_order:
            payload.extend(
                struct.pack(
                    prefix + "4d",
                    *format_quaternion(animation.rotations[frame_index, joint_index], quat_order),
                )
            )

    data = bytes(header) + bytes(payload)
    metadata = BinaryMetadata(
        format_version=format_version,
        endian=endian,
        quat_order=quat_order,
        index_width=index_width,
        joint_count=animation.joint_count,
        frame_count=animation.frame_count,
        interpolation=animation.interpolation,
        has_root_translation=root_present,
        sha256=sha256_bytes(data),
    )
    return data, metadata


def _read_u16(data: bytes, offset: int, prefix: str, field: str) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary payload is truncated",
            field=field,
        )
    return struct.unpack_from(prefix + "H", data, offset)[0], offset + 2


def _detect_endian(header: bytes) -> tuple[str, str]:
    marker = header[6:10]
    big_marker = struct.pack(">I", ENDIAN_MARKER_VALUE)
    little_marker = struct.pack("<I", ENDIAN_MARKER_VALUE)
    if marker == big_marker:
        detected = "big"
    elif marker == little_marker:
        detected = "little"
    else:
        raise PipelineError(
            ErrorCode.E_FORMAT_ENDIAN,
            "binary endian marker is invalid",
            field="endian_marker",
            details={"marker_hex": marker.hex()},
        )
    declared_code = header[5]
    declared = CODE_TO_ENDIAN.get(declared_code)
    if declared is None:
        raise PipelineError(
            ErrorCode.E_FORMAT_ENDIAN,
            "binary endian declaration byte is invalid",
            field="declared_endian",
            details={"declared_byte": declared_code, "detected_endian": detected},
        )
    if declared != detected:
        raise PipelineError(
            ErrorCode.E_FORMAT_ENDIAN,
            "binary endian declaration conflicts with marker",
            field="declared_endian",
            details={"declared_endian": declared, "detected_endian": detected},
        )
    return declared, ">" if declared == "big" else "<"


def parse_binary(data: bytes) -> ParsedBinary:
    if len(data) < HEADER_SIZE:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary trajectory header is truncated",
            field="header",
            details={"expected_size": HEADER_SIZE, "actual_size": len(data)},
        )
    if data[:4] != MAGIC:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary trajectory magic is invalid",
            field="magic",
            details={"actual_hex": data[:4].hex()},
        )
    endian, prefix = _detect_endian(data)
    format_version = int(data[4])
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise PipelineError(
            ErrorCode.E_SCHEMA_VERSION,
            f"unsupported binary format version {format_version}",
            field="format_version",
            details={"supported_versions": list(SUPPORTED_FORMAT_VERSIONS)},
        )
    index_width = int(data[10])
    if index_width not in (2, 4):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary index width must be 2 or 4",
            field="index_width",
        )
    quat_order = CODE_TO_QUAT_ORDER.get(int(data[11]))
    if quat_order is None:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary quaternion order code is invalid",
            field="quat_order",
        )
    interpolation = CODE_TO_INTERPOLATION.get(int(data[12]))
    if interpolation is None:
        if format_version == 1:
            interpolation = "slerp"
        else:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "binary interpolation code is invalid",
                field="interpolation",
            )
    flags = int(data[13])
    root_present = format_version >= 2 and bool(flags & 1)
    joint_count = struct.unpack_from(prefix + "I", data, 16)[0]
    frame_count = struct.unpack_from(prefix + "I", data, 20)[0]
    if joint_count <= 0:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary joint count must be positive",
            field="joint_count",
        )
    if frame_count <= 0:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary frame count must be positive",
            field="frame_count",
        )

    offset = HEADER_SIZE
    table_size = joint_count * index_width
    if offset + table_size > len(data):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary index table is truncated",
            field="index_table",
        )
    index_format = prefix + ("H" if index_width == 2 else "I")
    channel_joint_order = tuple(
        struct.unpack_from(index_format, data, offset + row * index_width)[0]
        for row in range(joint_count)
    )
    offset += table_size
    for row, joint_index in enumerate(channel_joint_order):
        if joint_index >= joint_count:
            raise PipelineError(
                ErrorCode.E_INDEX_OVERFLOW,
                "binary channel index exceeds joint count",
                field="index_table",
                details={
                    "channel_row": row,
                    "joint_index": joint_index,
                    "joint_count": joint_count,
                },
            )
    if len(set(channel_joint_order)) != joint_count:
        raise PipelineError(
            ErrorCode.E_INDEX_OVERFLOW,
            "binary index table contains duplicate joint indices",
            field="index_table",
            details={"channel_joint_order": list(channel_joint_order)},
        )

    joint_names: list[str] = []
    for joint_index in range(joint_count):
        name_length, offset = _read_u16(data, offset, prefix, "names")
        if offset + name_length > len(data):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "binary joint name is truncated",
                joint=f"index_{joint_index}",
                field="names",
            )
        try:
            name = data[offset : offset + name_length].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "binary joint name is not valid UTF-8",
                joint=f"index_{joint_index}",
                field="names",
            ) from exc
        if not name:
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "binary joint name must be non-empty",
                joint=f"index_{joint_index}",
                field="names",
            )
        offset += name_length
        joint_names.append(name)
    if len(set(joint_names)) != joint_count:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary joint names are not unique",
            field="names",
        )

    times_size = frame_count * 8
    if offset + times_size > len(data):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "binary timestamp block is truncated",
            field="times",
        )
    times = np.frombuffer(data, dtype=prefix + "f8", count=frame_count, offset=offset).astype(
        np.float64, copy=True
    )
    offset += times_size
    if not np.all(np.isfinite(times)):
        bad = int(np.nonzero(~np.isfinite(times))[0][0])
        raise PipelineError(
            ErrorCode.E_TIME_NONMONOTONIC,
            "binary timestamp is NaN or infinite",
            frame=bad,
            field="times",
        )
    violations = np.nonzero(np.diff(times) <= 0.0)[0]
    if violations.size:
        frame = int(violations[0] + 1)
        raise PipelineError(
            ErrorCode.E_TIME_NONMONOTONIC,
            "binary timestamps must be strictly increasing",
            frame=frame,
            field="times",
            details={"previous": float(times[frame - 1]), "current": float(times[frame])},
        )

    if root_present:
        roots_size = frame_count * 3 * 8
        if offset + roots_size > len(data):
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "binary root translation block is truncated",
                field="root_translation",
            )
        roots = np.frombuffer(
            data, dtype=prefix + "f8", count=frame_count * 3, offset=offset
        ).astype(np.float64, copy=True)
        roots = roots.reshape(frame_count, 3)
        offset += roots_size
        if not np.all(np.isfinite(roots)):
            bad = int(np.nonzero(~np.isfinite(roots).all(axis=1))[0][0])
            raise PipelineError(
                ErrorCode.E_FIELD_MISSING,
                "binary root translation contains non-finite values",
                frame=bad,
                field="root_translation",
            )
    else:
        roots = np.zeros((frame_count, 3), dtype=np.float64)

    rotation_values = frame_count * joint_count * 4
    rotation_bytes = rotation_values * 8
    remaining = len(data) - offset
    if remaining < rotation_bytes:
        available_values = max(remaining // 8, 0)
        available_quaternions = available_values // 4
        missing_frame = min(available_quaternions // joint_count, frame_count - 1)
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            "binary quaternion component count does not match frame and joint counts",
            frame=missing_frame,
            field="rotations",
            details={
                "expected_float64_components": rotation_values,
                "actual_float64_components": available_values,
                "joint_count": joint_count,
                "frame_count": frame_count,
            },
        )
    if remaining > rotation_bytes:
        raise PipelineError(
            ErrorCode.E_QUAT_INVALID,
            "binary quaternion component count does not match frame and joint counts",
            frame=frame_count - 1,
            field="rotations",
            details={
                "expected_float64_components": rotation_values,
                "actual_float64_components": remaining // 8,
                "joint_count": joint_count,
                "frame_count": frame_count,
                "extra_bytes": remaining - rotation_bytes,
            },
        )
    raw_rotations = np.frombuffer(
        data, dtype=prefix + "f8", count=rotation_values, offset=offset
    ).astype(np.float64, copy=True).reshape(frame_count, joint_count, 4)
    if quat_order == "xyzw":
        raw_rotations = raw_rotations[..., [3, 0, 1, 2]]
    canonical_rotations = np.empty_like(raw_rotations)
    for channel_row, joint_index in enumerate(channel_joint_order):
        canonical_rotations[:, joint_index, :] = raw_rotations[:, channel_row, :]
    for frame_index in range(frame_count):
        for joint_index, joint_name in enumerate(joint_names):
            normalize_quaternion(
                canonical_rotations[frame_index, joint_index],
                joint=joint_name,
                frame=frame_index,
            )

    animation = Animation(
        joint_names=tuple(joint_names),
        times=times,
        rotations=canonical_rotations,
        root_translation=roots,
        interpolation=interpolation,
        fps=None,
        source_schema_version=None,
    )
    metadata = BinaryMetadata(
        format_version=format_version,
        endian=endian,
        quat_order=quat_order,
        index_width=index_width,
        joint_count=joint_count,
        frame_count=frame_count,
        interpolation=interpolation,
        has_root_translation=root_present,
        sha256=sha256_bytes(data),
    )
    return ParsedBinary(animation=animation, metadata=metadata)


def parse_binary_file(path: str | Path) -> ParsedBinary:
    return parse_binary(Path(path).read_bytes())


def write_binary_file(
    path: str | Path,
    animation: Animation,
    *,
    endian: str = "little",
    index_width: int = 4,
    quat_order: str = "wxyz",
    format_version: int = FORMAT_VERSION,
    channel_joint_order: Sequence[int] | None = None,
) -> tuple[str, BinaryMetadata]:
    data, metadata = animation_to_binary(
        animation,
        endian=endian,
        index_width=index_width,
        quat_order=quat_order,
        format_version=format_version,
        channel_joint_order=channel_joint_order,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return metadata.sha256, metadata


def json_document_to_binary(
    document: dict[str, Any],
    *,
    endian: str = "little",
    index_width: int = 4,
    quat_order: str = "wxyz",
    format_version: int = FORMAT_VERSION,
) -> tuple[bytes, ParsedBinary, ConversionReport]:
    animation = animation_from_document(document)
    data, _writer_metadata = animation_to_binary(
        animation,
        endian=endian,
        index_width=index_width,
        quat_order=quat_order,
        format_version=format_version,
    )
    parsed = parse_binary(data)
    before_hash = animation_semantic_sha256(animation)
    after_hash = animation_semantic_sha256(parsed.animation)
    report = ConversionReport(
        direction="json_to_binary",
        source_semantic_sha256=before_hash,
        destination_semantic_sha256=after_hash,
        source_artifact_sha256=None,
        destination_artifact_sha256=parsed.metadata.sha256,
        joint_count=parsed.animation.joint_count,
        frame_count=parsed.animation.frame_count,
        equal_within_tolerance=before_hash == after_hash,
        metadata=parsed.metadata.to_dict(),
    )
    return data, parsed, report


def binary_to_json_document(
    data: bytes,
    *,
    schema_version: int = LATEST_SCHEMA_VERSION,
    quat_order: str = "wxyz",
) -> tuple[dict[str, Any], ParsedBinary, ConversionReport]:
    parsed = parse_binary(data)
    output_order = quat_order if schema_version != 1 else "xyzw"
    document = animation_to_document(
        parsed.animation, schema_version=schema_version, quat_order=output_order
    )
    reparsed = animation_from_document(document)
    before_hash = animation_semantic_sha256(parsed.animation)
    after_hash = animation_semantic_sha256(reparsed)
    report = ConversionReport(
        direction="binary_to_json",
        source_semantic_sha256=before_hash,
        destination_semantic_sha256=after_hash,
        source_artifact_sha256=parsed.metadata.sha256,
        destination_artifact_sha256=sha256_bytes(json_bytes_for_hash(document)),
        joint_count=reparsed.joint_count,
        frame_count=reparsed.frame_count,
        equal_within_tolerance=before_hash == after_hash,
        metadata=parsed.metadata.to_dict(),
    )
    return document, parsed, report


def json_bytes_for_hash(document: dict[str, Any]) -> bytes:
    from .json_format import document_json_bytes

    return document_json_bytes(document, pretty=False)
