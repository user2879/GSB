"""Local binary trajectory format (``.btrj``).

Layout (all multi-byte integers/floats follow the declared endianness)::

    offset  size  field
    0       4     magic b'BTRJ'
    4       2     version (uint16, currently 1)
    6       1     endian marker: 1 = little endian, 2 = big endian
    7       1     index width: 1, 2, 4 or 8 bytes
    8       1     quaternion order: 1 = wxyz, 2 = xyzw
    9       1     flags: bit0 = root translation channel present
    10      2     joint count (uint16)
    12      4     frame count (uint32)
    16      ...   per frame:
                    t            float64
                    root xyz     3 * float32   (when flag bit0 set)
                    per joint:
                      joint index   uint of declared width
                      quaternion    4 * float32 in declared order

The index channel makes joint ordering explicit on disk; an index that does
not fit the declared width raises ``E_INDEX_OVERFLOW``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .animation import Animation
from .errors import E_INDEX_OVERFLOW
from .quat import QUAT_ORDERS, align_sequence, check_quat, reorder

MAGIC = b"BTRJ"
VERSION = 1
ENDIAN_MARKERS = {1: "<", 2: ">"}
ENDIAN_REVERSE = {"<": 1, ">": 2}
VALID_INDEX_WIDTHS = (1, 2, 4, 8)
QUAT_ORDER_MARKERS = {1: "wxyz", 2: "xyzw"}
QUAT_ORDER_REVERSE = {"wxyz": 1, "xyzw": 2}
FLAG_ROOT = 1
HEADER_SIZE = 16


@dataclass(frozen=True)
class BinarySpec:
    endianness: str = "<"  # '<' little, '>' big
    index_width: int = 4
    quat_order: str = "wxyz"
    include_root: bool = True

    def validate(self) -> None:
        if self.endianness not in ("<", ">"):
            raise ValueError("endianness must be '<' or '>'")
        if self.index_width not in VALID_INDEX_WIDTHS:
            raise ValueError(f"index width must be one of {VALID_INDEX_WIDTHS}")
        if self.quat_order not in QUAT_ORDERS:
            raise ValueError(f"quat_order must be one of {QUAT_ORDERS}")


def _index_max(width: int) -> int:
    return (1 << (8 * width)) - 1


def _index_fmt(spec: BinarySpec) -> str:
    return {1: "B", 2: "H", 4: "I", 8: "Q"}[spec.index_width]


def encode_binary(anim: Animation, spec: BinarySpec | None = None) -> bytes:
    spec = spec or BinarySpec()
    spec.validate()
    if anim.joint_count - 1 > _index_max(spec.index_width):
        raise E_INDEX_OVERFLOW(
            "joint index does not fit declared index width",
            {
                "field": "joint_index",
                "max_index": anim.joint_count - 1,
                "index_width": spec.index_width,
                "capacity": _index_max(spec.index_width),
            },
        )
    endian = spec.endianness
    header = (
        MAGIC
        + struct.pack(endian + "H", VERSION)
        + bytes(
            [
                ENDIAN_REVERSE[endian],
                spec.index_width,
                QUAT_ORDER_REVERSE[spec.quat_order],
                FLAG_ROOT if spec.include_root else 0,
            ]
        )
        + struct.pack(endian + "HI", anim.joint_count, anim.frame_count)
    )
    if len(header) != HEADER_SIZE:
        raise AssertionError("binary header size drift")

    ordered = reorder(align_sequence(anim.rotations), "wxyz", spec.quat_order)
    index_pack = struct.Struct(endian + _index_fmt(spec))
    parts = [header]
    for frame in range(anim.frame_count):
        parts.append(struct.pack(endian + "d", float(anim.times[frame])))
        if spec.include_root:
            parts.append(
                struct.pack(endian + "3f", *(float(v) for v in anim.root_translations[frame]))
            )
        for joint in range(anim.joint_count):
            parts.append(index_pack.pack(joint))
            parts.append(
                struct.pack(
                    endian + "4f", *(float(v) for v in ordered[frame, joint])
                )
            )
    return b"".join(parts)


def dump_binary(anim: Animation, path: str, spec: BinarySpec | None = None) -> None:
    with open(path, "wb") as handle:
        handle.write(encode_binary(anim, spec))


def decode_header(blob: bytes) -> tuple[BinarySpec, int, int]:
    if len(blob) < HEADER_SIZE or blob[:4] != MAGIC:
        raise ValueError("not a BTRJ stream: bad magic")
    # The version field precedes the endianness marker, so try both
    # byte orders and select the one whose marker is valid.
    endian_marker = blob[6]
    if endian_marker not in ENDIAN_MARKERS:
        raise ValueError(f"unknown endian marker {endian_marker}")
    endian = ENDIAN_MARKERS[endian_marker]
    version = struct.unpack(endian + "H", blob[4:6])[0]
    if version != VERSION:
        raise ValueError(
            "unsupported BTRJ version: "
            f"found {version}, supported {VERSION}"
        )
    index_width = blob[7]
    if index_width not in VALID_INDEX_WIDTHS:
        raise E_INDEX_OVERFLOW(
            "unsupported index width",
            {"field": "index_width", "found": index_width},
        )
    quat_order = QUAT_ORDER_MARKERS[blob[8]]
    flags = blob[9]
    joint_count, frame_count = struct.unpack(endian + "HI", blob[10:16])
    spec = BinarySpec(
        endianness=endian,
        index_width=index_width,
        quat_order=quat_order,
        include_root=bool(flags & FLAG_ROOT),
    )
    return spec, joint_count, frame_count


def decode_binary(
    blob: bytes,
    joint_names: tuple[str, ...],
    *,
    interp: str = "slerp",
    parameter_version: str = "baseline",
) -> Animation:
    spec, joint_count, frame_count = decode_header(blob)
    if len(joint_names) != joint_count:
        raise E_INDEX_OVERFLOW(
            "joint count disagrees with provided joint names",
            {"field": "joint_count", "found": joint_count},
        )
    endian = spec.endianness
    index_struct = struct.Struct(endian + _index_fmt(spec))
    offset = HEADER_SIZE
    times = np.zeros(frame_count, dtype=np.float64)
    roots = np.zeros((frame_count, 3), dtype=np.float64)
    rotations = np.zeros((frame_count, joint_count, 4), dtype=np.float64)

    for frame in range(frame_count):
        times[frame] = struct.unpack_from(endian + "d", blob, offset)[0]
        offset += 8
        if spec.include_root:
            roots[frame] = struct.unpack_from(endian + "3f", blob, offset)
            offset += 12
        seen: set[int] = set()
        for _ in range(joint_count):
            joint = index_struct.unpack_from(blob, offset)[0]
            offset += spec.index_width
            if joint >= joint_count:
                raise E_INDEX_OVERFLOW(
                    "joint index exceeds declared joint count",
                    {
                        "field": "joint_index",
                        "frame": frame,
                        "index": joint,
                        "joint_count": joint_count,
                    },
                )
            if joint in seen:
                raise E_INDEX_OVERFLOW(
                    "duplicate joint index inside frame",
                    {"field": "joint_index", "frame": frame, "index": joint},
                )
            seen.add(joint)
            values = struct.unpack_from(endian + "4f", blob, offset)
            offset += 16
            rotations[frame, joint] = reorder(
                np.asarray(values, dtype=np.float64), spec.quat_order, "wxyz"
            )
    if offset != len(blob):
        raise ValueError("trailing bytes in BTRJ stream")

    for frame_index, frame_rot in enumerate(rotations):
        for joint_index, quat in enumerate(frame_rot):
            check_quat(quat, joint=joint_index, frame=frame_index)
    rotations = align_sequence(rotations)
    return Animation(
        joint_names=tuple(joint_names),
        times=times,
        rotations=rotations,
        root_translations=roots,
        interp=interp,
        quat_order="wxyz",
        parameter_version=parameter_version,
    )


def load_binary(
    path: str,
    joint_names: tuple[str, ...],
    *,
    interp: str = "slerp",
    parameter_version: str = "baseline",
) -> Animation:
    with open(path, "rb") as handle:
        blob = handle.read()
    return decode_binary(blob, joint_names, interp=interp, parameter_version=parameter_version)
