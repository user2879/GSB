"""Deterministic uncompressed AVI writing and strict streaming reads."""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

from .errors import frame_format


def _chunk(chunk_id: bytes, data: bytes) -> bytes:
    padding = b"\x00" if len(data) % 2 else b""
    return chunk_id + struct.pack("<I", len(data)) + data + padding


def _list(list_type: bytes, payload: bytes) -> bytes:
    data = list_type + payload
    return b"LIST" + struct.pack("<I", len(data)) + data


def _avih(width: int, height: int, fps: int, frame_count: int, frame_size: int) -> bytes:
    data = struct.pack(
        "<IIIIIIIIII4I",
        int(round(1_000_000 / fps)),
        frame_size * fps,
        0,
        0x10,
        frame_count,
        0,
        1,
        frame_size,
        width,
        height,
        0,
        0,
        0,
        0,
    )
    return _chunk(b"avih", data)


def _strh(width: int, height: int, fps: int, frame_count: int, frame_size: int) -> bytes:
    data = struct.pack(
        "<4s4sIHHIIIIIIII4h",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        fps,
        0,
        frame_count,
        frame_size,
        0xFFFFFFFF,
        frame_size,
        0,
        0,
        width,
        height,
    )
    return _chunk(b"strh", data)


def _strf(width: int, height: int, frame_size: int) -> bytes:
    data = struct.pack(
        "<IiiHHIIiiII",
        40,
        width,
        -height,
        1,
        24,
        0,
        frame_size,
        0,
        0,
        0,
        0,
    )
    return _chunk(b"strf", data)


class DeterministicAVIWriter:
    """Writes an indexed BGR24 AVI with no timestamp or platform metadata."""

    def __init__(self, path: Path, width: int, height: int, fps: int, frame_count: int):
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            raise frame_format(
                "invalid AVI dimensions or timeline",
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
            )
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.frame_size = width * height * 3
        self._written_frames = 0
        self._digest = hashlib.sha256()
        self._temporary = self.path.with_name(f".{self.path.name}.pending")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._temporary.unlink()
        except FileNotFoundError:
            pass
        self._handle: BinaryIO = self._temporary.open("wb")
        self._write(self._prologue())

    def _prologue(self) -> bytes:
        avih = _avih(
            self.width, self.height, self.fps, self.frame_count, self.frame_size
        )
        strh = _strh(
            self.width, self.height, self.fps, self.frame_count, self.frame_size
        )
        strf = _strf(self.width, self.height, self.frame_size)
        strl = _list(b"strl", strh + strf)
        hdrl = _list(b"hdrl", avih + strl)
        frame_block = 8 + self.frame_size + (self.frame_size % 2)
        movi_payload_size = 4 + self.frame_count * frame_block
        movi_header = b"LIST" + struct.pack("<I", movi_payload_size) + b"movi"
        index_chunk = _chunk(b"idx1", self._index_bytes())
        riff_payload_size = (
            4 + len(hdrl) + len(movi_header) + self.frame_count * frame_block + len(index_chunk)
        )
        if riff_payload_size > 0xFFFFFFFF:
            raise frame_format(
                "AVI exceeds the RIFF size limit",
                width=self.width,
                height=self.height,
                frame_count=self.frame_count,
            )
        return (
            b"RIFF"
            + struct.pack("<I", riff_payload_size)
            + b"AVI "
            + hdrl
            + movi_header
        )

    def _index_bytes(self) -> bytes:
        entries = []
        stride = 8 + self.frame_size + (self.frame_size % 2)
        for index in range(self.frame_count):
            entries.append(
                struct.pack("<4sIII", b"00db", 0x10, 4 + index * stride, self.frame_size)
            )
        return b"".join(entries)

    def _write(self, data: bytes | bytearray | memoryview) -> None:
        view = memoryview(data)
        self._handle.write(view)
        self._digest.update(view)

    def add_frame(self, frame: np.ndarray) -> bytes:
        if self._written_frames >= self.frame_count:
            raise frame_format(
                "too many frames supplied to AVI writer",
                expected=self.frame_count,
                received=self._written_frames + 1,
            )
        if not isinstance(frame, np.ndarray):
            raise frame_format(
                "frame must be a NumPy array",
                frame_index=self._written_frames,
                actual_type=type(frame).__name__,
            )
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise frame_format(
                "frame has an invalid dtype or shape",
                frame_index=self._written_frames,
                expected_shape=[self.height, self.width, 3],
                actual_shape=list(frame.shape),
                expected_dtype="uint8",
                actual_dtype=str(frame.dtype),
            )
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        data = frame.tobytes(order="C")
        padding = b"\x00" if len(data) % 2 else b""
        self._write(b"00db" + struct.pack("<I", len(data)) + data + padding)
        self._written_frames += 1
        return data

    def close(self) -> tuple[str, int]:
        if self._written_frames != self.frame_count:
            self._handle.close()
            try:
                self._temporary.unlink()
            except FileNotFoundError:
                pass
            raise frame_format(
                "AVI frame count did not match the declared timeline",
                expected=self.frame_count,
                actual=self._written_frames,
            )
        self._write(_chunk(b"idx1", self._index_bytes()))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        os.replace(self._temporary, self.path)
        if os.name != "nt":
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return self._digest.hexdigest(), self.path.stat().st_size

    def abort(self) -> None:
        try:
            self._handle.close()
        finally:
            try:
                self._temporary.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "DeterministicAVIWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class AVIVideoReader:
    """Strict AVI reader that validates every frame's shape and byte count."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise frame_format("video file is missing", path=str(self.path))
        self._handle = self.path.open("rb")
        self.width = 0
        self.height = 0
        self.fps = 0
        self.frame_count = 0
        self.frame_size = 0
        self._offsets: list[int] = []
        try:
            self._parse()
        except BaseException:
            self._handle.close()
            raise

    def _read_exact(self, size: int, *, context: str) -> bytes:
        data = self._handle.read(size)
        if len(data) != size:
            raise frame_format(
                "unexpected end of AVI file",
                path=str(self.path),
                context=context,
                expected_bytes=size,
                actual_bytes=len(data),
                offset=self._handle.tell(),
            )
        return data

    def _parse(self) -> None:
        header = self._read_exact(12, context="RIFF header")
        if header[:4] != b"RIFF" or header[8:12] != b"AVI ":
            raise frame_format(
                "file is not a RIFF AVI",
                path=str(self.path),
                riff_id=header[:4].decode("latin-1"),
                form_type=header[8:12].decode("latin-1"),
            )
        riff_end = 8 + struct.unpack("<I", header[4:8])[0]
        movi_data_start = -1
        movi_data_end = -1
        while self._handle.tell() + 8 <= riff_end:
            chunk_header = self._read_exact(8, context="top-level chunk")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk_start = self._handle.tell()
            if chunk_id == b"LIST":
                list_type = self._read_exact(4, context="LIST type")
                if list_type == b"hdrl":
                    self._parse_hdrl(chunk_start + chunk_size)
                elif list_type == b"movi":
                    movi_data_start = self._handle.tell()
                    movi_data_end = chunk_start + chunk_size
                    self._build_offsets(movi_data_start, movi_data_end)
            self._handle.seek(chunk_start + chunk_size + (chunk_size % 2))
        if movi_data_start < 0:
            raise frame_format("AVI has no movi list", path=str(self.path))
        if self.frame_count != len(self._offsets):
            raise frame_format(
                "AVI index frame count mismatch",
                path=str(self.path),
                declared=self.frame_count,
                found=len(self._offsets),
            )

    def _parse_hdrl(self, end_offset: int) -> None:
        found_avih = False
        while self._handle.tell() + 8 <= end_offset:
            chunk_header = self._read_exact(8, context="hdrl chunk")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk_start = self._handle.tell()
            if chunk_id == b"avih":
                data = self._read_exact(chunk_size, context="avih")
                values = struct.unpack("<IIIIIIIIII4I", data)
                micros_per_frame = values[0]
                self.frame_count = values[4]
                self.width = values[8]
                self.height = values[9]
                if micros_per_frame <= 0:
                    raise frame_format("invalid AVI frame duration", value=micros_per_frame)
                self.fps = int(round(1_000_000 / micros_per_frame))
                self.frame_size = self.width * self.height * 3
                found_avih = True
            self._handle.seek(chunk_start + chunk_size + (chunk_size % 2))
        if not found_avih:
            raise frame_format("AVI header has no avih chunk", path=str(self.path))

    def _build_offsets(self, start: int, end: int) -> None:
        self._handle.seek(start)
        offsets: list[int] = []
        while self._handle.tell() + 8 <= end:
            chunk_header = self._read_exact(8, context="movi chunk")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            data_offset = self._handle.tell()
            if chunk_id in (b"00db", b"00dc"):
                if chunk_size != self.frame_size:
                    raise frame_format(
                        "movi frame chunk has an unexpected size",
                        path=str(self.path),
                        frame_index=len(offsets),
                        expected=self.frame_size,
                        actual=chunk_size,
                    )
                offsets.append(data_offset)
            self._handle.seek(data_offset + chunk_size + (chunk_size % 2))
        self._offsets = offsets

    def read_frame(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.frame_count:
            raise frame_format(
                "frame index is outside the source",
                frame_index=index,
                frame_count=self.frame_count,
            )
        self._handle.seek(self._offsets[index])
        data = self._read_exact(self.frame_size, context=f"frame {index}")
        return np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        ).copy()

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for index in range(self.frame_count):
            yield index, self.read_frame(index)

    def validate_metadata(
        self, *, width: int, height: int, fps: int, frame_count: int
    ) -> None:
        actual = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
        }
        expected = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        }
        if actual != expected:
            raise frame_format(
                "AVI metadata differs from the source manifest",
                path=str(self.path),
                expected=expected,
                actual=actual,
            )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "AVIVideoReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
