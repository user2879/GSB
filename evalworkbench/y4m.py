"""Small, dependency-free YUV4MPEG2 writer/reader used for lossless fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

import numpy as np

from .errors import BusinessError
from .util import atomic_binary_stream


@dataclass
class Y4MFrame:
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


def write_y4m(path: Path, width: int, height: int, fps: int, frames: Iterable[Y4MFrame]) -> None:
    if width % 2 or height % 2:
        raise ValueError("420 output requires even dimensions")
    header = f"YUV4MPEG2 W{width} H{height} F{fps}:1 Ip A1:1 C420mpeg2\n".encode("ascii")
    with atomic_binary_stream(path) as handle:
        handle.write(header)
        for frame in frames:
            handle.write(b"FRAME\n")
            handle.write(np.ascontiguousarray(frame.y, dtype=np.uint8).tobytes())
            handle.write(np.ascontiguousarray(frame.u, dtype=np.uint8).tobytes())
            handle.write(np.ascontiguousarray(frame.v, dtype=np.uint8).tobytes())


class Y4MReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.handle: BinaryIO = path.open("rb")
        except OSError as exc:
            raise BusinessError("E_DECODE", f"cannot open Y4M {path}", {"path": str(path)}) from exc
        header = self.handle.readline()
        if not header.startswith(b"YUV4MPEG2 "):
            self.handle.close()
            raise BusinessError("E_DECODE", "Y4M magic is missing", {"path": str(path)})
        fields = header.decode("ascii", errors="strict").strip().split()[1:]
        parsed: dict[str, str] = {}
        for field in fields:
            if field:
                parsed[field[0]] = field[1:]
        try:
            self.width = int(parsed["W"])
            self.height = int(parsed["H"])
            self.fps_num, self.fps_den = (int(part) for part in parsed["F"].split(":", 1))
        except (KeyError, ValueError) as exc:
            self.handle.close()
            raise BusinessError("E_DECODE", "invalid Y4M header", {"path": str(path)}) from exc
        self.chroma = parsed.get("C", "420")
        if not self.chroma.startswith("420"):
            self.handle.close()
            raise BusinessError(
                "E_DECODE",
                "this workbench receives yuv420p fixtures",
                {"path": str(path), "chroma": self.chroma},
            )

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "Y4MReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _read_frame_once(self) -> Y4MFrame:
        line = self.handle.readline()
        if not line:
            raise EOFError("end of Y4M stream")
        if not line.startswith(b"FRAME"):
            raise ValueError("missing FRAME marker")
        y_size = self.width * self.height
        c_size = (self.width // 2) * (self.height // 2)
        y_data = self.handle.read(y_size)
        u_data = self.handle.read(c_size)
        v_data = self.handle.read(c_size)
        if len(y_data) != y_size or len(u_data) != c_size or len(v_data) != c_size:
            raise ValueError("truncated Y4M frame")
        return Y4MFrame(
            np.frombuffer(y_data, dtype=np.uint8).reshape(self.height, self.width).copy(),
            np.frombuffer(u_data, dtype=np.uint8).reshape(self.height // 2, self.width // 2).copy(),
            np.frombuffer(v_data, dtype=np.uint8).reshape(self.height // 2, self.width // 2).copy(),
        )

    def read_frame(self, retries: int = 2) -> tuple[Y4MFrame | None, int, str | None]:
        """Read one frame with bounded, same-offset retries."""

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            position = self.handle.tell()
            try:
                return self._read_frame_once(), attempt, None
            except EOFError:
                return None, attempt, "E_EOF"
            except Exception as exc:  # frame isolation is deliberate
                last_error = exc
                try:
                    self.handle.seek(position)
                except OSError:
                    break
        return None, retries, f"E_DECODE:{type(last_error).__name__}"


def iter_y4m(path: Path) -> Iterator[Y4MFrame]:
    with Y4MReader(path) as reader:
        while True:
            frame, _attempt, error = reader.read_frame(retries=0)
            if frame is None:
                if error == "E_EOF":
                    return
                raise BusinessError("E_DECODE", error or "frame decode failed", {"path": str(path)})
            yield frame
