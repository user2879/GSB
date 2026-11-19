"""Loopback device transport, budgeted concurrent dispatch, and receipts."""

from __future__ import annotations

import base64
import ipaddress
import json
import os
import socket
import socketserver
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .codec import ProtocolFormatError, WireFrame, decode_frame, encode_frame, transcode_frame
from .constants import CHANNEL_INDEX, PROTOCOL_V1, PROTOCOL_V2
from .frames import FrameRecord, digest_encoded_frames
from .quantization import DEFAULT_QUANT_PROFILE, QuantProfile

LENGTH = struct.Struct(">I")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class TransportError(RuntimeError):
    pass


class DispatchTermination(RuntimeError):
    """Raised by the deterministic batch-number termination hook."""


def _assert_loopback(host: str) -> None:
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(f"invalid loopback host: {host!r}") from exc
    if not address.is_loopback:
        raise ValueError(f"refusing non-loopback host: {host!r}")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("connection closed before complete message")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise TransportError("message exceeds MAX_MESSAGE_BYTES")
    sock.sendall(LENGTH.pack(len(data)) + data)


def receive_message(sock: socket.socket) -> dict[str, Any]:
    (length,) = LENGTH.unpack(_recv_exact(sock, LENGTH.size))
    if length > MAX_MESSAGE_BYTES:
        raise TransportError("incoming message exceeds MAX_MESSAGE_BYTES")
    data = _recv_exact(sock, length)
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise TransportError("message root must be an object")
    return value


@dataclass(frozen=True)
class ReceiptChannel:
    channel: str
    amplitude: float
    frequency_hz: float | None
    amplitude_code: int
    frequency_code: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "amplitude": self.amplitude,
            "frequency_hz": self.frequency_hz,
            "amplitude_code": self.amplitude_code,
            "frequency_code": self.frequency_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReceiptChannel":
        frequency_code = value.get("frequency_code")
        frequency_hz = value.get("frequency_hz")
        return cls(
            channel=str(value["channel"]),
            amplitude=float(value["amplitude"]),
            frequency_hz=None if frequency_hz is None else float(frequency_hz),
            amplitude_code=int(value["amplitude_code"]),
            frequency_code=None if frequency_code is None else int(frequency_code),
        )


@dataclass(frozen=True)
class DeviceReceipt:
    seq: int
    timestamp_us: int
    protocol_version: int
    channels: tuple[ReceiptChannel, ...]
    mask: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_us": self.timestamp_us,
            "protocol_version": self.protocol_version,
            "channels": [channel.to_dict() for channel in self.channels],
            "mask": self.mask,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeviceReceipt":
        return cls(
            seq=int(value["seq"]),
            timestamp_us=int(value["timestamp_us"]),
            protocol_version=int(value["protocol_version"]),
            channels=tuple(ReceiptChannel.from_dict(item) for item in value["channels"]),
            mask=int(value["mask"]),
        )

    @property
    def channel_by_name(self) -> dict[str, ReceiptChannel]:
        return {channel.channel: channel for channel in self.channels}


@dataclass(frozen=True)
class BatchReceipt:
    receipts: tuple[DeviceReceipt, ...]
    observed_batch_digest: str
    request_id: str


def _receipt_from_wire(
    frame: WireFrame,
    profile: QuantProfile,
) -> DeviceReceipt:
    return DeviceReceipt(
        seq=frame.seq,
        timestamp_us=frame.timestamp_us,
        protocol_version=frame.protocol_version,
        mask=frame.mask,
        channels=tuple(
            ReceiptChannel(
                channel=item.channel,
                amplitude=profile.decode_amplitude(item.amplitude_code),
                frequency_hz=(
                    None
                    if item.frequency_code is None
                    else profile.decode_frequency(item.frequency_code)
                ),
                amplitude_code=item.amplitude_code,
                frequency_code=item.frequency_code,
            )
            for item in frame.channels
        ),
    )


class MockDevice:
    def __init__(
        self,
        profile: QuantProfile = DEFAULT_QUANT_PROFILE,
        drop_sequences: Iterable[int] = (),
        downgrade_protocol: int | None = None,
    ) -> None:
        self.profile = profile
        self.drop_sequences = set(int(seq) for seq in drop_sequences)
        self.downgrade_protocol = downgrade_protocol

    def process_batch(self, encoded_frames: Sequence[bytes], request_id: str) -> dict[str, Any]:
        receipts: list[DeviceReceipt] = []
        accepted: list[tuple[int, bytes]] = []
        protocol_stamp: int | None = None
        errors: list[dict[str, Any]] = []
        for raw in encoded_frames:
            try:
                frame = decode_frame(raw)
            except ProtocolFormatError as exc:
                errors.append(exc.to_dict())
                continue
            if protocol_stamp is None:
                protocol_stamp = frame.protocol_version
            if protocol_stamp != frame.protocol_version:
                errors.append(
                    {
                        "code": "E_PROTO_FORMAT",
                        "frame_seq": frame.seq,
                        "field": "protocol_version",
                        "detail": "mixed protocol versions in request",
                    }
                )
                continue
            accepted.append((frame.seq, raw))
            if frame.seq in self.drop_sequences:
                continue
            response_frame = frame
            if self.downgrade_protocol is not None:
                response_frame = transcode_frame(frame, self.downgrade_protocol)
            receipts.append(_receipt_from_wire(response_frame, self.profile))
        version = protocol_stamp if protocol_stamp is not None else PROTOCOL_V2
        digest = digest_encoded_frames(accepted, version)
        return {
            "request_id": request_id,
            "receipts": [receipt.to_dict() for receipt in receipts],
            "observed_batch_digest": digest,
            "errors": errors,
        }


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _make_handler(device: MockDevice, delay_ms: float) -> type[socketserver.BaseRequestHandler]:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                request = receive_message(self.request)
                encoded_frames = [base64.b64decode(item) for item in request.get("frames", [])]
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)
                response = device.process_batch(encoded_frames, str(request.get("request_id", "")))
                send_message(self.request, response)
            except Exception as exc:  # The process must keep serving after a bad request.
                try:
                    send_message(
                        self.request,
                        {
                            "request_id": "",
                            "receipts": [],
                            "observed_batch_digest": "",
                            "errors": [
                                {
                                    "code": "E_TRANSPORT",
                                    "frame_seq": -1,
                                    "field": "request",
                                    "detail": str(exc),
                                }
                            ],
                        },
                    )
                except Exception:
                    pass

    return Handler


class LoopbackServer:
    """In-process server for tests.  The CLI also offers a separate process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        profile: QuantProfile = DEFAULT_QUANT_PROFILE,
        drop_sequences: Iterable[int] = (),
        downgrade_protocol: int | None = None,
        delay_ms: float = 0.0,
    ) -> None:
        _assert_loopback(host)
        self.device = MockDevice(profile, drop_sequences, downgrade_protocol)
        self.server = _ThreadingTCPServer((host, port), _make_handler(self.device, delay_ms))
        self.host, self.port = self.server.server_address[:2]
        self._thread: threading.Thread | None = None

    def start(self) -> "LoopbackServer":
        self._thread = threading.Thread(target=self.server.serve_forever, name="haptic-mock-device", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "LoopbackServer":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


class LoopbackClient:
    def __init__(self, host: str, port: int, timeout_s: float = 2.0) -> None:
        _assert_loopback(host)
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._request_counter = 0
        self._counter_lock = threading.Lock()

    def _next_request_id(self) -> str:
        with self._counter_lock:
            self._request_counter += 1
            return f"req-{self._request_counter:08d}"

    def dispatch_batch(self, frames: Sequence[FrameRecord]) -> BatchReceipt:
        request_id = self._next_request_id()
        payload = {
            "request_id": request_id,
            "protocol_version": frames[0].protocol_version if frames else None,
            "frames": [base64.b64encode(frame.encode()).decode("ascii") for frame in frames],
        }
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.settimeout(self.timeout_s)
            send_message(sock, payload)
            response = receive_message(sock)
        if response.get("errors"):
            raise TransportError(f"device reported errors: {response['errors']}")
        receipts = tuple(sorted((DeviceReceipt.from_dict(item) for item in response.get("receipts", [])), key=lambda item: item.seq))
        return BatchReceipt(
            receipts=receipts,
            observed_batch_digest=str(response.get("observed_batch_digest", "")),
            request_id=str(response.get("request_id", request_id)),
        )


@dataclass(frozen=True)
class DispatchConfig:
    max_workers: int = 4
    batch_memory_limit_bytes: int = 128 * 1024
    batch_time_budget_ms: float = 250.0
    batch_size: int = 16
    estimated_frame_ms: float = 0.35
    estimated_batch_overhead_ms: float = 1.0
    socket_timeout_s: float = 2.0
    abort_after_batches: int | None = None
    hard_abort: bool = False

    def __post_init__(self) -> None:
        if self.max_workers <= 0 or self.batch_size <= 0:
            raise ValueError("max_workers and batch_size must be positive")
        if self.batch_memory_limit_bytes <= 0 or self.batch_time_budget_ms <= 0:
            raise ValueError("budget limits must be positive")
        if self.abort_after_batches is not None and self.abort_after_batches <= 0:
            raise ValueError("abort_after_batches must be positive when supplied")


@dataclass(frozen=True)
class Degradation:
    reason: str
    priority: int
    before: str
    after: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "priority": self.priority,
            "before": self.before,
            "after": self.after,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DispatchBatch:
    index: int
    frames: tuple[FrameRecord, ...]
    estimated_bytes: int
    elapsed_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class DispatchResult:
    receipts: tuple[DeviceReceipt, ...]
    expected_sequences: tuple[int, ...]
    missing_sequences: tuple[int, ...]
    duplicate_receipt_sequences: tuple[int, ...]
    sequence_gaps: tuple[int, ...]
    batches: tuple[DispatchBatch, ...]
    degradations: tuple[Degradation, ...]
    effective_workers: int
    observed_batch_digests: tuple[str, ...]
    observed_batch_ranges: tuple[tuple[int, int], ...]
    dispatch_errors: tuple[str, ...]

    @property
    def receipts_by_seq(self) -> dict[int, DeviceReceipt]:
        return {receipt.seq: receipt for receipt in self.receipts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_sequences": list(self.expected_sequences),
            "missing_sequences": list(self.missing_sequences),
            "duplicate_receipt_sequences": list(self.duplicate_receipt_sequences),
            "sequence_gaps": list(self.sequence_gaps),
            "degradations": [item.to_dict() for item in self.degradations],
            "effective_workers": self.effective_workers,
            "observed_batch_digests": list(self.observed_batch_digests),
            "observed_batch_ranges": [list(item) for item in self.observed_batch_ranges],
            "dispatch_errors": list(self.dispatch_errors),
            "batches": [
                {
                    "index": batch.index,
                    "first_seq": batch.frames[0].seq,
                    "last_seq": batch.frames[-1].seq,
                    "frame_count": len(batch.frames),
                    "estimated_bytes": batch.estimated_bytes,
                    "elapsed_ms": batch.elapsed_ms,
                    "error": batch.error,
                }
                for batch in self.batches
            ],
        }


def estimated_wire_message_bytes(frames: Sequence[FrameRecord]) -> int:
    encoded = sum(len(frame.encode()) for frame in frames)
    base64_bytes = ((encoded + 2) // 3) * 4
    return 256 + base64_bytes + 160 * len(frames)


def plan_dispatch(
    frames: Sequence[FrameRecord],
    config: DispatchConfig,
) -> tuple[list[tuple[FrameRecord, ...]], int, list[Degradation]]:
    ordered = sorted(frames, key=lambda frame: frame.seq)
    if not ordered:
        return [], 0, []
    degradations: list[Degradation] = []
    batch_size = config.batch_size
    max_frame_bytes = max(estimated_wire_message_bytes([frame]) for frame in ordered)
    memory_frames = max(1, config.batch_memory_limit_bytes // max_frame_bytes)
    if max_frame_bytes > config.batch_memory_limit_bytes:
        old = batch_size
        batch_size = 1
        degradations.append(
            Degradation(
                reason="memory_budget_floor",
                priority=1,
                before=f"batch_size={old}",
                after="batch_size=1",
                detail=(
                    f"a single frame requires {max_frame_bytes} estimated bytes, above "
                    f"the {config.batch_memory_limit_bytes}-byte limit; the protocol cannot split a frame"
                ),
            )
        )
    elif batch_size > memory_frames:
        old = batch_size
        batch_size = memory_frames
        degradations.append(
            Degradation(
                reason="memory_budget",
                priority=1,
                before=f"batch_size={old}",
                after=f"batch_size={batch_size}",
                detail=(
                    f"single-frame estimated message is {max_frame_bytes} bytes; "
                    f"limit is {config.batch_memory_limit_bytes} bytes"
                ),
            )
        )
    time_frames = max(
        1,
        int(
            (config.batch_time_budget_ms - config.estimated_batch_overhead_ms)
            / max(config.estimated_frame_ms, 1e-9)
        ),
    )
    if batch_size > time_frames:
        old = batch_size
        batch_size = time_frames
        degradations.append(
            Degradation(
                reason="time_budget",
                priority=2,
                before=f"batch_size={old}",
                after=f"batch_size={batch_size}",
                detail=(
                    f"estimated per-frame time {config.estimated_frame_ms} ms; "
                    f"batch budget {config.batch_time_budget_ms} ms"
                ),
            )
        )
    if config.batch_time_budget_ms <= config.estimated_batch_overhead_ms:
        old = batch_size
        batch_size = 1
        degradations.append(
            Degradation(
                reason="time_budget_floor",
                priority=2,
                before=f"batch_size={old}",
                after="batch_size=1,workers=1",
                detail="time budget is below one estimated batch overhead",
            )
        )

    batches: list[tuple[FrameRecord, ...]] = []
    current: list[FrameRecord] = []
    for frame in ordered:
        trial = current + [frame]
        if current and (
            len(trial) > batch_size
            or estimated_wire_message_bytes(trial) > config.batch_memory_limit_bytes
        ):
            batches.append(tuple(current))
            current = [frame]
        else:
            current = trial
    if current:
        batches.append(tuple(current))

    effective_workers = min(config.max_workers, len(batches))
    if config.batch_time_budget_ms <= config.estimated_batch_overhead_ms:
        effective_workers = min(effective_workers, 1)
    if effective_workers < config.max_workers:
        if batch_size == 1 and config.batch_time_budget_ms <= config.estimated_batch_overhead_ms:
            pass
        else:
            degradations.append(
                Degradation(
                    reason="thread_cap",
                    priority=3,
                    before=f"workers={config.max_workers}",
                    after=f"workers={effective_workers}",
                    detail=f"only {len(batches)} batch(es) are available",
                )
            )
    return batches, effective_workers, degradations


def dispatch_concurrently(
    frames: Sequence[FrameRecord],
    client: LoopbackClient,
    config: DispatchConfig,
) -> DispatchResult:
    ordered = sorted(frames, key=lambda frame: frame.seq)
    expected = tuple(frame.seq for frame in ordered)
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate frame sequence in dispatch input")
    sequence_gaps: tuple[int, ...] = ()
    if expected:
        expected_contiguous = set(range(expected[0], expected[-1] + 1))
        sequence_gaps = tuple(sorted(expected_contiguous - set(expected)))
    planned, effective_workers, degradations = plan_dispatch(ordered, config)
    if not planned:
        return DispatchResult(
            receipts=(),
            expected_sequences=(),
            missing_sequences=(),
            duplicate_receipt_sequences=(),
            sequence_gaps=(),
            batches=(),
            degradations=tuple(degradations),
            effective_workers=effective_workers,
            observed_batch_digests=(),
            observed_batch_ranges=(),
            dispatch_errors=(),
        )

    all_receipts: dict[int, DeviceReceipt] = {}
    duplicate_receipts: set[int] = set()
    completed_batches: list[DispatchBatch] = []
    observed_digests: list[str] = []
    observed_ranges: list[tuple[int, int]] = []
    dispatch_errors: list[str] = []
    current_workers = effective_workers
    index = 0
    with ThreadPoolExecutor(max_workers=max(1, effective_workers), thread_name_prefix="haptic-dispatch") as pool:
        while index < len(planned):
            wave = planned[index : index + current_workers]

            def run_batch(batch_index: int, batch_frames: tuple[FrameRecord, ...]) -> tuple[int, float, BatchReceipt | None, str | None]:
                started = time.perf_counter()
                try:
                    receipt = client.dispatch_batch(batch_frames)
                    return batch_index, (time.perf_counter() - started) * 1000.0, receipt, None
                except Exception as exc:
                    return batch_index, (time.perf_counter() - started) * 1000.0, None, str(exc)

            futures = [
                pool.submit(run_batch, index + offset, wave_frames)
                for offset, wave_frames in enumerate(wave)
            ]
            wave_overrun = False
            for future in futures:
                batch_index, elapsed_ms, batch_receipt, error = future.result()
                batch_frames = planned[batch_index]
                completed_batches.append(
                    DispatchBatch(
                        index=batch_index,
                        frames=batch_frames,
                        estimated_bytes=estimated_wire_message_bytes(batch_frames),
                        elapsed_ms=round(elapsed_ms, 6),
                        error=error,
                    )
                )
                if elapsed_ms > config.batch_time_budget_ms:
                    wave_overrun = True
                if error is not None:
                    dispatch_errors.append(f"batch {batch_index}: {error}")
                    continue
                assert batch_receipt is not None
                observed_digests.append(batch_receipt.observed_batch_digest)
                observed_ranges.append(
                    (batch_frames[0].seq, batch_frames[-1].seq)
                )
                for receipt in batch_receipt.receipts:
                    if receipt.seq in all_receipts:
                        duplicate_receipts.add(receipt.seq)
                    else:
                        all_receipts[receipt.seq] = receipt
                if (
                    config.abort_after_batches is not None
                    and len(completed_batches) >= config.abort_after_batches
                ):
                    if config.hard_abort:
                        os._exit(137)
                    raise DispatchTermination(
                        f"injected termination after {len(completed_batches)} completed batch(es)"
                    )
            if wave_overrun and current_workers > 1:
                old_workers = current_workers
                current_workers = 1
                degradations.append(
                    Degradation(
                        reason="time_budget_overrun",
                        priority=2,
                        before=f"workers={old_workers}",
                        after="workers=1",
                        detail=f"observed batch exceeded {config.batch_time_budget_ms} ms; future waves run serially",
                    )
                )
            index += len(wave)

    receipts = tuple(all_receipts[seq] for seq in sorted(all_receipts))
    missing = tuple(seq for seq in expected if seq not in all_receipts)
    return DispatchResult(
        receipts=receipts,
        expected_sequences=expected,
        missing_sequences=missing,
        duplicate_receipt_sequences=tuple(sorted(duplicate_receipts)),
        sequence_gaps=sequence_gaps,
        batches=tuple(sorted(completed_batches, key=lambda batch: batch.index)),
        degradations=tuple(degradations),
        effective_workers=effective_workers,
        observed_batch_digests=tuple(observed_digests),
        observed_batch_ranges=tuple(observed_ranges),
        dispatch_errors=tuple(dispatch_errors),
    )
