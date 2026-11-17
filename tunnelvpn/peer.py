"""Loopback peer with deterministic behavior injection."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from .codec import FrameDecoder, Reassembler
from .config import PeerDefinition, TunnelConfig, load_config
from .errors import ErrorCode, TunnelError
from .logging import StructuredLogger

MAX_WIRE_FRAME = 65535
LENGTH_PREFIX = struct.Struct(">I")


def recv_exact(sock: socket.socket, count: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except socket.timeout as exc:
            raise TunnelError(
                ErrorCode.E_TIMEOUT, "读取回环套接字超时", details={"remaining": remaining}
            ) from exc
        except OSError as exc:
            raise TunnelError(
                ErrorCode.E_PEER, f"回环套接字读取失败: {exc}"
            ) from exc
        if not chunk:
            if remaining == count:
                return None
            raise TunnelError(
                ErrorCode.E_ENCAP,
                "帧长度前缀之后的连接提前结束",
                offset=count - remaining,
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, frame: bytes) -> None:
    sock.sendall(LENGTH_PREFIX.pack(len(frame)) + frame)


def read_frames(sock: socket.socket) -> list[bytes]:
    frames: list[bytes] = []
    while True:
        raw_length = recv_exact(sock, LENGTH_PREFIX.size)
        if raw_length is None:
            return frames
        (length,) = LENGTH_PREFIX.unpack(raw_length)
        if length == 0:
            raise TunnelError(ErrorCode.E_ENCAP, "零长度线帧非法", offset=0)
        if length > MAX_WIRE_FRAME:
            raise TunnelError(
                ErrorCode.E_ENCAP,
                "线帧长度超过上限",
                offset=0,
                details={"length": length, "max": MAX_WIRE_FRAME},
            )
        payload = recv_exact(sock, length)
        if payload is None:
            raise TunnelError(ErrorCode.E_ENCAP, "线帧缺失负载", offset=0)
        frames.append(payload)


def read_json_line(sock: socket.socket, *, max_bytes: int = 65536) -> dict[str, Any]:
    data = bytearray()
    while len(data) < max_bytes:
        try:
            chunk = sock.recv(1)
        except socket.timeout as exc:
            raise TunnelError(ErrorCode.E_TIMEOUT, "等待对端回送超时") from exc
        except OSError as exc:
            raise TunnelError(ErrorCode.E_PEER, f"读取对端回送失败: {exc}") from exc
        if not chunk:
            raise TunnelError(ErrorCode.E_PEER, "对端在回送完整 JSON 前关闭连接")
        data.extend(chunk)
        if chunk == b"\n":
            try:
                parsed = json.loads(bytes(data).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TunnelError(ErrorCode.E_PEER, "对端回送不是合法 JSON") from exc
            if not isinstance(parsed, dict):
                raise TunnelError(ErrorCode.E_PEER, "对端回送必须是 JSON 对象")
            return parsed
    raise TunnelError(ErrorCode.E_PEER, "对端回送超过大小上限")


@dataclass(frozen=True)
class PeerReply:
    code: str
    tunnel_id: str | None
    sequence: int | None
    payload_bytes: int
    payload_digest: str | None
    frame_digests: tuple[str, ...]
    fragmented: bool
    duplicates: int
    fragment_count: int
    missing_offsets: tuple[int, ...] = ()
    message: str = ""
    error_offset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "tunnel_id": self.tunnel_id,
            "seq": self.sequence,
            "payload_bytes": self.payload_bytes,
            "payload_digest": self.payload_digest,
            "frame_digests": list(self.frame_digests),
            "fragmented": self.fragmented,
            "duplicates": self.duplicates,
            "fragment_count": self.fragment_count,
        }
        if self.missing_offsets:
            result["missing_offsets"] = list(self.missing_offsets)
        if self.error_offset is not None:
            result["error_offset"] = self.error_offset
        return result


class PeerServer:
    """Multi-connection loopback receiver.

    The forwarding engine is unaware of the selected behavior; behavior lives
    entirely in this peer process and its JSON script parameters.
    """

    def __init__(
        self,
        config: TunnelConfig,
        peer: PeerDefinition,
        *,
        log_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.peer = peer
        self.routes = {
            route.id: route for route in config.routes if route.peer_id == peer.id
        }
        if not self.routes:
            raise TunnelError(
                ErrorCode.E_CONFIG,
                "对端没有关联的隧道路由",
                field=f"peers[{peer.id}]",
            )
        self.logger = StructuredLogger(log_path)
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._script_cache: dict[str, Any] | None = None
        self._connection_count = 0
        self._lock = threading.Lock()

    @property
    def address(self) -> tuple[str, int]:
        return self.peer.endpoint.host, self.peer.endpoint.port

    def start(self) -> None:
        if self.peer.behavior == "unreachable":
            self.logger.event(
                "peer.unreachable",
                code=ErrorCode.E_PEER.value,
                peer_id=self.peer.id,
                port=self.peer.endpoint.port,
            )
            return
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(self.address)
            listener.listen(32)
            listener.settimeout(0.2)
        except OSError as exc:
            listener.close()
            raise TunnelError(
                ErrorCode.E_CONFIG,
                f"对端无法绑定回环端口: {exc}",
                field=f"peers[{self.peer.id}].port",
                details={"port": self.peer.endpoint.port},
            ) from exc
        self._listener = listener
        self._thread = threading.Thread(
            target=self._accept_loop,
            name=f"peer-{self.peer.id}",
            daemon=True,
        )
        self._thread.start()
        self.logger.event(
            "peer.started",
            peer_id=self.peer.id,
            behavior=self.peer.behavior,
            host=self.peer.endpoint.host,
            port=self.peer.endpoint.port,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            handler = threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                name=f"peer-{self.peer.id}-conn",
                daemon=True,
            )
            handler.start()

    def _handle_connection(self, conn: socket.socket) -> None:
        with self._lock:
            self._connection_count += 1
            connection_index = self._connection_count
        # The sender may legitimately pause between bandwidth-limited chunks.
        # Response latency is enforced by the engine, not by the receiver's
        # inter-chunk socket timeout.
        conn.settimeout(30.0)
        reply: PeerReply
        try:
            frames = read_frames(conn)
            reply = self._process_frames(frames)
        except TunnelError as exc:
            reply = PeerReply(
                code=exc.code,
                tunnel_id=exc.tunnel_id,
                sequence=(
                    int(exc.details["sequence"])
                    if isinstance(exc.details.get("sequence"), int)
                    else None
                ),
                payload_bytes=0,
                payload_digest=None,
                frame_digests=(),
                fragmented=False,
                duplicates=0,
                fragment_count=0,
                missing_offsets=tuple(exc.details.get("missing_offsets", ())),
                message=exc.message,
                error_offset=exc.offset,
            )
        except Exception as exc:  # keep the peer responsive to malformed clients
            reply = PeerReply(
                code=ErrorCode.E_PEER.value,
                tunnel_id=None,
                sequence=None,
                payload_bytes=0,
                payload_digest=None,
                frame_digests=(),
                fragmented=False,
                duplicates=0,
                fragment_count=0,
                message=str(exc),
            )

        delay_ms = self._behavior_delay_ms(reply)
        if delay_ms:
            time.sleep(delay_ms / 1000)
        if self.peer.behavior == "digest-error" and reply.payload_digest:
            first = reply.payload_digest[:2]
            replacement = "00" if first != "00" else "01"
            reply = PeerReply(
                code=reply.code,
                tunnel_id=reply.tunnel_id,
                sequence=reply.sequence,
                payload_bytes=reply.payload_bytes,
                payload_digest=replacement + reply.payload_digest[2:],
                frame_digests=reply.frame_digests,
                fragmented=reply.fragmented,
                duplicates=reply.duplicates,
                fragment_count=reply.fragment_count,
                missing_offsets=reply.missing_offsets,
                message=reply.message,
                error_offset=reply.error_offset,
            )
        try:
            conn.sendall(
                (json.dumps(reply.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
                .encode("utf-8")
            )
        except OSError:
            pass
        self.logger.event(
            "peer.response",
            code=reply.code,
            peer_id=self.peer.id,
            behavior=self.peer.behavior,
            connection=connection_index,
            tunnel_id=reply.tunnel_id,
            session_sequence=reply.sequence,
            payload_bytes=reply.payload_bytes,
            fragmented=reply.fragmented,
            fragment_count=reply.fragment_count,
            duplicates=reply.duplicates,
            missing_offsets=list(reply.missing_offsets),
            delay_ms=delay_ms,
        )
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()

    def _process_frames(self, frames: list[bytes]) -> PeerReply:
        if not frames:
            raise TunnelError(ErrorCode.E_PEER, "连接未携带任何隧道帧")
        decoder = FrameDecoder()
        reassembler = Reassembler(
            timeout_ms=self.config.global_limits.fragment_timeout_ms
        )
        completed = []
        first_tunnel: str | None = None
        first_sequence: int | None = None
        for frame in frames:
            # The header carries the tunnel id; decode first without route MTU
            # so an unknown tunnel can be reported with its identifier.
            decoded = decoder.decode(frame)
            route = self.routes.get(decoded.tunnel_id)
            if route is None:
                raise TunnelError(
                    ErrorCode.E_ENCAP,
                    "对端未声明该隧道",
                    tunnel_id=decoded.tunnel_id,
                    offset=8,
                )
            if len(frame) > route.mtu:
                raise TunnelError(
                    ErrorCode.E_ENCAP,
                    "线帧超过该隧道 MTU",
                    tunnel_id=decoded.tunnel_id,
                    offset=0,
                    details={"mtu": route.mtu, "length": len(frame)},
                )
            if first_tunnel is None:
                first_tunnel = decoded.tunnel_id
                first_sequence = decoded.sequence
            elif decoded.tunnel_id != first_tunnel or decoded.sequence != first_sequence:
                raise TunnelError(
                    ErrorCode.E_ENCAP,
                    "单次连接混入多个隧道或序号",
                    tunnel_id=decoded.tunnel_id,
                    offset=40,
                )
            message = reassembler.add(decoded)
            if message is not None:
                decoder.mark_complete(message.tunnel_id, message.sequence)
                completed.append(message)

        if not completed:
            # A real timeout window, followed by the explicit missing-offset
            # report required by E_FRAG.
            deadline = time.monotonic() + (
                self.config.global_limits.fragment_timeout_ms / 1000
            )
            while time.monotonic() < deadline:
                time.sleep(0.01)
            errors = reassembler.sweep(force=True)
            if errors:
                raise errors[0]
            raise TunnelError(ErrorCode.E_FRAG, "没有可完成的分片重组")
        if len(completed) > 1:
            raise TunnelError(
                ErrorCode.E_ENCAP,
                "单次连接只能承载一个逻辑消息",
                tunnel_id=first_tunnel,
            )
        pending = reassembler.sweep(force=True)
        if pending:
            raise pending[0]
        message = completed[0]
        return PeerReply(
            code=ErrorCode.OK.value,
            tunnel_id=message.tunnel_id,
            sequence=message.sequence,
            payload_bytes=len(message.payload),
            payload_digest=message.payload_digest,
            frame_digests=message.frame_digests,
            fragmented=message.fragment_count > 1,
            duplicates=message.duplicates,
            fragment_count=message.fragment_count,
        )

    def _behavior_delay_ms(self, reply: PeerReply) -> int:
        if self.peer.behavior != "delay-script":
            return 0
        delay = self.peer.delay_ms
        script = self._load_script()
        if not script or reply.tunnel_id is None:
            return delay
        by_sequence = script.get("by_sequence")
        if isinstance(by_sequence, dict):
            key = f"{reply.tunnel_id}:{reply.sequence}"
            value = by_sequence.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                delay = value
        delays = script.get("delays_ms")
        if isinstance(delays, dict):
            tunnel_delays = delays.get(reply.tunnel_id)
            if isinstance(tunnel_delays, list) and reply.sequence is not None:
                if 0 <= reply.sequence < len(tunnel_delays):
                    value = tunnel_delays[reply.sequence]
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        delay = value
        return delay

    def _load_script(self) -> dict[str, Any] | None:
        if self._script_cache is not None:
            return self._script_cache
        if not self.peer.script:
            self._script_cache = {}
            return self._script_cache
        path = Path(self.peer.script)
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError("script root must be an object")
            self._script_cache = raw
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise TunnelError(
                ErrorCode.E_CONFIG,
                f"无法读取对端脚本: {exc}",
                field=f"peers[{self.peer.id}].script",
            ) from exc
        return self._script_cache


def run_peer(
    config: TunnelConfig,
    peer_id: str,
    log_path: str | Path | None,
    *,
    behavior: str | None = None,
    delay_ms: int | None = None,
    script: str | None = None,
) -> None:
    peers = config.peer_map()
    if peer_id not in peers:
        raise TunnelError(
            ErrorCode.E_CONFIG, "未知对端标识", field="peer_id", details={"peer_id": peer_id}
        )
    peer = peers[peer_id]
    if behavior is not None or delay_ms is not None or script is not None:
        if behavior is not None and behavior not in {
            "normal",
            "delay-script",
            "unreachable",
            "digest-error",
        }:
            raise TunnelError(
                ErrorCode.E_CONFIG,
                "不支持的对端行为",
                field="--behavior",
            )
        if delay_ms is not None and delay_ms < 0:
            raise TunnelError(ErrorCode.E_CONFIG, "delay-ms 不能为负", field="--delay-ms")
        peer = replace(
            peer,
            behavior=behavior if behavior is not None else peer.behavior,
            delay_ms=delay_ms if delay_ms is not None else peer.delay_ms,
            script=script if script is not None else peer.script,
        )
    server = PeerServer(config, peer, log_path=log_path)
    server.start()
    if peer.behavior == "unreachable":
        return
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback tunnel peer")
    parser.add_argument("--config", required=True, help="Route JSON path")
    parser.add_argument("--peer-id", required=True, help="Peer id from config")
    parser.add_argument("--log", default=None, help="Optional JSONL log path")
    parser.add_argument(
        "--behavior",
        choices=["normal", "delay-script", "unreachable", "digest-error"],
        default=None,
        help="Override behavior from JSON without changing the engine",
    )
    parser.add_argument("--delay-ms", type=int, default=None, help="Behavior delay override")
    parser.add_argument("--script", default=None, help="Delay-script JSON override")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        run_peer(
            config,
            args.peer_id,
            args.log,
            behavior=args.behavior,
            delay_ms=args.delay_ms,
            script=args.script,
        )
        return 0
    except TunnelError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
