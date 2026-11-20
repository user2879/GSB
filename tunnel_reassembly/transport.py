"""真实 127.0.0.1 回环 UDP 套接字收发。"""

from __future__ import annotations

import socket

from .clock import Clock, MonotonicClock
from .errors import BusinessError, E_PROTOCOL
from .model import ChannelSpec, FragmentEvent
from .protocol import PacketCodec


def assert_port_free(host: str, port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise BusinessError(E_PROTOCOL, f"端口 {host}:{port} 已被占用: {exc}")
    finally:
        probe.close()


class UdpReceiver:
    """在通道回环端口上阻塞接收数据报，用注入时钟记录到达时刻。"""

    def __init__(self, spec: ChannelSpec, clock: Clock | None = None) -> None:
        self.spec = spec
        self.clock = clock or MonotonicClock()
        self.codec = PacketCodec()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((spec.host, spec.port))
        self.sock.settimeout(2.0)

    def recv_one(self) -> FragmentEvent | None:
        try:
            datagram, _addr = self.sock.recvfrom(65535)
        except socket.timeout:
            return None
        arrive_ms = self.clock.now_ms()
        fields = self.codec.decode(datagram, self.spec.payload_limit)
        if fields["channel_id"] != self.spec.channel_id:
            raise BusinessError(
                E_PROTOCOL,
                f"数据报通道标识 {fields['channel_id']} 与监听通道 {self.spec.channel_id} 不符",
            )
        return FragmentEvent(
            channel_id=self.spec.channel_id,
            group_seq=fields["group_seq"],
            frag_offset=fields["frag_offset"],
            frag_length=fields["frag_length"],
            total_frags=fields["total_frags"],
            send_ms=fields["send_ms"],
            arrive_ms=arrive_ms,
            payload=fields["payload"],
        )

    def close(self) -> None:
        self.sock.close()


class UdpSender:
    """按发送脚本把数据报发往回环端口（演示与真实链路验证用）。"""

    def __init__(self) -> None:
        self.codec = PacketCodec()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_fragment(self, spec: ChannelSpec, frag: dict) -> None:
        datagram = self.codec.encode(
            channel_id=spec.channel_id,
            group_seq=frag["group_seq"],
            frag_offset=frag["frag_offset"],
            frag_length=frag["frag_length"],
            total_frags=frag["total_frags"],
            send_ms=frag["send_ms"],
            payload=frag["payload"],
        )
        self.sock.sendto(datagram, (spec.host, spec.port))

    def close(self) -> None:
        self.sock.close()
