"""环形 reduce-scatter 与 all-gather 的纯算法层。

该层只依赖 RingTransport 接口，不关心底层是真实回环 TCP 还是进程内队列。
设计保证：
  * 压缩后整数沿环以固定块大小切块转发，累加在整数域精确完成；
  * 消息按 (chunk, round, block) 显式编号，乱序到达时先入缓存再归位；
  * 阶段标记不匹配、重复块、越界块等协议违规一律 E_PROTO；
  * 最终 reduce-scatter 的归约结果归属 rank = chunk-1 (mod n)，
    all-gather 后每个 rank 拼出完整全局梯度。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import protocol as proto
from .errors import E_PROTO, GradCompError


def _proto(message: str, details: dict | None = None):
    raise GradCompError(E_PROTO, message, details=details)


@dataclass
class RingCounters:
    sent_blocks: int = 0
    recv_blocks: int = 0
    phases: list[str] = field(default_factory=list)

    def record_phase(self, phase: str) -> None:
        if not self.phases or self.phases[-1] != phase:
            self.phases.append(phase)

    def to_dict(self) -> dict:
        return {"sent_blocks": self.sent_blocks,
                "recv_blocks": self.recv_blocks, "phases": list(self.phases)}


def split_blocks(chunk: np.ndarray, block_size: int) -> list[np.ndarray]:
    return [chunk[i:i + block_size] for i in range(0, chunk.size, block_size)]


def _expect_phase(msg: proto.Message, phase: int, step: int) -> None:
    if msg.phase != phase:
        _proto("阶段标记不匹配",
               {"expected": proto.phase_name(phase),
                "got": proto.phase_name(msg.phase), "step": step})
    if msg.step != step:
        _proto("训练步标记不匹配",
               {"expected_step": step, "got_step": msg.step})


def _expect_data(msg: proto.Message, phase: int, step: int) -> None:
    _expect_phase(msg, phase, step)
    if msg.msg_type != proto.MSG_DATA:
        _proto("数据阶段收到非 DATA 消息", {"msg_type": msg.msg_type})


class RingEngine:
    def __init__(self, rank: int, n: int, block_size: int, transport,
                 counters: RingCounters | None = None):
        self.rank = rank
        self.n = n
        self.block_size = block_size
        self.transport = transport
        self.counters = counters or RingCounters()
        self.next_rank = (rank + 1) % n
        self._pending: dict[tuple[int, int, int], np.ndarray] = {}

    def _data(self, phase: int, step: int, chunk: int, rnd: int,
              block: np.ndarray, block_index: int) -> None:
        self.transport.send(self.next_rank, proto.Message(
            msg_type=proto.MSG_DATA, phase=phase, step=step,
            src=self.rank, dst=self.next_rank, chunk=chunk,
            round_index=rnd, block=block_index, payload=block))
        self.counters.sent_blocks += 1

    def reduce_scatter(self, local_chunks: list[np.ndarray],
                       step: int) -> list[np.ndarray | None]:
        """标准环形 reduce-scatter；返回长度 n 列表，仅归属槽位非空。"""
        self.counters.record_phase(proto.phase_name(proto.PHASE_REDUCE_SCATTER))
        buffers = [c.copy() for c in local_chunks]
        seen: set[tuple[int, int, int]] = set()

        for rnd in range(self.n - 1):
            send_chunk = (self.rank - rnd) % self.n
            recv_chunk = (self.rank - rnd - 1) % self.n
            blocks = split_blocks(buffers[send_chunk], self.block_size)
            for b, block in enumerate(blocks):
                self._data(proto.PHASE_REDUCE_SCATTER, step, send_chunk,
                           rnd, block, b)
            for b in range(len(blocks)):
                incoming = self._recv_ordered(
                    step, proto.PHASE_REDUCE_SCATTER, recv_chunk, rnd, b, seen)
                self._add_block(buffers[recv_chunk], b, incoming)
                self.counters.recv_blocks += 1

        owner = (self.rank + 1) % self.n  # rank r 最终持有 chunk r-1
        result: list[np.ndarray | None] = [None] * self.n
        result[owner] = buffers[owner]
        return result

    def all_gather(self, owned: list[np.ndarray | None],
                   step: int) -> np.ndarray:
        """环形 all-gather：把归属块沿环广播，拼出完整全局梯度。"""
        self.counters.record_phase(proto.phase_name(proto.PHASE_ALL_GATHER))
        owner = (self.rank + 1) % self.n
        if owned[owner] is None:
            _proto("all-gather 起始缺少本 rank 归属块",
                   {"rank": self.rank, "owner_chunk": owner})
        queues: dict[int, list[np.ndarray]] = {
            c: [] for c in range(self.n) if c != owner}
        current = owned[owner].copy()
        current_chunk = owner
        seen: set[tuple[int, int, int]] = set()

        for rnd in range(self.n - 1):
            blocks = split_blocks(current, self.block_size)
            for b, block in enumerate(blocks):
                self._data(proto.PHASE_ALL_GATHER, step, current_chunk,
                           rnd, block, b)
            recv_chunk = (current_chunk - 1) % self.n
            for b in range(len(blocks)):
                incoming = self._recv_ordered(
                    step, proto.PHASE_ALL_GATHER, recv_chunk, rnd, b, seen)
                if recv_chunk == owner:
                    _proto("all-gather 收到已归属块的冗余副本",
                           {"chunk": recv_chunk})
                queues[recv_chunk].append(incoming)
                self.counters.recv_blocks += 1
            current = np.concatenate(queues[recv_chunk])
            current_chunk = recv_chunk

        ordered: list[np.ndarray] = []
        for c in range(self.n):
            if c == owner:
                ordered.append(owned[c])
            else:
                ordered.append(np.concatenate(queues[c]))
        return np.concatenate(ordered)

    def barrier(self, step: int) -> None:
        """空负载阶段标记沿环转一圈，作为阶段边界与协议检查点。"""
        self.counters.record_phase(proto.phase_name(proto.PHASE_BARRIER))
        self.transport.send(self.next_rank, proto.Message(
            msg_type=proto.MSG_PHASE, phase=proto.PHASE_BARRIER, step=step,
            src=self.rank, dst=self.next_rank, chunk=0, round_index=0,
            block=0, payload=np.zeros(0, dtype=np.int64)))
        msg = self.transport.recv(self.rank)
        _expect_phase(msg, proto.PHASE_BARRIER, step)
        if msg.msg_type != proto.MSG_PHASE:
            _proto("barrier 收到非 PHASE 消息", {"msg_type": msg.msg_type})

    def _recv_ordered(self, step: int, phase: int, expect_chunk: int,
                      rnd: int, block: int,
                      seen: set[tuple[int, int, int]]) -> np.ndarray:
        key = (expect_chunk, rnd, block)
        if key in seen:
            _proto("收到重复数据块",
                   {"chunk": expect_chunk, "round": rnd, "block": block})
        if key in self._pending:
            seen.add(key)
            return self._pending.pop(key)

        while True:
            msg = self.transport.recv(self.rank)
            _expect_data(msg, phase, step)
            recv_key = (msg.chunk, msg.round_index, msg.block)
            if recv_key in seen or recv_key in self._pending:
                _proto("收到重复数据块",
                       {"chunk": msg.chunk, "round": msg.round_index,
                        "block": recv_key[2]})
            if recv_key == key:
                seen.add(key)
                return msg.payload
            # 乱序提前到达的其它 (chunk, round, block) 先缓存，之后归位。
            self._pending[recv_key] = msg.payload

    @staticmethod
    def _add_block(buffer: np.ndarray, block_index: int,
                   block: np.ndarray) -> None:
        start = block_index * block.size
        end = start + block.size
        if end > buffer.size:
            _proto("数据块超出分片边界",
                   {"block_size": block.size, "buffer_size": buffer.size})
        if block.dtype != buffer.dtype:
            _proto("累加块 dtype 与缓冲区不一致",
                   {"buffer_dtype": str(buffer.dtype),
                    "block_dtype": str(block.dtype)})
        buffer[start:end] = buffer[start:end] + block
