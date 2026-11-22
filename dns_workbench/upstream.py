"""本地回环上游 TCP 转发与应答链一致性校验。

错误码：
- E_UPSTREAM：连接/传输失败（不可达），按配置重试且重发同一报文；
- E_TXN：事务标识与请求不一致；
- E_CHAIN：问题回显或应答记录名/类型链不自洽。
"""
from __future__ import annotations

import socket
from dataclasses import dataclass

from .errors import ChainError, TxnError, UpstreamError, WireError
from .wire import (
    DnsMessage,
    Question,
    ResourceRecord,
    encode_tcp_frame,
    parse_message,
    read_tcp_frame,
)


@dataclass
class ForwardResult:
    response: DnsMessage
    attempts: int
    consistent: bool
    detail: str


class TcpUpstreamClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5354, *, timeout: float = 2.0, retries: int = 1) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = max(0, retries)

    def forward(self, request: DnsMessage) -> ForwardResult:
        frame = encode_tcp_frame(request)
        last_error: Exception | None = None
        attempts = 0
        for attempt in range(self.retries + 1):
            attempts += 1
            try:
                with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                    sock.sendall(frame)  # 重试保持单次转发语义：字节级同一报文
                    payload = read_tcp_frame(sock)
                response = parse_message(payload)
                self.verify_chain(request, response)
                return ForwardResult(response=response, attempts=attempts, consistent=True, detail="ok")
            except WireError:
                raise
            except (OSError, socket.timeout) as exc:
                last_error = exc
                continue
        raise UpstreamError(
            "本地回环上游不可达",
            context={"host": self.host, "port": self.port, "attempts": attempts, "reason": str(last_error)},
        )

    @staticmethod
    def verify_chain(request: DnsMessage, response: DnsMessage) -> None:
        if response.txn_id != request.txn_id:
            raise TxnError(
                "上游应答事务标识错误",
                fields=["txn_id"],
                context={"expected": request.txn_id, "actual": response.txn_id},
            )
        request_question = request.first_question()
        response_question = response.first_question()
        if request_question is None or response_question is None:
            raise ChainError("问题段缺失", fields=["question"])
        if (
            response_question.qname != request_question.qname
            or response_question.qtype != request_question.qtype
            or response_question.qclass != request_question.qclass
        ):
            raise ChainError(
                "问题回显不自洽",
                fields=["question.qname", "question.qtype", "question.qclass"],
                context={"request": _question_dict(request_question), "response": _question_dict(response_question)},
            )
        qname = request_question.qname
        for index, record in enumerate(response.answers):
            if not isinstance(record, ResourceRecord) or not record.name:
                raise ChainError(f"应答记录 {index} 名称缺失", fields=[f"answers[{index}].name"])
            if record.rtype <= 0:
                raise ChainError(f"应答记录 {index} 类型非法", fields=[f"answers[{index}].qtype"])
        # 名称/类型链：首条记录名必须等于问题名，后续 CNAME 链须逐跳衔接
        if response.answers:
            expected_name = qname
            for index, record in enumerate(response.answers):
                if record.name != expected_name:
                    raise ChainError(
                        f"应答记录名链在第 {index} 条断裂",
                        fields=[f"answers[{index}].name"],
                        context={"expected": expected_name, "actual": record.name},
                    )
                if record.rtype == 5 and index < len(response.answers) - 1:
                    expected_name = record.rdata_text


def _question_dict(question: Question) -> dict:
    return {"name": question.qname, "qtype": question.qtype, "qclass": question.qclass}
