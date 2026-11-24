"""Real loopback UDP send/receive transport used by runs and replay."""

from __future__ import annotations

import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from .declarations import ProbeDefinition, SessionDefinition
from .protocol import (
    canonical_json,
    decode_json_datagram,
    encode_request,
    make_request,
)

ACK = "ACK"
NO_RESPONSE = "NO_RESPONSE"
SEQ_MISMATCH = "SEQ_MISMATCH"
PROTOCOL_ERROR = "PROTOCOL_ERROR"
SEND_ERROR = "SEND_ERROR"


@dataclass(frozen=True)
class ProbeBatch:
    records: tuple[dict[str, Any], ...]


def build_probe_batch(
    session: SessionDefinition,
    probes: Mapping[str, ProbeDefinition],
    *,
    cycle: int,
    attempt: int,
    lease_version: int,
    planned_mono_ns: int,
) -> ProbeBatch:
    records = []
    for probe_id in session.probe_ids:
        tx_id = uuid.uuid4().hex
        request = make_request(
            session_id=session.session_id,
            probe_id=probe_id,
            cycle=cycle,
            lease_version=lease_version,
            attempt=attempt,
            tx_id=tx_id,
        )
        records.append(
            {
                "probe_id": probe_id,
                "tx_id": tx_id,
                "request_payload": canonical_json(request),
                "request": request,
                "request_lease_version": lease_version,
                "planned_mono_ns": planned_mono_ns,
            }
        )
    return ProbeBatch(tuple(records))


def transmit_and_collect(
    session: SessionDefinition,
    batch: ProbeBatch,
    *,
    cycle: int,
    lease_version: int,
    attempt: int,
) -> list[dict[str, Any]]:
    """Send one datagram per probe and apply real receive deadlines."""

    exchange = UdpExchange(
        session,
        batch,
        cycle=cycle,
        lease_version=lease_version,
        attempt=attempt,
    )
    try:
        exchange.send_all()
        exchange.collect()
        return exchange.records()
    finally:
        exchange.close()


class UdpExchange:
    """A real UDP exchange split into send and receive phases."""

    def __init__(
        self,
        session: SessionDefinition,
        batch: ProbeBatch,
        *,
        cycle: int,
        lease_version: int,
        attempt: int,
    ) -> None:
        self.session = session
        self.batch = batch
        self.cycle = cycle
        self.lease_version = lease_version
        self.attempt = attempt
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pending: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {
            record["tx_id"]: {
                "tx_id": record["tx_id"],
                "probe_id": record["probe_id"],
                "request_lease_version": record["request_lease_version"],
                "send_mono_ns": None,
                "receive_mono_ns": None,
                "outcome": NO_RESPONSE,
                "reason": "timeout:no_response",
                "response_seq": None,
            }
            for record in batch.records
        }
        self._sent = False
        self._collected = False

    def send_all(self) -> None:
        if self._sent:
            return
        self._sent = True
        address = ("127.0.0.1", self.session.port)
        for record in self.batch.records:
            sent_ns = time.monotonic_ns()
            try:
                self.sock.sendto(encode_request(record["request"]), address)
            except OSError as exc:
                self._records[record["tx_id"]]["send_mono_ns"] = sent_ns
                self._records[record["tx_id"]]["outcome"] = SEND_ERROR
                self._records[record["tx_id"]]["reason"] = f"send_error:{exc.errno}"
                continue
            self._records[record["tx_id"]]["send_mono_ns"] = sent_ns
            self.pending[record["tx_id"]] = record

    def collect(self) -> None:
        if self._collected:
            return
        if not self._sent:
            self.send_all()
        self._collected = True
        if not self.pending:
            return

        last_send_ns = max(
            int(self._records[tx_id]["send_mono_ns"])
            for tx_id in self.pending
            if self._records[tx_id]["send_mono_ns"] is not None
        )
        deadline_ns = last_send_ns + self.session.datagram_timeout_ms * 1_000_000
        while self.pending:
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                break
            self.sock.settimeout(remaining_ns / 1_000_000_000)
            try:
                payload, _source = self.sock.recvfrom(65_535)
            except socket.timeout:
                break
            except OSError as exc:
                for tx_id in list(self.pending):
                    self._records[tx_id]["outcome"] = NO_RESPONSE
                    self._records[tx_id]["reason"] = f"receive_error:{exc.errno}"
                    self._records[tx_id]["receive_mono_ns"] = time.monotonic_ns()
                    self.pending.pop(tx_id, None)
                break
            try:
                response = decode_json_datagram(payload)
            except ValueError:
                continue
            tx_id = response.get("tx_id")
            if not isinstance(tx_id, str) or tx_id not in self.pending:
                continue
            self.pending.pop(tx_id, None)
            received_ns = time.monotonic_ns()
            item = self._records[tx_id]
            item["receive_mono_ns"] = received_ns
            if response.get("kind") != "probe.response":
                item["outcome"] = PROTOCOL_ERROR
                item["reason"] = "protocol_error:kind"
                continue
            seq = response.get("seq")
            if not isinstance(seq, int) or isinstance(seq, bool):
                item["outcome"] = PROTOCOL_ERROR
                item["reason"] = "protocol_error:seq_not_integer"
                continue
            item["response_seq"] = seq
            if seq != self.cycle:
                item["outcome"] = SEQ_MISMATCH
                item["reason"] = (
                    f"sequence_mismatch:expected={self.cycle}:received={seq}"
                )
                continue
            identity_mismatches = []
            for key, expected in (
                ("session_id", self.session.session_id),
                ("probe_id", item["probe_id"]),
                ("cycle", self.cycle),
                ("lease_version", self.lease_version),
                ("protocol_version", 1),
            ):
                if response.get(key) != expected:
                    identity_mismatches.append(
                        f"{key}:expected={expected}:received={response.get(key)}"
                    )
            if identity_mismatches:
                item["outcome"] = PROTOCOL_ERROR
                item["reason"] = "protocol_mismatch:" + ",".join(identity_mismatches)
                continue
            item["outcome"] = ACK
            item["reason"] = "ack"
        now_ns = time.monotonic_ns()
        for tx_id in list(self.pending):
            self._records[tx_id]["receive_mono_ns"] = now_ns
            self._records[tx_id]["outcome"] = NO_RESPONSE
            self._records[tx_id]["reason"] = "timeout:no_response"
            self.pending.pop(tx_id, None)

    def records(self) -> list[dict[str, Any]]:
        return [self._records[record["tx_id"]] for record in self.batch.records]

    def close(self) -> None:
        self.sock.close()


