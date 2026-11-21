"""Real loopback TLS 1.2 handshake with exact wire-chain capture."""

from __future__ import annotations

import json
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import digest_obj, digest_without_field, write_canonical_json
from .certfamily import Family, family_digest as compute_family_digest, validate_port
from .errors import E_CHAIN, E_HANDSHAKE, E_SNAPSHOT, WorkbenchError
from .verify import snapshot_chain_records
from .x509 import CertRecord, parse_certificate


@dataclass
class CaptureState:
    certificate_der: list[bytes] = field(default_factory=list)
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


class TLS12CertificateParser:
    """Incrementally parse plaintext TLS 1.2 records on the server-to-client leg."""

    def __init__(self, state: CaptureState):
        self.state = state
        self.record_buffer = bytearray()
        self.handshake_buffer = bytearray()

    def feed(self, data: bytes) -> None:
        self.record_buffer.extend(data)
        try:
            while len(self.record_buffer) >= 5:
                content_type = self.record_buffer[0]
                length = int.from_bytes(self.record_buffer[3:5], "big")
                if length > 1_048_576:
                    raise ValueError("TLS record too large")
                total = 5 + length
                if len(self.record_buffer) < total:
                    break
                payload = bytes(self.record_buffer[5:total])
                del self.record_buffer[:total]
                if content_type == 22:
                    self.handshake_buffer.extend(payload)
                    self._parse_handshake_messages()
        except Exception as exc:  # handler records malformed wire data
            self.state.error = f"TLS record parse failed: {exc}"
            self.state.done.set()

    def _parse_handshake_messages(self) -> None:
        while len(self.handshake_buffer) >= 4:
            message_type = self.handshake_buffer[0]
            length = int.from_bytes(self.handshake_buffer[1:4], "big")
            if length > 16_777_216:
                raise ValueError("TLS handshake message too large")
            total = 4 + length
            if len(self.handshake_buffer) < total:
                return
            body = bytes(self.handshake_buffer[4:total])
            del self.handshake_buffer[:total]
            if message_type == 11:
                self._parse_certificate_message(body)

    def _parse_certificate_message(self, body: bytes) -> None:
        if len(body) < 3:
            raise ValueError("truncated Certificate message")
        total = int.from_bytes(body[:3], "big")
        if 3 + total != len(body):
            raise ValueError("Certificate message length mismatch")
        position = 3
        certificates: list[bytes] = []
        while position < len(body):
            if position + 3 > len(body):
                raise ValueError("truncated certificate length")
            length = int.from_bytes(body[position : position + 3], "big")
            position += 3
            if length == 0 or position + length > len(body):
                raise ValueError("invalid certificate length")
            certificates.append(body[position : position + length])
            position += length
        if not certificates:
            raise ValueError("server sent an empty certificate list")
        self.state.certificate_der = certificates
        self.state.done.set()


def _pump(
    src: socket.socket,
    dst: socket.socket,
    parser: TLS12CertificateParser | None = None,
) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            if parser is not None:
                parser.feed(data)
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class CaptureProxy(threading.Thread):
    def __init__(self, listener: socket.socket, backend_port: int, parser: TLS12CertificateParser):
        super().__init__(daemon=True, name="tlstrust-capture-proxy")
        self.listener = listener
        self.backend_port = backend_port
        self.parser = parser
        self.error: str | None = None
        self.port = listener.getsockname()[1]

    def run(self) -> None:
        try:
            self.listener.settimeout(15)
            client, _ = self.listener.accept()
        except OSError as exc:
            self.error = f"capture proxy accept failed: {exc}"
            return
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection(("127.0.0.1", self.backend_port), timeout=10)
            upstream.settimeout(None)
            client.settimeout(None)
            up = threading.Thread(target=_pump, args=(client, upstream, None), daemon=True)
            down = threading.Thread(target=_pump, args=(upstream, client, self.parser), daemon=True)
            up.start()
            down.start()
            up.join(15)
            down.join(15)
        except OSError as exc:
            self.error = f"capture proxy bridge failed: {exc}"
        finally:
            for sock in (client, upstream):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            try:
                self.listener.close()
            except OSError:
                pass


class TLSServer(threading.Thread):
    def __init__(self, listener: socket.socket, context: ssl.SSLContext):
        super().__init__(daemon=True, name="tlstrust-tls-server")
        self.listener = listener
        self.context = context
        self.error: str | None = None
        self.port = listener.getsockname()[1]

    def run(self) -> None:
        try:
            self.listener.settimeout(15)
            raw, _ = self.listener.accept()
        except OSError as exc:
            self.error = f"TLS server accept failed: {exc}"
            return
        try:
            raw.settimeout(15)
            with self.context.wrap_socket(raw, server_side=True) as tls:
                tls.do_handshake()
                try:
                    while tls.recv(4096):
                        pass
                except (ssl.SSLError, OSError):
                    pass
        except Exception as exc:
            self.error = f"TLS server handshake failed: {exc}"
        finally:
            try:
                raw.close()
            except OSError:
                pass
            try:
                self.listener.close()
            except OSError:
                pass


def _make_contexts(chain_file: Path, key_file: Path) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.minimum_version = ssl.TLSVersion.TLSv1_2
    server.maximum_version = ssl.TLSVersion.TLSv1_2
    server.set_ciphers("DEFAULT:@SECLEVEL=0")
    server.load_cert_chain(str(chain_file), str(key_file))
    client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client.check_hostname = False
    client.verify_mode = ssl.CERT_NONE
    client.minimum_version = ssl.TLSVersion.TLSv1_2
    client.maximum_version = ssl.TLSVersion.TLSv1_2
    client.set_ciphers("DEFAULT:@SECLEVEL=0")
    return server, client


def _bind_loopback(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(1)
    return sock


def build_input_snapshot(
    records: list[CertRecord],
    *,
    leaf_id: str,
    negotiated: dict[str, Any],
    captured_monotonic_ms: int,
    endpoints: dict[str, Any],
    family_digest: str,
) -> dict[str, Any]:
    chain = []
    for position, record in enumerate(records):
        entry = record.to_snapshot_dict()
        entry["position"] = position
        chain.append(entry)
    fingerprints = [record.fingerprint for record in records]
    payload: dict[str, Any] = {
        "format_version": 1,
        "kind": "tlstrust.input_snapshot",
        "validator_version": 1,
        "capture_method": "stdlib_ssl_tls12_wire_proxy",
        "leaf_id": leaf_id,
        "family_digest": family_digest,
        "chain": chain,
        "chain_fingerprints": fingerprints,
        "chain_serial_numbers": [record.serial_number for record in records],
        "chain_digest": digest_obj(fingerprints),
        "negotiated": negotiated,
        "captured_monotonic_ms": int(captured_monotonic_ms),
        "endpoints": endpoints,
    }
    payload["snapshot_digest"] = digest_without_field(payload, "snapshot_digest")
    return payload


def capture_real_handshake(
    family: Family,
    *,
    leaf_id: str | None = None,
    port: int = 0,
    inject_chain_fingerprint: bool = False,
) -> dict[str, Any]:
    validate_port(port)
    leaf = family.leaf(leaf_id)
    expected_fingerprints = list(leaf["chain_fingerprints"])
    server_ctx, client_ctx = _make_contexts(family.chain_file(leaf["id"]), family.key_file(leaf["id"]))
    backend = _bind_loopback(port)
    proxy_listener = _bind_loopback(0)
    server = TLSServer(backend, server_ctx)
    state = CaptureState()
    parser = TLS12CertificateParser(state)
    proxy = CaptureProxy(proxy_listener, server.port, parser)
    server.start()
    proxy.start()

    tls: ssl.SSLSocket | None = None
    try:
        raw = socket.create_connection(("127.0.0.1", proxy.port), timeout=10)
        raw.settimeout(15)
        tls = client_ctx.wrap_socket(raw, server_hostname="localhost")
        tls.do_handshake()
        negotiated = {
            "tls_version": tls.version(),
            "cipher_name": tls.cipher()[0] if tls.cipher() else None,
            "cipher_protocol": tls.cipher()[1] if tls.cipher() else None,
            "cipher_bits": tls.cipher()[2] if tls.cipher() else None,
            "compression": tls.compression(),
            "server_hostname": "localhost",
            "selected_alpn_protocol": tls.selected_alpn_protocol(),
        }
        captured_monotonic_ms = time.monotonic_ns() // 1_000_000
        if not state.done.wait(5):
            raise WorkbenchError(
                E_HANDSHAKE,
                "TLS handshake completed but the server certificate chain was not captured",
                layer=0,
                field_path="handshake.certificate",
                details={"proxy_error": proxy.error, "server_error": server.error},
            )
        if state.error:
            raise WorkbenchError(
                E_HANDSHAKE,
                state.error,
                layer=0,
                field_path="handshake.certificate",
            )
    except WorkbenchError:
        raise
    except Exception as exc:
        raise WorkbenchError(
            E_HANDSHAKE,
            f"real TLS handshake failed: {exc}",
            layer=0,
            field_path="handshake",
            details={"proxy_error": proxy.error, "server_error": server.error},
        ) from exc
    finally:
        if tls is not None:
            try:
                tls.close()
            except OSError:
                pass
        try:
            proxy_listener.close()
        except OSError:
            pass
        proxy.join(5)
        server.join(5)

    if not state.certificate_der:
        raise WorkbenchError(
            E_HANDSHAKE,
            "server did not send a certificate chain",
            layer=0,
            field_path="handshake.certificate",
        )
    records: list[CertRecord] = []
    for layer, der in enumerate(state.certificate_der):
        try:
            records.append(parse_certificate(der))
        except Exception as exc:
            raise WorkbenchError(
                E_CHAIN,
                "server sent a certificate that cannot be parsed",
                layer=layer,
                field_path=f"handshake.chain[{layer}]",
            ) from exc
    actual_fingerprints = [record.fingerprint for record in records]
    if len(actual_fingerprints) != len(expected_fingerprints):
        raise WorkbenchError(
            E_HANDSHAKE,
            "server did not send the complete certificate chain",
            layer=min(len(actual_fingerprints), len(expected_fingerprints)),
            field_path="handshake.chain",
            details={
                "expected_length": len(expected_fingerprints),
                "actual_length": len(actual_fingerprints),
            },
        )
    if actual_fingerprints[-1] != expected_fingerprints[-1]:
        raise WorkbenchError(
            E_HANDSHAKE,
            "captured chain does not terminate at the configured trust-anchor certificate",
            layer=len(actual_fingerprints) - 1,
            field_path="handshake.chain",
        )
    if actual_fingerprints != expected_fingerprints:
        mismatch = next(
            index
            for index, (expected, actual) in enumerate(zip(expected_fingerprints, actual_fingerprints))
            if expected != actual
        )
        raise WorkbenchError(
            E_CHAIN,
            "wire-captured certificate fingerprint sequence differs from the served chain",
            layer=mismatch,
            field_path=f"handshake.chain[{mismatch}].fingerprint",
            details={"expected": expected_fingerprints, "actual": actual_fingerprints},
        )

    snapshot = build_input_snapshot(
        records,
        leaf_id=leaf["id"],
        negotiated=negotiated,
        captured_monotonic_ms=captured_monotonic_ms,
        endpoints={
            "bind_host": "127.0.0.1",
            "tls_server_port": server.port,
            "capture_proxy_port": proxy.port,
        },
        family_digest=compute_family_digest(family),
    )
    if inject_chain_fingerprint:
        index = 0 if len(snapshot["chain"]) == 1 else 1
        fake = "sha256:" + "0" * 64
        snapshot["chain"][index]["fingerprint"] = fake
        snapshot["chain_fingerprints"][index] = fake
        snapshot["chain_digest"] = digest_obj(snapshot["chain_fingerprints"])
        snapshot["snapshot_digest"] = digest_without_field(snapshot, "snapshot_digest")
        try:
            snapshot_chain_records(snapshot)
        except WorkbenchError:
            raise
        raise WorkbenchError(E_CHAIN, "injected chain fingerprint mismatch", layer=index, field_path=f"snapshot.chain[{index}].fingerprint")
    return snapshot


def write_input_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    write_canonical_json(path, snapshot)


def load_input_snapshot(path: str | Path) -> dict[str, Any]:
    try:
        snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError(E_SNAPSHOT, f"cannot load input snapshot: {exc}", field_path="snapshot") from exc
    snapshot_chain_records(snapshot)
    return snapshot
