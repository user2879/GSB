from __future__ import annotations

import json
import socketserver
import threading
from typing import Any, Sequence

from .backend import BackendResult, ForwardEngine, LocalBackendClient
from .errors import GatewayError, RequestError
from .ledger import LedgerStore
from .network import (
    BackendCluster,
    BatchDispatcher,
    RemoteBackendClient,
    _ThreadingJsonServer,
)
from .routing import decide_route
from .spec import GatewaySpec, LOOPBACK_HOST
from .util import canonical_json


class GatewayEngine:
    def __init__(
        self,
        spec: GatewaySpec,
        ledger: LedgerStore,
        forward_engine: ForwardEngine,
        *,
        backend: Any | None = None,
        dispatcher: BatchDispatcher | None = None,
    ):
        self.spec = spec
        self.ledger = ledger
        self.forward_engine = forward_engine
        self.backend = backend or LocalBackendClient(forward_engine)
        self.dispatcher = dispatcher

    def _validate_request(self, payload: Any) -> tuple[str, str, list[float]]:
        if not isinstance(payload, dict):
            raise RequestError("E_REQUEST", "$", "request must be a JSON object")
        client_key = payload.get("client_key")
        feature = payload.get("feature")
        sample = payload.get("sample")
        if not isinstance(client_key, str) or not client_key:
            raise RequestError(
                "E_REQUEST", "$.client_key", "must be a non-empty string"
            )
        if not isinstance(feature, str) or not feature:
            raise RequestError(
                "E_REQUEST", "$.feature", "must be a non-empty string"
            )
        if not isinstance(sample, list) or not sample:
            raise RequestError(
                "E_REQUEST", "$.sample", "must be a non-empty JSON array"
            )
        return client_key, feature, sample

    def _infer(
        self,
        *,
        request_seq: int,
        rule_id: str,
        batch_size: int,
        model_version: str,
        sample: Sequence[float],
    ) -> tuple[BackendResult, int]:
        if self.dispatcher is not None:
            return self.dispatcher.submit(
                rule_id=rule_id,
                model_version=model_version,
                batch_size=batch_size,
                sample=sample,
            )
        results = self.backend.infer(model_version, [sample])
        if len(results) != 1:
            raise RequestError(
                "E_BACKEND", "$.backend.results", "expected exactly one result"
            )
        batch_seq = ((request_seq - 1) // batch_size) + 1
        return results[0], batch_seq

    def handle(self, payload: Any) -> dict[str, Any]:
        client_key, feature, sample = self._validate_request(payload)
        route = self.spec.route_for_feature(feature)
        request_seq = self.ledger.reserve_request_seq()
        decision = decide_route(self.spec, route, request_seq, client_key)
        result, batch_seq = self._infer(
            request_seq=request_seq,
            rule_id=route.rule_id,
            batch_size=route.batch_size,
            model_version=decision.backend_version,
            sample=sample,
        )
        hit = self.ledger.append_hit(
            request_seq=request_seq,
            rule_id=decision.rule_id,
            stage_id=decision.stage_id,
            stage_weight_bp=decision.stage_weight_bp,
            hash_bucket=decision.hash_bucket,
            side=decision.side,
            backend_version=decision.backend_version,
            batch_seq=batch_seq,
            output_digest=result.output_digest,
            elapsed_us=result.elapsed_us,
            client_key=client_key,
            feature=feature,
            sample=[float(value) for value in sample],
        )
        return {
            "ok": True,
            "ledger_version": hit.ledger_version,
            "request_seq": hit.request_seq,
            "rule_id": hit.rule_id,
            "stage_id": hit.stage_id,
            "stage_weight_bp": hit.stage_weight_bp,
            "hash_bucket": hit.hash_bucket,
            "side": hit.side,
            "backend_version": hit.backend_version,
            "batch_seq": hit.batch_seq,
            "output_digest": hit.output_digest,
            "elapsed_us": hit.elapsed_us,
        }


class _GatewayHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                return
            try:
                payload = json.loads(line.decode("utf-8"))
                engine: GatewayEngine = self.server.engine  # type: ignore[attr-defined]
                response = engine.handle(payload)
            except GatewayError as exc:
                response = exc.to_dict()
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                response = {
                    "ok": False,
                    "code": "E_REQUEST",
                    "field_path": "$",
                    "message": f"invalid JSON line: {exc}",
                }
            except Exception as exc:  # pragma: no cover - protocol safety net
                response = {
                    "ok": False,
                    "code": "E_INTERNAL",
                    "field_path": "$",
                    "message": str(exc),
                }
            self.wfile.write((canonical_json(response) + "\n").encode("utf-8"))
            self.wfile.flush()


class GatewayTCPServer(_ThreadingJsonServer):
    def __init__(self, address: tuple[str, int], engine: GatewayEngine):
        self.engine = engine
        super().__init__(address, _GatewayHandler)


class GatewayStack:
    """Runs real loopback JSONL backend servers and the gateway listener."""

    def __init__(self, spec: GatewaySpec, ledger: LedgerStore):
        self.spec = spec
        self.ledger = ledger
        self.forward_engine = ForwardEngine(spec)
        self.backend_cluster = BackendCluster(spec, self.forward_engine)
        self.dispatcher: BatchDispatcher | None = None
        self.gateway_server: GatewayTCPServer | None = None
        self.gateway_thread: threading.Thread | None = None

    def start(self) -> None:
        self.backend_cluster.start()
        try:
            remote = RemoteBackendClient(self.spec)
            self.dispatcher = BatchDispatcher(remote, wait_window_ms=2)
            engine = GatewayEngine(
                self.spec,
                self.ledger,
                self.forward_engine,
                dispatcher=self.dispatcher,
            )
            self.gateway_server = GatewayTCPServer(
                (LOOPBACK_HOST, self.spec.port), engine
            )
            self.gateway_thread = threading.Thread(
                target=self.gateway_server.serve_forever,
                name="gateway-listener",
                daemon=True,
            )
            self.gateway_thread.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self.gateway_server is not None:
            self.gateway_server.shutdown()
            self.gateway_server.server_close()
            self.gateway_server = None
        if self.gateway_thread is not None:
            self.gateway_thread.join(timeout=3.0)
            self.gateway_thread = None
        if self.dispatcher is not None:
            self.dispatcher.close()
            self.dispatcher = None
        self.backend_cluster.stop()

    def __enter__(self) -> "GatewayStack":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

