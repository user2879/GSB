from __future__ import annotations

import argparse
import base64
import json
import socket
import socketserver
import sys
import time
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json
from .deterministic import BackendParameters, build_backend_parameters, build_input_tensor, forward_fixed
from .errors import LabError
from .specs import BackendSpec, SampleSpec, validate_backend


@dataclass
class WorkerRuntime:
    spec: BackendSpec
    version: int = 1

    def __post_init__(self) -> None:
        self.params: BackendParameters = build_backend_parameters(self.spec, self.version)

    def update(self, spec: BackendSpec, version: int) -> dict[str, Any]:
        if version <= self.version:
            raise LabError("E_FORMAT", "update version must increase")
        if spec.backend_id != self.spec.backend_id or spec.port != self.spec.port:
            raise LabError("E_FORMAT", "update cannot change backend_id or port")
        self.spec = spec
        self.version = version
        self.params = build_backend_parameters(spec, version)
        return self.params.summary()

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        batch_id = payload.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            raise LabError("E_FORMAT", "batch_id is required", field="batch_id")
        if payload.get("param_version") != self.version:
            raise LabError(
                "E_FORMAT",
                "request parameter version does not match backend runtime",
                field="param_version",
                details={"expected": self.version, "actual": payload.get("param_version")},
            )
        samples_raw = payload.get("samples")
        if not isinstance(samples_raw, list) or not samples_raw:
            raise LabError("E_FORMAT", "samples must be a non-empty array", field="samples")
        samples: list[SampleSpec] = []
        for index, item in enumerate(samples_raw):
            if not isinstance(item, dict):
                raise LabError("E_FORMAT", "sample must be an object", field=f"samples[{index}]")
            sample_id = item.get("sample_id")
            input_seed = item.get("input_seed")
            if not isinstance(sample_id, str) or not sample_id:
                raise LabError("E_FORMAT", "sample_id is required", field=f"samples[{index}].sample_id")
            if not isinstance(input_seed, int) or isinstance(input_seed, bool) or input_seed < 0:
                raise LabError("E_FORMAT", "input_seed is invalid", field=f"samples[{index}].input_seed")
            samples.append(
                SampleSpec(
                    sample_id=sample_id,
                    sample_seq=index + 1,
                    client_key="worker",
                    rule_id="worker",
                    request_path="/worker",
                    input_seed=input_seed,
                )
            )
        inputs = [build_input_tensor(item, self.spec.model_dim) for item in samples]
        start = time.perf_counter_ns()
        output_digest, rows, _ = forward_fixed(self.params, inputs)
        elapsed = time.perf_counter_ns() - start
        duration_ms = max(1, (elapsed + 999_999) // 1_000_000)
        return {
            "ok": True,
            "op": "infer",
            "backend_id": self.spec.backend_id,
            "batch_id": batch_id,
            "output_digest": output_digest,
            "sample_outputs": rows,
            "duration_ms": duration_ms,
            "backend_param_digest": self.params.param_digest,
            "backend_param_version": self.version,
            "input_digests": [item.digest for item in inputs],
        }


class BackendRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        runtime: WorkerRuntime = self.server.runtime  # type: ignore[attr-defined]
        for raw_line in self.rfile:
            try:
                payload = json.loads(raw_line.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise LabError("E_FORMAT", "request must be a JSON object")
                operation = payload.get("op")
                if operation == "ping":
                    response = {
                        "ok": True,
                        "op": "ping",
                        "backend_id": runtime.spec.backend_id,
                        "backend_param_digest": runtime.params.param_digest,
                        "backend_param_version": runtime.version,
                    }
                elif operation == "infer":
                    response = runtime.infer(payload)
                elif operation == "update":
                    raw_spec = payload.get("spec")
                    version = payload.get("version")
                    if not isinstance(raw_spec, dict) or not isinstance(version, int):
                        raise LabError("E_FORMAT", "update requires spec and integer version")
                    candidate = validate_backend(raw_spec, 0, set(), set())
                    response = {
                        "ok": True,
                        "op": "update",
                        "summary": runtime.update(candidate, version),
                    }
                else:
                    raise LabError("E_FORMAT", "unsupported operation", field="op")
            except LabError as exc:
                response = {"ok": False, **exc.to_dict()}
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": "E_RUNTIME",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            self.wfile.write((canonical_json(response) + "\n").encode("utf-8"))
            self.wfile.flush()


class BackendTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_server(spec: BackendSpec, version: int = 1) -> None:
    runtime = WorkerRuntime(spec, version)
    with BackendTCPServer((spec.host, spec.port), BackendRequestHandler) as server:
        server.runtime = runtime  # type: ignore[attr-defined]
        print(f"READY {spec.backend_id} {spec.host}:{spec.port} v{version}", flush=True)
        server.serve_forever(poll_interval=0.1)


def _decode_backend(value: str) -> BackendSpec:
    try:
        raw = json.loads(base64.b64decode(value).decode("utf-8"))
    except Exception as exc:
        raise LabError("E_SPEC", f"cannot decode backend argument: {exc}") from exc
    return validate_backend(raw, 0, set(), set())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local loopback deterministic inference backend")
    parser.add_argument("--backend-b64", required=True, help="base64-encoded backend declaration JSON")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        spec = _decode_backend(args.backend_b64)
        if args.version <= 0:
            raise LabError("E_SPEC", "version must be positive", field="version")
        run_server(spec, args.version)
        return 0
    except LabError as exc:
        print(canonical_json(exc.to_dict()), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
