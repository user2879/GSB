"""Exercise every coded failure path and order-independence guarantee.

Covered codes: E_SOURCE, E_FORMAT, E_PORT, E_TIMEOUT, E_ACK, E_REPLAY,
plus byte-identical output under shuffled source reporting order.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slowquery_diag.codec import FrameBuilder  # noqa: E402
from slowquery_diag.config import load_declaration  # noqa: E402
from slowquery_diag.engine import DiagnosticEngine  # noqa: E402
from slowquery_diag.errors import (  # noqa: E402
    AckError,
    FormatError,
    PortError,
    ReplayMismatchError,
    ReplayTimeoutError,
    SourceDeclarationError,
)
from slowquery_diag.models import (  # noqa: E402
    SOURCE_DB,
    canonical_json,
)
from slowquery_diag.replay_server import (  # noqa: E402
    DeterministicReplayService,
    ReplayClient,
)


def show(title: str, exc: Exception) -> None:
    print(f"[OK] {title:28s} -> {exc}")


def main() -> int:
    example_dir = os.path.join(ROOT, "examples")
    with tempfile.TemporaryDirectory() as tmp:
        # E_SOURCE: duplicate source id and negative integer field.
        bad_decl = os.path.join(tmp, "bad_decl.json")
        with open(os.path.join(example_dir, "declaration.json"),
                  encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["sources"][1]["source_id"] = payload["sources"][0]["source_id"]
        # Keep every referenced frame file valid so uniqueness is the
        # first violation that validation can reach.
        payload["sources"][1]["frame_file"] = payload["sources"][0]["frame_file"]
        with open(bad_decl, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        try:
            load_declaration(bad_decl)
        except SourceDeclarationError as exc:
            show("E_SOURCE duplicate id", exc)

        bad_decl2 = os.path.join(tmp, "bad_decl2.json")
        with open(os.path.join(example_dir, "declaration.json"),
                  encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["sources"][0]["sample_rows"] = -3
        with open(bad_decl2, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        try:
            load_declaration(bad_decl2)
        except SourceDeclarationError as exc:
            show("E_SOURCE negative int", exc)

        # E_FORMAT: unknown record version with row localization.
        bad_frame = os.path.join(tmp, "bad.frame")
        builder = FrameBuilder(SOURCE_DB, 2)
        builder.add(2, (1, 100, 10, 1))
        raw = bytearray(builder.build())
        # Header is 11 bytes, then u16 frame_len; the record version byte
        # is the first byte inside that framed payload (index 13).
        raw[13] = 9
        with open(bad_frame, "wb") as handle:
            handle.write(bytes(raw))
        from slowquery_diag.models import SourceDeclaration
        declaration = SourceDeclaration(
            source_id="s", source_type=SOURCE_DB, format_version=2,
            sample_rows=1, latency_caliber="p95", sample_period_ms=1000,
            latency_algebra={}, frame_file=bad_frame,
        )
        from slowquery_diag.codec import load_frame_file
        try:
            load_frame_file(declaration)
        except FormatError as exc:
            show("E_FORMAT bad row version", exc)

        # E_PORT: connect to a port with no listener.  Some platforms
        # answer refused, others time out; both socket-level failures are
        # normalized into the coded PortError/ReplayTimeoutError family,
        # and the E_PORT branch is exercised deterministically by
        # attempting to bind an already-bound port below.
        try:
            ReplayClient(port=1, timeout_ms=500).hello()
        except (PortError, ReplayTimeoutError) as exc:
            show("no-listener socket failure", exc)

        occupied = DeterministicReplayService({})
        occupied_port = occupied.start(0)
        try:
            clash = DeterministicReplayService({})
            clash.start(occupied_port)
        except PortError as exc:
            show("E_PORT port occupied", exc)
        finally:
            occupied.stop()

        # E_TIMEOUT: stall entry exceeds the real socket timeout.
        catalog = {
            "stall@src": {
                "candidate_id": "stall", "source_id": "src",
                "est_latency_us": 1, "est_rows_scanned": 1,
                "stall_ms": 1500,
            }
        }
        service = DeterministicReplayService(catalog)
        port = service.start(0)
        try:
            ReplayClient(port=port, timeout_ms=300).replay(
                "stall", "src", "SELECT 1"
            )
        except ReplayTimeoutError as exc:
            show("E_TIMEOUT socket timeout", exc)
        finally:
            service.stop()

        # E_ACK: service returns a receipt missing required fields.
        from slowquery_diag import replay_server as rs
        original = DeterministicReplayService._handle_replay

        def broken_handle(self, conn, request):
            rs._send_frame(conn, {"type": "ack", "candidate_id": "x"})

        DeterministicReplayService._handle_replay = broken_handle
        service = DeterministicReplayService({"x@src": {
            "candidate_id": "x", "source_id": "src",
            "est_latency_us": 1, "est_rows_scanned": 1,
        }})
        port = service.start(0)
        try:
            ReplayClient(port=port, timeout_ms=1000).replay(
                "x", "src", "SELECT 1"
            )
        except AckError as exc:
            show("E_ACK missing field", exc)
        finally:
            DeterministicReplayService._handle_replay = original
            service.stop()

        # E_REPLAY: ledger/fresh receipt field divergence.
        ledger_path = os.path.join(tmp, "err_ledger.db")
        engine = DiagnosticEngine(ledger_path, scale_exponent=1)
        result = engine.run(os.path.join(example_dir, "declaration.json"))
        try:
            engine.ledger.verify_replay_against_ledger(result.scheme_version, [{
                "candidate_id": "ghost", "source_id": "src",
                "ack": {"candidate_id": "ghost", "source_id": "src",
                        "est_latency_us": 1, "est_rows_scanned": 1},
            }])
        except ReplayMismatchError as exc:
            show("E_REPLAY divergence", exc)
        engine.close()

        # Order independence: shuffled source list yields identical bytes.
        def run_with_order(order):
            decl_path = os.path.join(tmp, f"decl_{order}.json")
            with open(os.path.join(example_dir, "declaration.json"),
                      encoding="utf-8") as src:
                base = json.load(src)
            if order == "reverse":
                base["sources"] = list(reversed(base["sources"]))
                base["candidates"] = list(reversed(base["candidates"]))
            # Frame paths are relative to the declaration; rewrite them to
            # absolute example paths so copies in tmp still resolve.
            for source in base["sources"]:
                source["frame_file"] = os.path.join(
                    example_dir, os.path.basename(source["frame_file"])
                )
            with open(decl_path, "w", encoding="utf-8") as handle:
                json.dump(base, handle)
            eng = DiagnosticEngine(
                os.path.join(tmp, f"ledger_{order}.db"), scale_exponent=1
            )
            res = eng.run(decl_path)
            from slowquery_diag.cli import _serialize
            rendered = canonical_json(_serialize(res))
            eng.close()
            return rendered

        forward = run_with_order("forward")
        reverse = run_with_order("reverse")
        assert forward == reverse, "output must be byte-identical under shuffle"
        print("[OK] order independence        -> shuffled bytes identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
