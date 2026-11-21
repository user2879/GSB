from __future__ import annotations

import dataclasses
import json
import random
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from lossdiag.alignment import align_observations
from lossdiag.diagnosis import diagnose_streams
from lossdiag.echo import (
    EchoStateStore,
    build_echo_plan,
    build_echo_response,
    recover_echo_state,
    run_echo_replay,
    serve_peer,
    verify_echo_response,
)
from lossdiag.errors import DiagnosticError
from lossdiag.events import load_event_bundle, normalize_event_document
from lossdiag.model import FiveTuple
from lossdiag.pipeline import build_session
from lossdiag.rates import aggregate_rates
from lossdiag.replay import verify_ledger
from lossdiag.store import WorkbenchStore, restore_snapshot, restored_session_view
from lossdiag.util import canonical_json, content_digest
from lossdiag.wire import HEADER_SIZE, decode_wire, encode_frame, read_wire_bundle


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
WIRE = SAMPLES / "observations.bin"
MANIFEST = SAMPLES / "wire_manifest.json"
EVENTS_V1 = SAMPLES / "events_v1.json"
EVENTS_V2 = SAMPLES / "events_v2.json"
EVENTS_V3 = SAMPLES / "events_v3.json"


def _session(version: int = 3, window_ms: int = 100):
    return build_session(
        wire_path=str(WIRE),
        wire_manifest_path=str(MANIFEST),
        event_paths=[str(SAMPLES / f"events_v{version}.json")],
        window_ms=window_ms,
    )


class WireCodecTests(unittest.TestCase):
    def test_roundtrip_and_frame_summary(self):
        flow = FiveTuple("192.0.2.1", "198.51.100.2", 12345, 443, "tcp")
        encoded = encode_frame(
            five_tuple=flow,
            seq=7,
            send_ms=9876,
            flags=0x01,
            arrival_ordinal=3,
            payload=b"abcd",
        )
        frames = decode_wire(encoded, source_id="test")
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].seq, 7)
        self.assertEqual(frames[0].send_ms, 9876)
        self.assertEqual(frames[0].payload_len, 4)
        self.assertEqual(frames[0].arrival_ordinal, 3)
        self.assertEqual(frames[0].flow_id, flow.flow_id)
        self.assertEqual(frames[0].byte_offset, 0)
        self.assertEqual(frames[0].frame_index, 1)

    def test_truncated_frame_reports_offset_and_index(self):
        with self.assertRaises(DiagnosticError) as context:
            decode_wire(WIRE.read_bytes()[:-1], source_id="test")
        error = context.exception
        self.assertEqual(error.code, "E_WIRE")
        self.assertIn("byte_offset", error.details)
        self.assertIn("frame_index", error.details)
        self.assertEqual(error.details["frame_index"], 27)

    def test_illegal_flags_are_e_wire(self):
        flow = FiveTuple("192.0.2.1", "198.51.100.2", 12345, 443, "tcp")
        data = bytearray(
            encode_frame(
                five_tuple=flow,
                seq=1,
                send_ms=1,
                flags=0x01,
                arrival_ordinal=1,
            )
        )
        data[35] = 0x00
        with self.assertRaises(DiagnosticError) as context:
            decode_wire(bytes(data), source_id="test")
        self.assertEqual(context.exception.code, "E_WIRE")
        self.assertEqual(context.exception.details["byte_offset"], 0)
        self.assertEqual(context.exception.details["frame_index"], 1)

    def test_flow_identifier_and_length_consistency(self):
        flow = FiveTuple("192.0.2.1", "198.51.100.2", 12345, 443, "tcp")
        valid = bytearray(
            encode_frame(
                five_tuple=flow,
                seq=1,
                send_ms=1,
                flags=0x01,
                arrival_ordinal=1,
                payload=b"x",
            )
        )
        bad_flow = bytearray(valid)
        bad_flow[6] ^= 0xFF
        with self.assertRaises(DiagnosticError) as context:
            decode_wire(bytes(bad_flow), source_id="test")
        self.assertEqual(context.exception.code, "E_WIRE")
        self.assertIn("flow identifier", context.exception.message)

        bad_length = bytearray(valid)
        bad_length[4:6] = (HEADER_SIZE + 1 + 1).to_bytes(2, "little")
        with self.assertRaises(DiagnosticError) as context:
            decode_wire(bytes(bad_length) + b"z", source_id="test")
        self.assertEqual(context.exception.code, "E_WIRE")
        self.assertIn("payload length", context.exception.message)

    def test_injected_wire_corruption(self):
        with self.assertRaises(DiagnosticError) as context:
            read_wire_bundle(WIRE, MANIFEST, corrupt=True)
        self.assertEqual(context.exception.code, "E_WIRE")


class EventVersionTests(unittest.TestCase):
    def test_v1_v2_v3_normalization_and_padding(self):
        v1 = load_event_bundle(EVENTS_V1)
        v2 = load_event_bundle(EVENTS_V2)
        v3 = load_event_bundle(EVENTS_V3)
        self.assertEqual(v1.version, "v1")
        self.assertEqual(v2.version, "v2")
        self.assertEqual(v3.version, "v3")
        self.assertEqual(
            v1.records[0].padded_fields,
            ("link_id", "reorder_depth", "retransmit", "time_window_ms"),
        )
        self.assertEqual(v1.records[0].time_window_ms, None)
        self.assertEqual(v2.records[0].padded_fields, ("reorder_depth", "retransmit"))
        self.assertIsNotNone(v2.records[0].time_window_ms)
        self.assertEqual(v3.records[0].padded_fields, ())
        self.assertEqual(v3.records[0].source_version, "v3")
        self.assertTrue(any(record.retransmit for record in v3.records))
        self.assertTrue(any(record.reorder_depth > 0 for record in v3.records))

    def _document(self, version: int = 3):
        return json.loads((SAMPLES / f"events_v{version}.json").read_text(encoding="utf-8"))

    def test_negative_count_is_e_format_with_record_and_field(self):
        document = self._document()
        document["records"][0]["loss_count"] = -1
        with self.assertRaises(DiagnosticError) as context:
            normalize_event_document(document, source_digest="x")
        self.assertEqual(context.exception.code, "E_FORMAT")
        self.assertEqual(context.exception.details["record_number"], 1)
        self.assertEqual(context.exception.details["field"], "loss_count")

    def test_unknown_version_is_e_format(self):
        document = self._document()
        document["version"] = "v9"
        with self.assertRaises(DiagnosticError) as context:
            normalize_event_document(document, source_digest="x")
        self.assertEqual(context.exception.code, "E_FORMAT")
        self.assertEqual(context.exception.details["field"], "version")

    def test_missing_measured_window_is_not_padded(self):
        document = self._document(version=2)
        document["records"][0].pop("time_window_ms")
        with self.assertRaises(DiagnosticError) as context:
            normalize_event_document(document, source_digest="x")
        self.assertEqual(context.exception.code, "E_FORMAT")
        self.assertEqual(context.exception.details["record_number"], 1)
        self.assertEqual(context.exception.details["field"], "time_window_ms")

    def test_v1_requires_explicit_defaults_and_null_window_default(self):
        document = self._document(version=1)
        document.pop("normalization_defaults")
        with self.assertRaises(DiagnosticError) as context:
            normalize_event_document(document, source_digest="x")
        self.assertEqual(context.exception.code, "E_FORMAT")
        self.assertEqual(context.exception.details["field"], "normalization_defaults")

        document = self._document(version=1)
        document["normalization_defaults"]["time_window_ms"] = {"start_ms": 0, "end_ms": 1}
        with self.assertRaises(DiagnosticError) as context:
            normalize_event_document(document, source_digest="x")
        self.assertEqual(context.exception.code, "E_FORMAT")
        self.assertEqual(
            context.exception.details["field"],
            "normalization_defaults.time_window_ms",
        )

    def test_alignment_scale_is_rejected(self):
        document = self._document()
        document["alignment"]["scale"] = 2
        with self.assertRaises(DiagnosticError) as context:
            normalize_event_document(document, source_digest="x")
        self.assertEqual(context.exception.code, "E_FORMAT")
        self.assertEqual(context.exception.details["field"], "alignment.scale")


class AlignmentTests(unittest.TestCase):
    def test_translation_only_alignment(self):
        session = _session()
        self.assertEqual(session.alignment.unaligned_flow_ids, [])
        self.assertTrue(all(record.aligned for record in session.alignment.records))
        self.assertTrue(all(record.offset_to_frame_ms == 1000 for record in session.alignment.records))
        flow_a = next(
            flow for flow in session.diagnoses if "10.10.0.1" in flow.flow_id
        )
        gap = next(gap for gap in flow_a.gaps if gap.start_seq == 3)
        self.assertEqual(gap.time_window_ms, {"start_ms": 1015, "end_ms": 1035})

    def test_missing_basis_goes_to_unaligned_set_and_is_not_guessed(self):
        frame_bundle = read_wire_bundle(WIRE, MANIFEST)
        document = json.loads(EVENTS_V3.read_text(encoding="utf-8"))
        document.pop("alignment")
        bundle = normalize_event_document(document, source_digest="x")
        alignment = align_observations(frame_bundle, [bundle])
        self.assertEqual(len(alignment.unaligned_flow_ids), 3)
        self.assertTrue(
            all("event_offset_to_frame_ms" in record.missing_basis for record in alignment.records)
        )
        self.assertTrue(all(record.offset_to_frame_ms is None for record in alignment.records))
        self.assertEqual(bundle.records[0].time_window_ms.start_ms, 15)

    def test_v1_has_sequence_only_alignment(self):
        session = _session(version=1)
        self.assertEqual(len(session.alignment.unaligned_flow_ids), 3)
        self.assertTrue(all(record.sequence_only for record in session.alignment.records))
        self.assertTrue(
            all("event_time_window" in record.missing_basis for record in session.alignment.records)
        )


class DeterminismAndRateTests(unittest.TestCase):
    def test_gap_classification(self):
        session = _session()
        flow_a = next(flow for flow in session.diagnoses if "10.10.0.1" in flow.flow_id)
        classes = {
            (gap.start_seq, gap.end_seq): gap.classification for gap in flow_a.gaps
        }
        self.assertEqual(classes[(3, 3)], "true_loss")
        self.assertEqual(classes[(5, 5)], "retransmission")
        self.assertEqual(classes[(6, 7)], "reordering")
        self.assertEqual(classes[(9, 9)], "true_loss")
        self.assertEqual(flow_a.true_loss_count, 2)
        self.assertEqual(flow_a.retransmission_count, 1)
        self.assertEqual(flow_a.reordering_count, 2)

    def test_all_event_versions_share_gap_cause_counts(self):
        signatures = []
        for version in (1, 2, 3):
            session = _session(version=version)
            signatures.append(
                [
                    (
                        flow.flow_id,
                        flow.true_loss_count,
                        flow.retransmission_count,
                        flow.reordering_count,
                    )
                    for flow in session.diagnoses
                ]
            )
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[1], signatures[2])

    def test_shuffled_traversal_produces_identical_conclusions(self):
        session = _session()
        base = {
            "diagnosis": [diagnosis.to_dict() for diagnosis in session.diagnoses],
            "rates": session.rate_report,
        }
        rng = random.Random(20261122)
        frames = list(session.frame_bundle.frames)
        rng.shuffle(frames)
        shuffled_frame_bundle = dataclasses.replace(session.frame_bundle, frames=frames)
        shuffled_event_bundles = []
        for bundle in session.event_bundles:
            records = list(bundle.records)
            rng.shuffle(records)
            shuffled_event_bundles.append(dataclasses.replace(bundle, records=records))
        diagnoses = diagnose_streams(shuffled_frame_bundle, shuffled_event_bundles)
        rates = aggregate_rates(shuffled_frame_bundle, diagnoses, 100)
        shuffled = {
            "diagnosis": [diagnosis.to_dict() for diagnosis in diagnoses],
            "rates": rates,
        }
        self.assertEqual(canonical_json(base), canonical_json(shuffled))
        self.assertEqual(content_digest(base), content_digest(shuffled))

    def test_fixed_point_rates_and_error_affected_stable_tie(self):
        session = _session()
        rates = session.rate_report
        self.assertEqual(rates["rounding"], "half_up")
        self.assertEqual(rates["rate_unit"], "ppm")
        self.assertEqual(rates["quantization_error_upper_ppm"], 1)
        self.assertEqual(rates["combined_error_bound_ppm"], 2)
        by_flow = {flow["flow_id"]: flow for flow in rates["flows"]}
        rate_a = by_flow["10.10.0.1:40000->10.20.0.1:5000/tcp"]
        rate_b = by_flow["10.10.0.2:41000->10.20.0.1:5001/tcp"]
        rate_c = by_flow["10.10.0.3:42000->10.20.0.1:5002/udp"]
        self.assertEqual(rate_a["rate_ppm"], 200000)
        self.assertEqual(rate_b["rate_ppm"], 181818)
        self.assertEqual(rate_c["rate_ppm"], 181818)
        self.assertEqual([item["flow_id"] for item in rates["ranking"]][1:], [rate_b["flow_id"], rate_c["flow_id"]])
        self.assertTrue(all(item["error_affected"] for item in rates["ranking"][1:]))
        self.assertEqual(rates["rate_comparison"][0]["tie_breaker"], "flow_id_asc")
        self.assertTrue(rates["rate_comparison"][0]["decision_affected"])

    def test_windows_are_sorted(self):
        session = _session()
        for flow in session.rate_report["flows"]:
            starts = [window["window_start_ms"] for window in flow["windows"]]
            self.assertEqual(starts, sorted(starts))


class SnapshotTests(unittest.TestCase):
    def test_snapshot_wholesale_replacement_on_version_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            with WorkbenchStore(Path(temp) / "workbench.db") as store:
                snapshot_id = store.save_snapshot(_session(version=3))
                current = _session(version=2)
                restored = restore_snapshot(store, snapshot_id, current_session=current)
            comparison = restored["comparison"]
            self.assertTrue(comparison["replaced"])
            self.assertEqual(comparison["replacement_mode"], "whole_payload_replace")
            self.assertFalse(comparison["incremental_merge"])
            self.assertTrue(any(item["field"] == "event_versions" for item in comparison["differences"]))
            self.assertTrue(all(item["difference_source"] for item in comparison["differences"]))
            self.assertEqual(restored["restored_model"]["version_combo"]["event_list_versions"], ["v3"])
            view = restored_session_view(restored)
            self.assertEqual(view.event_versions, ["v3"])
            self.assertEqual(view.report_digest, _session(version=3).report_digest)

    def test_snapshot_digest_mismatch_is_e_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            with WorkbenchStore(Path(temp) / "workbench.db") as store:
                snapshot_id = store.save_snapshot(_session())
                with self.assertRaises(DiagnosticError) as context:
                    restore_snapshot(
                        store,
                        snapshot_id,
                        inject_digest_mismatch=True,
                    )
            self.assertEqual(context.exception.code, "E_SNAPSHOT")
            self.assertEqual(context.exception.details["snapshot_uid"], snapshot_id)

    def test_snapshot_without_current_session_is_authoritative(self):
        with tempfile.TemporaryDirectory() as temp:
            with WorkbenchStore(Path(temp) / "workbench.db") as store:
                snapshot_id = store.save_snapshot(_session())
                restored = restore_snapshot(store, snapshot_id)
            self.assertTrue(restored["comparison"]["replaced"])
            self.assertEqual(
                restored["comparison"]["replacement_reason"],
                "no_current_session_snapshot_is_authoritative",
            )


class LedgerTests(unittest.TestCase):
    def test_ledger_replay_matches_then_localizes_first_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            with WorkbenchStore(temp_path / "workbench.db") as store:
                session = _session()
                ledger_id = store.save_ledger(session)
                matched = verify_ledger(store, ledger_id)
                self.assertTrue(matched["matches"])
                self.assertIsNone(matched["first_difference"])

                document = json.loads(EVENTS_V3.read_text(encoding="utf-8"))
                document["records"][0]["loss_count"] = 2
                modified = temp_path / "events_modified.json"
                modified.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
                with self.assertRaises(DiagnosticError) as context:
                    verify_ledger(store, ledger_id, event_paths=[str(modified)])
            error = context.exception
            self.assertEqual(error.code, "E_REPLAY")
            self.assertEqual(error.details["flow_id"], "10.10.0.1:40000->10.20.0.1:5000/tcp")
            self.assertEqual(error.details["gap_interval"], {"start_seq": 3, "end_seq": 3})
            self.assertEqual(error.details["field"], "gap_presence")
            self.assertEqual(error.details["ledger_uid"], ledger_id)

    def test_input_digest_difference_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            with WorkbenchStore(temp_path / "workbench.db") as store:
                session = _session()
                ledger_id = store.save_ledger(session)
                document = json.loads(EVENTS_V3.read_text(encoding="utf-8"))
                modified = temp_path / "events_whitespace.json"
                modified.write_text(json.dumps(document, indent=4), encoding="utf-8")
                with self.assertRaises(DiagnosticError) as context:
                    verify_ledger(store, ledger_id, event_paths=[str(modified)])
            self.assertEqual(context.exception.code, "E_REPLAY")
            self.assertEqual(context.exception.details["field"], "inputs.input_digest")


class EchoTests(unittest.TestCase):
    def test_echo_plan_reproduces_gap_shape(self):
        session = _session()
        plan = build_echo_plan(session)
        response = build_echo_response(
            {
                "op": "replay",
                "protocol": plan["protocol"],
                "request_id": "unit",
                "flows": plan["flows"],
            }
        )
        result = verify_echo_response(session, response)
        self.assertTrue(result["matches"])
        self.assertEqual(result["flow_count"], 3)
        self.assertEqual(result["gap_count"], 9)

    def test_echo_mismatch_is_e_echo_with_first_location(self):
        session = _session()
        plan = build_echo_plan(session)
        response = build_echo_response(
            {
                "op": "replay",
                "protocol": plan["protocol"],
                "request_id": "unit",
                "flows": plan["flows"],
            },
            inject_mismatch=True,
        )
        with self.assertRaises(DiagnosticError) as context:
            verify_echo_response(session, response)
        self.assertEqual(context.exception.code, "E_ECHO")
        self.assertEqual(
            context.exception.details["flow_id"],
            "10.10.0.1:40000->10.20.0.1:5000/tcp",
        )
        self.assertEqual(context.exception.details["field"], "gap_presence")

    def test_loopback_tcp_replay_and_ack(self):
        session = _session()
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        with tempfile.TemporaryDirectory() as temp:
            ready = threading.Event()
            state_db = str(Path(temp) / "echo.db")
            thread = threading.Thread(
                target=serve_peer,
                kwargs={
                    "host": "127.0.0.1",
                    "port": port,
                    "state_db": state_db,
                    "once": True,
                    "ack_timeout_seconds": 3.0,
                    "ready_event": ready,
                },
                daemon=True,
            )
            thread.start()
            self.assertTrue(ready.wait(3.0))
            result = run_echo_replay(
                session,
                host="127.0.0.1",
                port=port,
                request_id="unit-loopback",
                send_ack=True,
            )
            self.assertTrue(result["verification"]["matches"])
            thread.join(timeout=3.0)
            recovery = recover_echo_state(state_db)
        self.assertFalse(recovery["complete_incident_reconstruction"])
        self.assertEqual(recovery["recovery_source"], "sqlite_persisted_echo_state_only")
        self.assertEqual(recovery["jobs"][0]["persisted_status"], "acknowledged")
        self.assertTrue(recovery["jobs"][0]["peer_state_confirmed"])

    def test_forced_termination_reports_unconfirmed_peer_state(self):
        with tempfile.TemporaryDirectory() as temp:
            state_db = str(Path(temp) / "echo.db")
            store = EchoStateStore(state_db)
            store.prepare("forced", {"op": "replay", "request_id": "forced"})
            store.mark_sent("forced", {"ok": True, "request_id": "forced"})
            store.close()
            recovery = recover_echo_state(state_db)
        job = recovery["jobs"][0]
        self.assertFalse(recovery["complete_incident_reconstruction"])
        self.assertFalse(job["peer_state_confirmed"])
        self.assertEqual(job["state_confirmation"], "unconfirmed")
        self.assertIn("client_ack", job["unconfirmed_fields"])


class AnomalyCodeTests(unittest.TestCase):
    def test_three_required_anomaly_codes(self):
        with self.assertRaises(DiagnosticError) as wire_context:
            read_wire_bundle(WIRE, MANIFEST, corrupt=True)
        self.assertEqual(wire_context.exception.code, "E_WIRE")

        with self.assertRaises(DiagnosticError) as format_context:
            load_event_bundle(EVENTS_V3, inject_version_conflict=True)
        self.assertEqual(format_context.exception.code, "E_FORMAT")

        with tempfile.TemporaryDirectory() as temp:
            with WorkbenchStore(Path(temp) / "workbench.db") as store:
                snapshot_id = store.save_snapshot(_session())
                with self.assertRaises(DiagnosticError) as snapshot_context:
                    restore_snapshot(store, snapshot_id, inject_digest_mismatch=True)
        self.assertEqual(snapshot_context.exception.code, "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()
