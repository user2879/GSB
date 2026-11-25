"""Deterministic engine orchestrating arbitration, scheduling and ledger."""

import json

from .canonical import canonical_text, digest
from .error_bounds import round_error_attribution, session_error_attribution
from .evidence import (
    arbitrate_round,
    build_reference_graph,
    synthesize_session_view,
)
from .ledger import (
    Ledger,
    expected_from_config,
    recover_unknown_rounds,
    verify_replay_round,
)
from .scheduler import Scheduler


class Engine:
    def __init__(
        self,
        dataset,
        ledger_path,
        backup_dir,
        global_limit=3,
        per_session_limit=2,
        batch_size=8,
    ):
        self.dataset = dataset
        self.ledger_path = ledger_path
        self.backup_dir = backup_dir
        self.global_limit = global_limit
        self.per_session_limit = per_session_limit
        self.batch_size = batch_size
        self.round_records = {}
        self.error_records = {}
        self.schedule_reports = {}
        self.views = {}
        self.graph = None
        self.ledger = Ledger(
            ledger_path, backup_dir, expected_from_config(dataset.config)
        )

    def _all_jobs(self):
        jobs = []
        for session in sorted(
            self.dataset.sessions, key=lambda item: item["session_id"]
        ):
            session_id = session["session_id"]
            for request in sorted(session["rounds"], key=lambda item: item["round"]):
                jobs.append((session_id, request))
        return jobs

    def _execute_jobs(self, jobs):
        scheduler = Scheduler(self.global_limit, self.per_session_limit)
        submitted = []
        for session_id, request in jobs:
            job = scheduler.submit(
                session_id,
                request["round"],
                lambda sid=session_id, req=request: arbitrate_round(
                    self.dataset, sid, req
                ),
            )
            submitted.append(job)
        reports = scheduler.shutdown(submitted)
        limits = scheduler.observed_limits()
        for job, report in zip(submitted, reports):
            record = job.result
            error_record = round_error_attribution(record)
            self.round_records.setdefault(record["session_id"], {})
            self.round_records[record["session_id"]][record["round"]] = record
            self.error_records.setdefault(record["session_id"], {})
            self.error_records[record["session_id"]][record["round"]] = error_record
            self.schedule_reports.setdefault(record["session_id"], {})
            self.schedule_reports[record["session_id"]][record["round"]] = report
        return reports, limits

    def _commit_rounds(self, session_rounds):
        entries = []
        cursors = []
        for session_id in sorted(session_rounds):
            rounds = sorted(session_rounds[session_id])
            for round_index in rounds:
                entries.append(
                    {
                        "session_id": session_id,
                        "round": round_index,
                        "round_record": self.round_records[session_id][round_index],
                        "error_record": self.error_records[session_id][round_index],
                    }
                )
            total = sum(
                1
                for session in self.dataset.sessions
                if session["session_id"] == session_id
                for _ in session["rounds"]
            )
            known = sorted(self.round_records[session_id])
            cursors.append(
                {
                    "session_id": session_id,
                    "last_committed_round": known[-1],
                    "total_rounds": total,
                }
            )
        batch_meta = []
        for start in range(0, len(entries), self.batch_size):
            chunk = entries[start : start + self.batch_size]
            cursor_chunk = self._cursors_for_entries(chunk)
            batch_meta.append(
                self.ledger.write_batch(
                    chunk,
                    cursor_chunk,
                    [
                        {
                            "session_id": entry["session_id"],
                            "round": entry["round"],
                            "round_digest": entry["round_record"]["round_digest"],
                        }
                        for entry in chunk
                    ],
                )
            )
        return batch_meta

    def _cursors_for_entries(self, entries):
        cursors = []
        grouped = {}
        for entry in entries:
            grouped.setdefault(entry["session_id"], set()).add(entry["round"])
        for session_id in sorted(grouped):
            total = next(
                len(session["rounds"])
                for session in self.dataset.sessions
                if session["session_id"] == session_id
            )
            known = sorted(self.round_records[session_id])
            cursors.append(
                {
                    "session_id": session_id,
                    "last_committed_round": known[-1],
                    "total_rounds": total,
                }
            )
        return cursors

    def run(self):
        jobs = self._all_jobs()
        reports, limits = self._execute_jobs(jobs)
        session_rounds = {}
        for session_id, request in jobs:
            session_rounds.setdefault(session_id, set()).add(request["round"])
        batches = self._commit_rounds(session_rounds)
        return self._finalize(reports, limits, batches)

    def rerun_invalid(self, old_dataset, invalidation):
        if self.ledger.replacement is None:
            self._hydrate_committed_rounds()
        else:
            all_jobs = self._all_jobs()
            self._execute_jobs(all_jobs)
            session_rounds = {}
            for session_id, request in all_jobs:
                session_rounds.setdefault(session_id, set()).add(request["round"])
            self._commit_rounds(session_rounds)
        jobs = []
        wanted = {
            (item["session_id"], item["round"])
            for item in invalidation["invalid_rounds"]
        }
        for session_id, request in self._all_jobs():
            if (session_id, request["round"]) in wanted:
                jobs.append((session_id, request))
        reports, limits = self._execute_jobs(jobs)
        session_rounds = {}
        for session_id, request in jobs:
            session_rounds.setdefault(session_id, set()).add(request["round"])
        batches = self._commit_rounds(session_rounds)
        self.graph = build_reference_graph(self.dataset, self.round_records)
        for session in sorted(
            self.dataset.sessions, key=lambda item: item["session_id"]
        ):
            session_id = session["session_id"]
            records = [
                self.round_records[session_id][request["round"]]
                for request in sorted(
                    session["rounds"], key=lambda item: item["round"]
                )
            ]
            attrs = [
                self.error_records[session_id][request["round"]]
                for request in sorted(
                    session["rounds"], key=lambda item: item["round"]
                )
            ]
            self.views[session_id] = synthesize_session_view(
                self.dataset.config,
                session_id,
                records,
                self.graph["graph_digest"],
            )
            self.views[session_id]["session_error"] = session_error_attribution(
                session_id, records, attrs
            )
        return {
            "rerun_jobs": [
                {"session_id": sid, "round": req["round"]} for sid, req in jobs
            ],
            "schedule_reports": reports,
            "limits": limits,
            "batches": batches,
        }

    def _hydrate_committed_rounds(self):
        import json

        state = self.ledger.committed_state()
        for row in state["rounds"]:
            round_record = json.loads(row["round_json"])
            error_record = json.loads(row["error_json"])
            self.round_records.setdefault(row["session_id"], {})
            self.round_records[row["session_id"]][row["round_index"]] = round_record
            self.error_records.setdefault(row["session_id"], {})
            self.error_records[row["session_id"]][row["round_index"]] = error_record

    def _finalize(self, reports, limits, batches):
        self.graph = build_reference_graph(self.dataset, self.round_records)
        for session in sorted(
            self.dataset.sessions, key=lambda item: item["session_id"]
        ):
            session_id = session["session_id"]
            records = [
                self.round_records[session_id][request["round"]]
                for request in sorted(
                    session["rounds"], key=lambda item: item["round"]
                )
            ]
            attrs = [
                self.error_records[session_id][request["round"]]
                for request in sorted(
                    session["rounds"], key=lambda item: item["round"]
                )
            ]
            self.views[session_id] = synthesize_session_view(
                self.dataset.config,
                session_id,
                records,
                self.graph["graph_digest"],
            )
            self.views[session_id]["session_error"] = session_error_attribution(
                session_id, records, attrs
            )
        result = self.snapshot_result(reports, limits, batches)
        return result

    def snapshot_result(self, reports=None, limits=None, batches=None):
        deterministic_reports = [
            {
                "session_id": report["session_id"],
                "round": report["round"],
                "queued_under_bounded_scheduler": True,
            }
            for report in sorted(
                (reports if reports is not None else [
                    report
                    for per_session in self.schedule_reports.values()
                    for report in per_session.values()
                ]),
                key=lambda item: (item["session_id"], item["round"]),
            )
        ]
        return {
            "versions": dict(self.dataset.config.versions),
            "config_content_digest": self.dataset.config.digest,
            "graph": self.graph,
            "sessions": [
                self._session_payload(session["session_id"])
                for session in sorted(
                    self.dataset.sessions, key=lambda item: item["session_id"]
                )
            ],
            "schedule": {
                "reports": deterministic_reports,
                "observed_limits": limits or {},
            },
            "ledger_batches": batches or [],
            "ledger_replacement": self.ledger.replacement,
        }

    def _session_payload(self, session_id):
        records = [
            self.round_records[session_id][request["round"]]
            for session in self.dataset.sessions
            if session["session_id"] == session_id
            for request in sorted(session["rounds"], key=lambda item: item["round"])
        ]
        attrs = [
            self.error_records[session_id][record["round"]] for record in records
        ]
        return {
            "session_id": session_id,
            "rounds": records,
            "error_attribution": attrs,
            "view": self.views[session_id],
            "schedule": [
                {
                    "round": record["round"],
                    "queued_under_bounded_scheduler": True,
                }
                for record in records
            ],
        }

    def write_outputs(self, output_path):
        payload = self.snapshot_result()
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(canonical_text(payload))
            handle.write("\n")
        summary = {
            "config_content_digest": payload["config_content_digest"],
            "graph_digest": payload["graph"]["graph_digest"],
            "view_digests": {
                session_id: self.views[session_id]["view_digest"]
                for session_id in sorted(self.views)
            },
            "round_digests": {
                session_id: {
                    str(round_index): record["round_digest"]
                    for round_index, record in sorted(
                        self.round_records[session_id].items()
                    )
                }
                for session_id in sorted(self.round_records)
            },
        }
        summary_path = output_path + ".summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(canonical_text(summary))
            handle.write("\n")
        audit_path = output_path + ".schedule-audit.json"
        timing_reports = [
            report
            for session_id in sorted(self.schedule_reports)
            for per_session in [self.schedule_reports[session_id]]
            for report in sorted(
                per_session.values(),
                key=lambda item: (item["session_id"], item["round"]),
            )
        ]
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump({"timing_reports": timing_reports}, handle,
                      sort_keys=True, indent=2)
            handle.write("\n")
        return output_path, summary_path, digest(payload)

    def close(self):
        self.ledger.close()


def recover_and_replay(dataset, ledger_path, backup_dir, **engine_kwargs):
    engine = Engine(dataset, ledger_path, backup_dir, **engine_kwargs)
    state = engine.ledger.committed_state()
    unknown = recover_unknown_rounds(dataset, state)
    stored = {
        (row["session_id"], row["round_index"]): row for row in state["rounds"]
    }
    jobs_by_key = {
        (session["session_id"], request["round"]): request
        for session in dataset.sessions
        for request in session["rounds"]
    }
    for key in sorted(stored):
        row = stored[key]
        request = jobs_by_key[key]
        actual = arbitrate_round(dataset, key[0], request)
        actual_error = round_error_attribution(actual)
        verify_replay_round(
            key[0],
            request,
            json.loads(row["round_json"]),
            json.loads(row["error_json"]),
            actual,
            actual_error,
        )
        engine.round_records.setdefault(key[0], {})[key[1]] = actual
        engine.error_records.setdefault(key[0], {})[key[1]] = actual_error

    rerun_jobs = [
        (item["session_id"], jobs_by_key[(item["session_id"], item["round"])])
        for item in unknown
    ]
    if rerun_jobs:
        reports, limits = engine._execute_jobs(rerun_jobs)
        session_rounds = {}
        for session_id, request in rerun_jobs:
            session_rounds.setdefault(session_id, set()).add(request["round"])
        batches = engine._commit_rounds(session_rounds)
    else:
        reports, limits, batches = [], {}, []
    result = engine._finalize(reports, limits, batches)
    result["recovery"] = {
        "committed_batches": len(state["batches"]),
        "committed_rounds": len(state["rounds"]),
        "unknown_rerun_rounds": unknown,
        "replacement": engine.ledger.replacement,
        "replay_status": "all committed rounds matched ledger byte-for-field",
    }
    return engine, result
