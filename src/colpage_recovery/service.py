from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

from .codec import (
    content_digest,
    decode_page_bytes,
    initial_content,
    read_page_file,
    sha256_hex,
    version_digest,
    write_page_file,
)
from .engine import ReplayEngine, ReplayResult
from .errors import (
    LedgerStateError,
    ReplayMismatchError,
    SnapshotMismatchError,
)
from .ledger import Ledger
from .loader import load_bundle
from .models import Bundle


class DiagnosticService:
    def __init__(
        self,
        state_dir: str | Path,
        bundle: Bundle | None = None,
        sample_dir: str | Path | None = None,
    ) -> None:
        if bundle is None:
            if sample_dir is None:
                raise ValueError("bundle or sample_dir is required")
            bundle = load_bundle(sample_dir)
        self.bundle = bundle
        self.state_dir = Path(state_dir)
        self.engine = ReplayEngine(bundle)
        self.ledger: Ledger | None = None
        self.generation: int | None = None

    def __enter__(self) -> "DiagnosticService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
            self.ledger = None

    def initialize(self, *, reset: bool = False) -> int:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.state_dir / "recovery.db"
        if reset:
            if db_path.exists():
                db_path.unlink()
            for suffix in ("-journal", "-wal", "-shm"):
                sidecar = Path(str(db_path) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            pages_dir = self.state_dir / "pages"
            if pages_dir.exists():
                shutil.rmtree(pages_dir)
        self.ledger = Ledger(db_path)
        self.ledger.migrate()
        generation, _is_new = self.ledger.start_generation(self.bundle)
        self.generation = generation
        self.ledger.store_wal_entries(generation, self.bundle)
        for page in self.bundle.pages:
            state = initial_content(page)
            data = state.encode(page)
            digest = content_digest(data)
            vdigest = version_digest(page.page_id, 0, 0, digest)
            relative = self._page_relative_path(generation, page.page_id, 0)
            write_page_file(self.state_dir / relative, data)
            self.ledger.insert_baseline_version(
                generation,
                page.page_id,
                relative.as_posix(),
                len(data),
                digest,
                vdigest,
            )
        return generation

    def require_ledger(self) -> Ledger:
        if self.ledger is None:
            self.initialize(reset=False)
        assert self.ledger is not None
        if self.generation is None:
            self.generation = self.ledger.current_generation()
        return self.ledger

    @staticmethod
    def _page_relative_path(generation: int, page_id: str, version: int) -> Path:
        safe = hashlib.sha256(page_id.encode("utf-8")).hexdigest()[:20]
        return Path("pages") / f"g{generation}" / safe / f"v{version:06d}.colpage"

    def _absolute_path(self, relative: str | Path) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return self.state_dir / path

    def replay_and_commit(
        self,
        *,
        crash_before_round: int | None = None,
    ) -> dict[str, Any]:
        ledger = self.require_ledger()
        generation = ledger.current_generation()
        self.generation = generation
        cursor = ledger.get_cursor(generation)
        if cursor is None:
            raise LedgerStateError("progress cursor is missing", field="progress_cursor")
        cursor_lsn = int(cursor["last_committed_lsn"])
        states = self._load_latest_states(generation)
        usage = {
            row["page_id"]: int(row["bytes_consumed"])
            for row in ledger.query_dicts(
                "SELECT page_id, bytes_consumed FROM budget_usage WHERE generation=?",
                (generation,),
            )
        }
        committed_rounds: list[dict[str, Any]] = []

        for round_no, group in enumerate(self.bundle.batches(), start=1):
            ordered = sorted(group, key=lambda entry: entry.lsn)
            if ordered[-1].lsn <= cursor_lsn:
                continue
            attempt_no = ledger.next_attempt_no(generation, round_no)
            partial = self.engine.replay_groups(
                [ordered],
                initial_states=states,
                start_round=round_no,
                initial_page_usage=usage,
            )
            batch = partial.batches[0]
            batch_id = ledger.begin_batch(generation, round_no, attempt_no, batch, self.bundle)
            content_paths: dict[int, str] = {}
            for record in batch.records:
                relative = self._page_relative_path(generation, record.page_id, record.version)
                write_page_file(self._absolute_path(relative), record.content_bytes)
                content_paths[record.lsn] = relative.as_posix()

            should_crash = (
                crash_before_round == round_no
                and crash_before_round is not None
                and attempt_no == 1
            )

            def before_commit() -> None:
                if should_crash:
                    os._exit(137)

            revision_no = ledger.complete_batch(
                generation=generation,
                batch_id=batch_id,
                round_no=round_no,
                attempt_no=attempt_no,
                batch=batch,
                content_paths=content_paths,
                before_commit=before_commit if should_crash else None,
            )
            for page_id, added in batch.page_bytes.items():
                usage[page_id] = usage.get(page_id, 0) + added
            committed_rounds.append(
                {
                    "round_no": round_no,
                    "attempt_no": attempt_no,
                    "batch_id": batch_id,
                    "revision_no": revision_no,
                    "start_lsn": ordered[0].lsn,
                    "end_lsn": ordered[-1].lsn,
                    "slot_peak": batch.slot_peak,
                    "byte_consumed": batch.byte_consumed,
                }
            )

        cursor = ledger.get_cursor(generation)
        assert cursor is not None
        latest_rows = ledger.latest_visible_page_versions(generation)
        view_rows = [
            {
                "page_id": row["page_id"],
                "version": row["version"],
                "lsn": row["lsn"],
                "visible": bool(row["visible"]),
                "after_crash": bool(row["after_crash"]),
                "content_digest": row["content_digest"],
                "version_digest": row["version_digest"],
                "byte_length": row["byte_length"],
            }
            for row in latest_rows
        ]
        revision_no = int(cursor["current_revision"])
        view_id = ledger.replace_current_view(
            generation,
            revision_no,
            int(cursor["last_committed_lsn"]),
            view_rows,
            self.engine.crash_point,
        )
        return {
            "generation": generation,
            "committed_rounds": committed_rounds,
            "cursor_lsn": int(cursor["last_committed_lsn"]),
            "revision_no": revision_no,
            "view_snapshot_id": view_id,
        }

    def _load_latest_states(self, generation: int) -> dict[str, Any]:
        ledger = self.require_ledger()
        states: dict[str, Any] = {}
        for row in ledger.latest_page_versions(generation):
            path = self._absolute_path(row["content_path"])
            data = read_page_file(path, row["content_digest"])
            states[row["page_id"]] = decode_page_bytes(data)
        return states

    def _states_from_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for page in checkpoint["pages"]:
            path = self._absolute_path(page["content_path"])
            try:
                data = path.read_bytes()
            except FileNotFoundError as exc:
                raise SnapshotMismatchError(
                    "checkpoint page file is missing",
                    field="content_path",
                    lsn=page["lsn"],
                    page_id=page["page_id"],
                    details={"path": str(path)},
                ) from exc
            actual_digest = content_digest(data)
            if actual_digest != page["content_digest"]:
                raise SnapshotMismatchError(
                    "checkpoint content digest mismatch",
                    field="content_digest",
                    lsn=page["lsn"],
                    page_id=page["page_id"],
                    details={"expected": page["content_digest"], "actual": actual_digest},
                )
            actual_version_digest = version_digest(
                page["page_id"], page["version"], page["lsn"], actual_digest
            )
            if actual_version_digest != page["version_digest"]:
                raise SnapshotMismatchError(
                    "checkpoint version digest mismatch",
                    field="version_digest",
                    lsn=page["lsn"],
                    page_id=page["page_id"],
                )
            states[page["page_id"]] = decode_page_bytes(data)
        return states

    def verify_ledger_content_files(self, generation: int) -> list[dict[str, Any]]:
        ledger = self.require_ledger()
        rows = ledger.query_dicts(
            "SELECT * FROM page_version WHERE generation=? ORDER BY lsn,page_id,version",
            (generation,),
        )
        checked: list[dict[str, Any]] = []
        for row in rows:
            path = self._absolute_path(row["content_path"])
            try:
                data = path.read_bytes()
            except FileNotFoundError as exc:
                raise SnapshotMismatchError(
                    "ledger page content file is missing",
                    field="content_path",
                    lsn=row["lsn"],
                    page_id=row["page_id"],
                    details={"path": str(path)},
                ) from exc
            actual_digest = content_digest(data)
            if actual_digest != row["content_digest"]:
                raise SnapshotMismatchError(
                    "ledger content digest does not match persisted page bytes",
                    field="content_digest",
                    lsn=row["lsn"],
                    page_id=row["page_id"],
                    details={
                        "expected_ledger_digest": row["content_digest"],
                        "actual_file_digest": actual_digest,
                    },
                )
            actual_vdigest = version_digest(
                row["page_id"], row["version"], row["lsn"], actual_digest
            )
            if actual_vdigest != row["version_digest"]:
                raise SnapshotMismatchError(
                    "ledger version digest does not match page content",
                    field="version_digest",
                    lsn=row["lsn"],
                    page_id=row["page_id"],
                )
            registry = ledger.query_dicts(
                "SELECT digest FROM digest_registry WHERE generation=? AND kind='page_content' "
                "AND page_id=? AND version=? AND lsn=?",
                (generation, row["page_id"], row["version"], row["lsn"]),
            )
            if not registry or registry[0]["digest"] != row["content_digest"]:
                raise SnapshotMismatchError(
                    "digest registry does not match page_version ledger",
                    field="digest_registry",
                    lsn=row["lsn"],
                    page_id=row["page_id"],
                )
            checked.append(
                {
                    "page_id": row["page_id"],
                    "version": row["version"],
                    "lsn": row["lsn"],
                    "content_digest": row["content_digest"],
                }
            )
        return checked

    def _scoped_groups(
        self, base_lsn: int, end_lsn: int
    ) -> tuple[list[list[Any]], int]:
        groups: list[list[Any]] = []
        first_round = 1
        all_groups = self.bundle.batches()
        for index, group in enumerate(all_groups, start=1):
            scoped = sorted(
                (entry for entry in group if base_lsn < entry.lsn <= end_lsn),
                key=lambda entry: entry.lsn,
            )
            if scoped:
                if not groups:
                    first_round = index
                groups.append(scoped)
        return groups, first_round

    def _compare_replay_to_reference_and_ledger(
        self,
        replayed: ReplayResult,
        reference: ReplayResult,
        *,
        check_ledger: bool,
        phase: str,
    ) -> list[dict[str, Any]]:
        ledger = self.require_ledger()
        generation = ledger.current_generation()
        reference_by_lsn = reference.lsn_record_map()
        steps: list[dict[str, Any]] = []

        for batch in replayed.batches:
            if check_ledger:
                round_rows = ledger.query_dicts(
                    "SELECT round_no,attempt_no,start_lsn,end_lsn,status "
                    "FROM replay_batch WHERE generation=? AND round_no=? AND status='committed' "
                    "ORDER BY attempt_no DESC LIMIT 1",
                    (generation, batch.round_no),
                )
                if not round_rows:
                    raise LedgerStateError(
                        "replay round is not present as a committed ledger batch",
                        field="round_no",
                        details={"round_no": batch.round_no, "phase": phase},
                    )
                round_row = round_rows[0]
                round_lsns = [entry.lsn for entry in batch.entries]
                if not (
                    int(round_row["start_lsn"]) <= min(round_lsns)
                    and max(round_lsns) <= int(round_row["end_lsn"])
                ):
                    raise LedgerStateError(
                        "replayed interval is outside recorded ledger round",
                        field="lsn",
                        details={"ledger_round": round_row, "replayed_lsns": round_lsns},
                    )
            for actual in batch.records:
                expected = reference_by_lsn[actual.lsn]
                if actual.page_id != expected.page_id or actual.version != expected.version:
                    raise ReplayMismatchError(
                        "replayed page version differs from uninterrupted reference",
                        field="version",
                        lsn=actual.lsn,
                        page_id=actual.page_id,
                        details={
                            "expected_page_id": expected.page_id,
                            "expected_version": expected.version,
                            "actual_page_id": actual.page_id,
                            "actual_version": actual.version,
                            "phase": phase,
                        },
                    )
                if actual.content_digest != expected.content_digest:
                    raise ReplayMismatchError(
                        "replayed content digest differs from uninterrupted reference",
                        field="content_digest",
                        lsn=actual.lsn,
                        page_id=actual.page_id,
                        details={
                            "expected": expected.content_digest,
                            "actual": actual.content_digest,
                            "phase": phase,
                        },
                    )
                if actual.version_digest != expected.version_digest:
                    raise ReplayMismatchError(
                        "replayed version digest differs from uninterrupted reference",
                        field="version_digest",
                        lsn=actual.lsn,
                        page_id=actual.page_id,
                        details={"phase": phase},
                    )

                if check_ledger:
                    rows = ledger.query_dicts(
                        "SELECT * FROM page_version WHERE generation=? AND lsn=?",
                        (generation, actual.lsn),
                    )
                    if len(rows) != 1:
                        raise LedgerStateError(
                            "expected exactly one committed page_version for lsn",
                            field="page_version.lsn",
                            lsn=actual.lsn,
                            page_id=actual.page_id,
                            details={"row_count": len(rows)},
                        )
                    stored = rows[0]
                    if int(stored["version"]) != expected.version:
                        raise SnapshotMismatchError(
                            "ledger page version does not match deterministic replay",
                            field="version",
                            lsn=actual.lsn,
                            page_id=actual.page_id,
                            details={
                                "ledger": stored["version"],
                                "expected": expected.version,
                            },
                        )
                    if stored["content_digest"] != expected.content_digest:
                        raise SnapshotMismatchError(
                            "ledger content digest does not match deterministic replay",
                            field="content_digest",
                            lsn=actual.lsn,
                            page_id=actual.page_id,
                            details={
                                "ledger": stored["content_digest"],
                                "expected": expected.content_digest,
                            },
                        )

            steps.append(
                {
                    "phase": phase,
                    "round_no": batch.round_no,
                    "lsns": [entry.lsn for entry in batch.entries],
                    "record_count": len(batch.records),
                    "slot_peak": batch.slot_peak,
                    "status": "matched",
                }
            )
        return steps

    def _compare_visibility_sets(
        self, reference: ReplayResult, *, base_lsn: int
    ) -> dict[str, Any]:
        ledger = self.require_ledger()
        generation = ledger.current_generation()
        expected_committed = set(reference.committed_txids)
        actual_committed = {
            row["tx_id"]
            for row in ledger.query_dicts(
                "SELECT DISTINCT tx_id FROM page_version "
                "WHERE generation=? AND visible=1 AND tx_id<>''",
                (generation,),
            )
        }
        if actual_committed != expected_committed:
            difference = sorted(actual_committed.symmetric_difference(expected_committed))
            first_tx = difference[0] if difference else None
            first_record = next(
                (record for record in reference.records if record.tx_id == first_tx),
                None,
            )
            raise ReplayMismatchError(
                "committed transaction set differs from uninterrupted reference",
                field="committed_txids",
                tx_id=first_tx,
                lsn=first_record.lsn if first_record else None,
                page_id=first_record.page_id if first_record else None,
                details={
                    "expected": sorted(expected_committed),
                    "actual": sorted(actual_committed),
                },
            )

        expected_invisible = {
            (item["page_id"], item["version"], item["lsn"])
            for item in reference.invisible_versions
            if item["lsn"] > base_lsn
        }
        actual_invisible = {
            (row["page_id"], int(row["version"]), int(row["lsn"]))
            for row in ledger.query_dicts(
                "SELECT page_id,version,lsn FROM page_version "
                "WHERE generation=? AND visible=0 AND lsn>?",
                (generation, base_lsn),
            )
        }
        if actual_invisible != expected_invisible:
            expected_first = sorted(expected_invisible - actual_invisible)[:1]
            actual_first = sorted(actual_invisible - expected_invisible)[:1]
            location = (expected_first or actual_first or [(None, None, None)])[0]
            raise ReplayMismatchError(
                "invisible version set differs from uninterrupted reference",
                field="invisible_versions",
                page_id=location[0],
                lsn=location[2],
                details={
                    "expected_only": sorted(expected_invisible - actual_invisible),
                    "actual_only": sorted(actual_invisible - expected_invisible),
                },
            )
        return {
            "committed_txids": sorted(actual_committed),
            "invisible_version_count": len(actual_invisible),
            "invisible_versions": [
                {"page_id": page_id, "version": version, "lsn": lsn}
                for page_id, version, lsn in sorted(actual_invisible)
            ],
        }

    def recover_and_verify(self) -> dict[str, Any]:
        ledger = self.require_ledger()
        generation = ledger.current_generation()
        self.generation = generation
        unknown_batches = ledger.mark_running_unknown(generation)
        cursor_before = ledger.get_cursor(generation)
        checkpoint = ledger.latest_checkpoint(generation)
        self.verify_ledger_content_files(generation)

        reference = self.engine.replay_all()
        base_lsn = int(checkpoint["at_lsn"]) if checkpoint else 0
        checkpoint_states = (
            self._states_from_checkpoint(checkpoint) if checkpoint else None
        )
        verification_steps: list[dict[str, Any]] = []

        cursor_lsn_before = int(cursor_before["last_committed_lsn"]) if cursor_before else 0
        if base_lsn < cursor_lsn_before:
            groups, first_round = self._scoped_groups(base_lsn, cursor_lsn_before)
            overlap = self.engine.replay_groups(
                groups,
                initial_states=checkpoint_states,
                start_round=first_round,
            )
            verification_steps.extend(
                self._compare_replay_to_reference_and_ledger(
                    overlap,
                    reference,
                    check_ledger=True,
                    phase="committed_checkpoint_overlap",
                )
            )

        resume = self.replay_and_commit()
        self.verify_ledger_content_files(generation)

        end_lsn = max(entry.lsn for entry in self.bundle.entries)
        groups, first_round = self._scoped_groups(base_lsn, end_lsn)
        recovered = self.engine.replay_groups(
            groups,
            initial_states=self._states_from_checkpoint(checkpoint) if checkpoint else None,
            start_round=first_round,
        )
        verification_steps.extend(
            self._compare_replay_to_reference_and_ledger(
                recovered,
                reference,
                check_ledger=True,
                phase="full_recovery_interval",
            )
        )
        visibility = self._compare_visibility_sets(reference, base_lsn=base_lsn)
        cursor_after = ledger.get_cursor(generation)
        latest_checkpoint = ledger.latest_checkpoint(generation)
        return {
            "status": "verified",
            "code": "OK",
            "generation": generation,
            "unknown_batches": unknown_batches,
            "cursor_before": cursor_before,
            "resume": resume,
            "cursor_after": cursor_after,
            "recovery_base_checkpoint": checkpoint,
            "latest_checkpoint": latest_checkpoint,
            "verification_steps": verification_steps,
            "visibility": visibility,
            "reference_final_versions": reference.final_versions,
            "reference_checkpoints": [
                {"checkpoint_no": cp.checkpoint_no, "at_lsn": cp.at_lsn}
                for cp in reference.checkpoints
            ],
            "digest_fingerprint": sha256_hex(
                "|".join(
                    f"{page_id}:{digest}"
                    for page_id, digest in sorted(
                        json_safe_digest_map(reference.final_content).items()
                    )
                )
            ),
        }


def json_safe_digest_map(content: dict[str, bytes]) -> dict[str, str]:
    return {page_id: content_digest(data) for page_id, data in sorted(content.items())}
