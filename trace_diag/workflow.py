from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import AnalysisBundle, Analyzer
from .ledger import Ledger, manifest_digest
from .spec import PoliciesDoc, ServicesDoc


class SimulatedCrash(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowResult:
    bundle: AnalysisBundle
    view: dict[str, Any]
    run_id: str
    mode: str
    newly_committed: tuple[str, ...]
    skipped_committed: tuple[str, ...]


def analyze_and_persist(
    services_doc: ServicesDoc,
    policies_doc: PoliciesDoc,
    db_path: str | Path,
    *,
    logger: Any | None = None,
    crash_after: int | None = None,
    include_sensitivity: bool = True,
) -> WorkflowResult:
    with Ledger(db_path) as ledger:
        effective_services, effective_policies = ledger.allocate_versions(
            services_doc, policies_doc
        )
        digest = manifest_digest(effective_services, effective_policies)
        existing = ledger._run_by_manifest(digest, ("complete",))
        if existing and existing["active"] and not existing["stale"]:
            bundle = Analyzer(effective_services, effective_policies).build_bundle(
                include_sensitivity=include_sensitivity
            )
            view = ledger.get_run_view(existing["run_id"], verify=True)
            assert view is not None
            return WorkflowResult(
                bundle=bundle,
                view=view,
                run_id=existing["run_id"],
                mode="existing",
                newly_committed=(),
                skipped_committed=(),
            )

        bundle = Analyzer(effective_services, effective_policies).build_bundle(
            include_sensitivity=include_sensitivity
        )
        run = ledger.create_or_resume_run(effective_services, effective_policies)
        run_id = run["run_id"]
        if logger:
            logger.event(
                "run_open",
                context={
                    "run_id": run_id,
                    "mode": run["mode"],
                    "strategy_version": effective_policies.version,
                    "service_version": effective_services.version,
                    "manifest_digest": digest,
                },
            )
        batches = ledger.batch_map(run_id)
        newly_committed: list[str] = []
        skipped_committed: list[str] = []
        for ordinal, service in enumerate(effective_services.services):
            analysis = bundle.service_analyses[ordinal]
            batch = batches[ordinal]
            if batch["status"] == "committed":
                skipped_committed.append(service.id)
                continue
            ledger.commit_service(
                run_id,
                ordinal,
                analysis,
                list(bundle.ledger_flips),
            )
            newly_committed.append(service.id)
            if logger:
                logger.event(
                    "batch_committed",
                    context={
                        "run_id": run_id,
                        "ordinal": ordinal,
                        "service_id": service.id,
                        "progress_cursor": ordinal + 1,
                    },
                )
            if crash_after is not None and len(newly_committed) >= crash_after:
                raise SimulatedCrash(
                    f"intentional stop after {len(newly_committed)} committed batch(es)"
                )
        ledger.finalize_run(run_id)
        view = ledger.get_run_view(run_id, verify=True)
        assert view is not None
        if logger:
            logger.event(
                "run_complete",
                context={
                    "run_id": run_id,
                    "snapshot_digest": view["run"]["snapshot_digest"],
                    "newly_committed": newly_committed,
                    "skipped_committed": skipped_committed,
                },
            )
        return WorkflowResult(
            bundle=bundle,
            view=view,
            run_id=run_id,
            mode=run["mode"],
            newly_committed=tuple(newly_committed),
            skipped_committed=tuple(skipped_committed),
        )


def recover_latest(
    db_path: str | Path,
    *,
    logger: Any | None = None,
) -> WorkflowResult:
    with Ledger(db_path) as ledger:
        runs = ledger.recoverable_runs()
        if not runs:
            raise RuntimeError("no recoverable ledger run")
        run = runs[0]
        services_doc, policies_doc = ledger.run_documents(run["run_id"])
        unknown = ledger.unknown_batches(run["run_id"])
        if logger:
            logger.event(
                "recovery_started",
                context={
                    "run_id": run["run_id"],
                    "unknown_services": [batch["service_id"] for batch in unknown],
                },
            )
    return analyze_and_persist(
        services_doc,
        policies_doc,
        db_path,
        logger=logger,
        include_sensitivity=True,
    )
