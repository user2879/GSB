"""Deterministic local sample generation."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from .config import parse_config
from .engine import ArbitrationEngine
from .loader import load_evidence_text
from .numeric import sha256_bytes, write_json
from .rules import parse_rules


SEED = 20261121


EVIDENCE_TYPES = {
    "failed_login": "0.22",
    "credential_stuffing": "0.34",
    "mfa_push": "0.18",
    "known_device": "0.25",
    "sla_risk": "0.50",
    "billing_dispute": "0.30",
    "phishing_report": "0.28",
    "admin_override": "0.40",
    "telemetry_anomaly": "0.16",
    "low_signal": "0.001",
}
ROLE_TRUST = {
    "customer": "0.78",
    "agent": "0.92",
    "external": "0.55",
    "monitor": "1.00",
    "admin": "1.00",
    "backend": "0.88",
}
ROLE_BY_TYPE = {
    "failed_login": ("external", "customer"),
    "credential_stuffing": ("external",),
    "mfa_push": ("backend",),
    "known_device": ("customer",),
    "sla_risk": ("monitor",),
    "billing_dispute": ("customer",),
    "phishing_report": ("external", "customer"),
    "admin_override": ("admin",),
    "telemetry_anomaly": ("backend",),
    "low_signal": ("backend",),
}
TERMS = (
    "otp",
    "lockout",
    "invoice",
    "chargeback",
    "geo-jump",
    "device",
    "phish",
    "sla",
    "latency",
    "admin",
    "quarantine",
    "token",
    "replay",
    "refund",
    "travel",
    "known-device",
    "escalation",
    "audit",
)


def _rules_document() -> dict:
    return {
        "schema": "arbitration-rules/v1",
        "version": "2026.11",
        "evidence_type_weights": EVIDENCE_TYPES,
        "role_trust": ROLE_TRUST,
        "default_support_bonus": "0.05",
        "relations": [
            {
                "id": "rel-credential-vs-device",
                "kind": "exclusive",
                "left": {"evidence_type": "credential_stuffing", "source_roles": ["external"]},
                "right": {"evidence_type": "known_device", "source_roles": ["customer"]},
            },
            {
                "id": "rel-login-supports-mfa",
                "kind": "support",
                "left": {"evidence_type": "failed_login", "source_roles": ["external"]},
                "right": {"evidence_type": "mfa_push", "source_roles": ["backend"]},
                "bonus": "0.06",
            },
            {
                "id": "rel-admin-vs-billing",
                "kind": "exclusive",
                "left": {"evidence_type": "admin_override", "source_roles": ["admin"]},
                "right": {"evidence_type": "billing_dispute", "source_roles": ["customer"]},
            },
            {
                "id": "rel-telemetry-supports-sla",
                "kind": "support",
                "left": {"evidence_type": "telemetry_anomaly", "source_roles": ["backend"]},
                "right": {"evidence_type": "sla_risk", "source_roles": ["monitor"]},
                "bonus": "0.08",
            },
        ],
        "priority_bands": [
            {"level": 0, "name": "P0", "min_score": "2.10"},
            {"level": 1, "name": "P1", "min_score": "1.50"},
            {"level": 2, "name": "P2", "min_score": "0.90"},
            {"level": 3, "name": "P3", "min_score": "0.00"},
        ],
    }


def _config_document() -> dict:
    return {
        "fixed_decimal_places": 6,
        "quantization_units": 1,
        "decay_half_life": 3600,
        "exclusion_strength": "0.25",
        "support_bonus": "0.05",
        "fusion": "sum",
        "decision_threshold": "1.0",
        "threshold_equality_upgrades": True,
        "interaction_budget_ms": 120,
        "worker_threads": 4,
        "snapshot_commit_interval": 8,
        "snapshot_dir": "snapshots",
        "temp_dir": "tmp",
        "target_version": "v2",
    }


def _terms(rng: random.Random) -> list[str]:
    return sorted(rng.sample(TERMS, rng.randint(2, 5)))


def _regular_record(
    rng: random.Random,
    ticket_id: str,
    base_timestamp: int,
) -> list[dict]:
    count = rng.randint(2, 6)
    rows: list[dict] = []
    for seq in range(1, count + 1):
        evidence_type = rng.choice(sorted(EVIDENCE_TYPES))
        roles = ROLE_BY_TYPE[evidence_type]
        source_role = roles[rng.randrange(len(roles))]
        timestamp = base_timestamp + rng.randint(0, 10_800)
        rows.append(
            {
                "ticket_id": ticket_id,
                "evidence_seq": seq,
                "evidence_type": evidence_type,
                "source_role": source_role,
                "timestamp": timestamp,
                "terms": _terms(rng),
            }
        )
    if len({row["timestamp"] for row in rows}) == 1:
        rows[-1]["timestamp"] += 3600
    return rows


def _boundary_record(ticket_id: str, base_timestamp: int) -> list[dict]:
    # Two quantized 1.0-decay scores of 0.5 plus one effectively-zero score
    # produce exactly the configured 1.0 threshold.
    return [
        {
            "ticket_id": ticket_id,
            "evidence_seq": 1,
            "evidence_type": "sla_risk",
            "source_role": "monitor",
            "timestamp": base_timestamp,
            "terms": ["sla", "latency"],
        },
        {
            "ticket_id": ticket_id,
            "evidence_seq": 2,
            "evidence_type": "sla_risk",
            "source_role": "monitor",
            "timestamp": base_timestamp,
            "terms": ["sla", "escalation"],
        },
        {
            "ticket_id": ticket_id,
            "evidence_seq": 3,
            "evidence_type": "low_signal",
            "source_role": "backend",
            "timestamp": base_timestamp + 864_000,
            "terms": ["telemetry", "low-signal"],
        },
    ]


def _invalid_rows() -> list[str]:
    def encoded(record: dict) -> str:
        return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    return [
        "{not-json",
        encoded(
            {
                "ticket_id": "T9001",
                "evidence_seq": 1,
                "evidence_type": "unregistered_signal",
                "source_role": "customer",
                "timestamp": 100,
                "terms": ["unknown"],
            }
        ),
        encoded(
            {
                "ticket_id": "T9002",
                "evidence_seq": 1,
                "evidence_type": "failed_login",
                "source_role": "unknown_role",
                "timestamp": 101,
                "terms": ["role"],
            }
        ),
        encoded(
            {
                "ticket_id": "T9003",
                "evidence_seq": 1,
                "evidence_type": "failed_login",
                "source_role": "external",
                "timestamp": -1,
                "terms": ["negative"],
            }
        ),
        encoded(
            {
                "ticket_id": "T9004",
                "evidence_seq": 1,
                "evidence_type": "failed_login",
                "source_role": "external",
                "timestamp": 200,
                "terms": ["same"],
            }
        ),
        encoded(
            {
                "ticket_id": "T9004",
                "evidence_seq": 2,
                "evidence_type": "mfa_push",
                "source_role": "backend",
                "timestamp": 200,
                "terms": ["time"],
            }
        ),
        encoded(
            {
                "ticket_id": "T9005",
                "evidence_seq": 1,
                "evidence_type": "failed_login",
                "source_role": "external",
                "timestamp": 300,
                "terms": ["duplicate"],
            }
        ),
        encoded(
            {
                "ticket_id": "T9005",
                "evidence_seq": 1,
                "evidence_type": "known_device",
                "source_role": "customer",
                "timestamp": 301,
                "terms": ["pair"],
            }
        ),
        encoded(
            {
                "ticket_id": "T9006",
                "evidence_seq": 1,
                "evidence_type": "failed_login",
                "source_role": "external",
                "timestamp": 400,
                "terms": ["kept"],
            }
        ),
        encoded(
            {
                "ticket_id": "T9006",
                "evidence_seq": 2,
                "evidence_type": "mfa_push",
                "source_role": "backend",
                "timestamp": 401,
                "terms": ["valid"],
            }
        ),
    ]


def _exception_document(boundary_ticket: str, culprit_ticket: str) -> dict:
    return {
        "schema": "exception-injections/v1",
        "seed": SEED,
        "injections": [
            {
                "active": False,
                "ticket_id": boundary_ticket,
                "stage": "score",
                "exception_type": "SyntheticScoreFailure",
                "message": "inactive sample injection for score stage",
            },
            {
                "active": False,
                "ticket_id": culprit_ticket,
                "stage": "relations",
                "exception_type": "SyntheticRelationFailure",
                "message": "inactive sample injection for relation stage",
            },
            {
                "active": False,
                "ticket_id": culprit_ticket,
                "stage": "decision",
                "exception_type": "SyntheticDecisionFailure",
                "message": "inactive sample injection for decision stage",
            },
            {
                "active": False,
                "ticket_id": culprit_ticket,
                "stage": "convert",
                "exception_type": "SyntheticConversionFailure",
                "message": "inactive sample injection for conversion stage",
            },
            {
                "active": False,
                "ticket_id": culprit_ticket,
                "stage": "score",
                "action": "delay",
                "delay_ms": 50,
                "exception_type": "DelayOnly",
                "message": "inactive latency injection",
            },
        ],
    }


def generate_samples(root: str | Path, *, seed: int = SEED, generate_version_samples: bool = True) -> dict:
    root = Path(root)
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    for stale in (root / "snapshots" / "CURRENT",):
        if stale.exists():
            stale.unlink()
    generations = root / "snapshots" / "generations"
    if generations.exists():
        shutil.rmtree(generations)
    generations.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rows: list[dict] = []
    base_epoch = 1_760_000_000
    boundary_ticket = "T0047"
    culprit_ticket = "T0048"
    for index in range(1, 49):
        ticket_id = f"T{index:04d}"
        base_timestamp = base_epoch + index * 10_000
        if ticket_id == boundary_ticket:
            ticket_rows = _boundary_record(ticket_id, base_timestamp)
        else:
            ticket_rows = _regular_record(rng, ticket_id, base_timestamp)
        rows.extend(ticket_rows)

    evidence_text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows
    ) + "\n"
    rules = _rules_document()
    config = _config_document()
    (samples / "evidence.jsonl").write_text(evidence_text, encoding="utf-8")
    (samples / "rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (samples / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    invalid_text = "\n".join(_invalid_rows()) + "\n"
    (samples / "evidence_invalid.jsonl").write_text(invalid_text, encoding="utf-8")
    exception_doc = _exception_document(boundary_ticket, culprit_ticket)
    (samples / "exception_injections.json").write_text(
        json.dumps(exception_doc, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    version_info: dict = {}
    if generate_version_samples:
        version_dir = samples / "version_samples"
        if version_dir.exists():
            shutil.rmtree(version_dir)
        version_dir.mkdir(parents=True, exist_ok=True)
        parsed_rules = parse_rules(rules)
        parsed_config = parse_config(config)
        loaded = load_evidence_text(evidence_text, parsed_rules)
        engine = ArbitrationEngine(
            rules=parsed_rules,
            config=parsed_config,
            evidence_lines=evidence_text.splitlines(),
            workspace_root=root,
            cache_store=None,
            snapshot_store=None,
            output_dir="samples/version_samples",
            write_outputs=True,
        )
        result = engine.run(use_cache=False, mode="full")
        stale_diagnostic = version_dir / "diagnostic_report.json"
        if stale_diagnostic.exists():
            stale_diagnostic.unlink()
        artifacts = {
            name: sha256_bytes((version_dir / name).read_bytes())
            for name in (
                "arbitration.v1.txt",
                "arbitration.v1.txt.meta.json",
                "arbitration.v2.json",
                "arbitration.v3.bin",
            )
        }
        version_info = {
            "seed": seed,
            "record_count": len(result.tickets),
            "failure_count": len(result.failures),
            "boundary_ticket_ids": [
                ticket["ticket_id"] for ticket in result.tickets if ticket.get("boundary")
            ],
            "artifacts": artifacts,
            "roundtrip": result.output_info["conversion_report"].get("triple_roundtrip", {}),
        }
        write_json(version_dir / "sample_manifest.json", version_info, temp_dir=root / "tmp")

    manifest = {
        "schema": "local-sample-bundle/v1",
        "seed": seed,
        "ticket_count": 48,
        "evidence_count": len(rows),
        "boundary_ticket": boundary_ticket,
        "invalid_sample_count": 10,
        "files": {
            "evidence": "samples/evidence.jsonl",
            "invalid_evidence": "samples/evidence_invalid.jsonl",
            "rules": "samples/rules.json",
            "config": "samples/config.json",
            "exception_injections": "samples/exception_injections.json",
            "version_samples": "samples/version_samples",
        },
        "version_samples": version_info,
    }
    write_json(samples / "sample_manifest.json", manifest, temp_dir=root / "tmp")
    return manifest
