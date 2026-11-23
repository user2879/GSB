"""Deterministic alert stream generation, validation and ordering."""

from dataclasses import dataclass

from .canonical import canonical_json, digest_bytes, digest_of
from .errors import SpecError
from .model import Alert, SourceSpec
from .prng import SplitMix64
from .spec import Declaration


OPEN = "OPEN"
CLOSED = "CLOSED"


@dataclass(frozen=True)
class SourceSummary:
    source_id: str
    record_count: int
    open_count: int
    closed_count: int
    fingerprint_count: int
    min_ts: int
    max_ts: int
    digest: str

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "record_count": self.record_count,
            "open_count": self.open_count,
            "closed_count": self.closed_count,
            "fingerprint_count": self.fingerprint_count,
            "min_ts": self.min_ts,
            "max_ts": self.max_ts,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class StreamResult:
    alerts: tuple[Alert, ...]
    source_summaries: tuple[SourceSummary, ...]
    global_digest: str

    def to_dict(self) -> dict:
        return {
            "alerts": [alert.to_dict() for alert in self.alerts],
            "source_summaries": [s.to_dict() for s in self.source_summaries],
            "global_digest": self.global_digest,
        }


def generate_source(spec: SourceSpec) -> list[Alert]:
    """Generate one source's alerts; identical specs yield identical bytes."""
    rng = SplitMix64(spec.seed)
    services = sorted(spec.services)
    levels = sorted(spec.levels)
    # Deterministic bijection seq -> fingerprint slot, unique within source.
    # Slots are small enough that different sources legitimately collide on
    # shared fingerprints, which gives the bucketed path real same-bucket work
    # while the source prefix keeps each source's keys unique on their own.
    if spec.record_count > spec.fingerprint_namespace:
        raise SpecError(
            "fingerprint_namespace must be >= record_count",
            source_id=spec.source_id, field="fingerprint_namespace")
    if spec.record_count:
        slots = rng.permutation_indices(spec.fingerprint_namespace)
        chosen_slots = sorted(slots[:spec.record_count])
    else:
        chosen_slots = []
    alerts: list[Alert] = []
    timestamp = spec.start_ts_ms
    for seq in range(spec.record_count):
        if seq > 0:
            timestamp += 1 + rng.randrange(max(spec.max_gap_ms, 1))
        service = services[rng.randrange(len(services))]
        level = levels[rng.randrange(len(levels))]
        fingerprint = f"fp-{chosen_slots[seq]:08x}"
        label_count = rng.randrange(spec.max_labels + 1)
        if label_count:
            indices = sorted(
                rng.permutation_indices(len(spec.label_vocabulary))[:label_count])
            labels = tuple(spec.label_vocabulary[i] for i in indices)
        else:
            labels = ()
        status = CLOSED if rng.random_unit() < spec.closed_probability else OPEN
        alerts.append(Alert(
            alert_id=f"{spec.source_id}#a{seq:08d}",
            source_id=spec.source_id,
            timestamp_ms=timestamp,
            service_id=service,
            level=level,
            fingerprint=fingerprint,
            labels=labels,
            status=status,
            seq=seq,
        ))
    return alerts


def _summarize_source(spec: SourceSpec, alerts: list[Alert]) -> SourceSummary:
    fingerprints = {alert.fingerprint for alert in alerts}
    return SourceSummary(
        source_id=spec.source_id,
        record_count=len(alerts),
        open_count=sum(1 for a in alerts if a.status == OPEN),
        closed_count=sum(1 for a in alerts if a.status == CLOSED),
        fingerprint_count=len(fingerprints),
        min_ts=alerts[0].timestamp_ms if alerts else spec.start_ts_ms,
        max_ts=alerts[-1].timestamp_ms if alerts else spec.start_ts_ms,
        digest=digest_bytes(
            canonical_json([alert.to_dict() for alert in alerts])),
    )


def validate_stream(spec: SourceSpec, alerts: list[Alert]) -> None:
    """Check source-local monotonic timestamps and unique fingerprints."""
    previous_ts: int | None = None
    fingerprints: set[str] = set()
    for alert in alerts:
        if previous_ts is not None and alert.timestamp_ms < previous_ts:
            raise SpecError(
                "timestamps must be non-decreasing within a source",
                source_id=spec.source_id, field="timestamp_ms")
        previous_ts = alert.timestamp_ms
        if alert.fingerprint in fingerprints:
            raise SpecError(
                "fingerprint keys must be unique within a source",
                source_id=spec.source_id, field="fingerprint")
        fingerprints.add(alert.fingerprint)


def stable_sort(alerts: list[Alert]) -> list[Alert]:
    """Order by (timestamp, service, fingerprint); dict/hash order independent."""
    return sorted(alerts, key=lambda a: (
        a.timestamp_ms, a.service_id, a.fingerprint, a.alert_id))


def generate_stream(declaration: Declaration) -> StreamResult:
    per_source: list[list[Alert]] = []
    summaries: list[SourceSummary] = []
    for spec in declaration.sources:
        alerts = generate_source(spec)
        validate_stream(spec, alerts)
        per_source.append(alerts)
        summaries.append(_summarize_source(spec, alerts))
    ordered = stable_sort([alert for alerts in per_source for alert in alerts])
    global_digest = digest_bytes(canonical_json(
        [alert.to_dict() for alert in ordered]))
    return StreamResult(
        alerts=tuple(ordered),
        source_summaries=tuple(summaries),
        global_digest=global_digest,
    )


def stream_digest(alerts: list[Alert] | tuple[Alert, ...]) -> str:
    return digest_of([alert.to_dict() for alert in alerts])
