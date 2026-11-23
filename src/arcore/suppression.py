"""Closed-alert silence suppression keyed by fingerprint."""

from dataclasses import dataclass, field

from .model import Alert
from .stream import CLOSED


@dataclass
class SuppressionGroup:
    fingerprint: str
    representative_alert_id: str
    representative_timestamp_ms: int
    suppressed_alert_ids: list[str]
    member_alert_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "representative_alert_id": self.representative_alert_id,
            "representative_timestamp_ms": self.representative_timestamp_ms,
            "suppressed_alert_ids": list(self.suppressed_alert_ids),
            "member_alert_ids": list(self.member_alert_ids),
        }


@dataclass
class SuppressionResult:
    groups: list[SuppressionGroup] = field(default_factory=list)
    suppressed_alert_ids: list[str] = field(default_factory=list)
    participating_alert_ids: list[str] = field(default_factory=list)

    @property
    def suppressed_count(self) -> int:
        return len(self.suppressed_alert_ids)

    def to_dict(self) -> dict:
        return {
            "suppressed_count": self.suppressed_count,
            "suppressed_alert_ids": list(self.suppressed_alert_ids),
            "participating_alert_ids": list(self.participating_alert_ids),
            "groups": [group.to_dict() for group in self.groups],
        }


def suppress_closed(alerts: list[Alert], silence_window_ms: int) -> SuppressionResult:
    """Aggregate CLOSED alerts sharing a fingerprint inside silence windows.

    Grouping uses fingerprint only and timestamps are sorted explicitly, so the
    outcome is independent of arrival order or batch boundaries.
    """
    buckets: dict[str, list[Alert]] = {}
    for alert in alerts:
        if alert.status == CLOSED:
            buckets.setdefault(alert.fingerprint, []).append(alert)
    result = SuppressionResult()
    for fingerprint in sorted(buckets):
        members = sorted(buckets[fingerprint], key=lambda a: (
            a.timestamp_ms, a.alert_id))
        window_start = members[0].timestamp_ms
        representative = members[0]
        suppressed: list[str] = []
        for alert in members[1:]:
            if alert.timestamp_ms - window_start <= silence_window_ms:
                suppressed.append(alert.alert_id)
            else:
                window_start = alert.timestamp_ms
        result.groups.append(SuppressionGroup(
            fingerprint=fingerprint,
            representative_alert_id=representative.alert_id,
            representative_timestamp_ms=representative.timestamp_ms,
            suppressed_alert_ids=suppressed,
            member_alert_ids=[alert.alert_id for alert in members]))
        result.participating_alert_ids.extend(
            alert.alert_id for alert in members)
        result.suppressed_alert_ids.extend(suppressed)
    result.groups.sort(key=lambda group: group.fingerprint)
    result.suppressed_alert_ids.sort()
    result.participating_alert_ids.sort()
    return result
