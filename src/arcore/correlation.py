"""Exact pairwise and fingerprint-bucketed approximate correlation."""

from dataclasses import dataclass

from .errors import DivergeError
from .model import Alert, Topology
from .spec import Parameters


def pair_key(left: Alert, right: Alert) -> tuple[str, str]:
    """Canonical unordered pair identifier."""
    return tuple(sorted((left.alert_id, right.alert_id)))  # type: ignore[return-value]


def _pair_record(left: Alert, right: Alert, reason: str) -> dict:
    return {
        "pair": list(pair_key(left, right)),
        "timestamp_ms": max(left.timestamp_ms, right.timestamp_ms),
        "services": sorted((left.service_id, right.service_id)),
        "fingerprints": sorted((left.fingerprint, right.fingerprint)),
        "reason": reason,
    }


def _is_candidate(left: Alert, right: Alert, topology: Topology,
                  window_ms: int) -> bool:
    if abs(left.timestamp_ms - right.timestamp_ms) > window_ms:
        return False
    if not topology.are_related(left.service_id, right.service_id):
        return False
    return True


def exact_correlate(alerts: list[Alert], topology: Topology,
                    window_ms: int) -> list[dict]:
    """Inspect every candidate pair inside the time window (O(n * window))."""
    ordered = sorted(alerts, key=lambda a: (
        a.timestamp_ms, a.service_id, a.fingerprint, a.alert_id))
    results: list[dict] = []
    left = 0
    for right in range(len(ordered)):
        while ordered[right].timestamp_ms - ordered[left].timestamp_ms > window_ms:
            left += 1
        for j in range(left, right):
            candidate = ordered[j]
            current = ordered[right]
            if _is_candidate(candidate, current, topology, window_ms):
                results.append(_pair_record(
                    candidate, current,
                    "exact:time-adjacent+topology-related"))
    results.sort(key=lambda record: (record["pair"], record["reason"]))
    return results


def approximate_correlate(alerts: list[Alert], topology: Topology,
                          window_ms: int) -> list[dict]:
    """Bucket by fingerprint, evaluate the same predicate only within buckets."""
    buckets: dict[str, list[Alert]] = {}
    for alert in alerts:
        buckets.setdefault(alert.fingerprint, []).append(alert)
    results: list[dict] = []
    for fingerprint in sorted(buckets):
        bucket = sorted(buckets[fingerprint], key=lambda a: (
            a.timestamp_ms, a.service_id, a.fingerprint, a.alert_id))
        left = 0
        for right in range(len(bucket)):
            while bucket[right].timestamp_ms - bucket[left].timestamp_ms > window_ms:
                left += 1
            for j in range(left, right):
                candidate = bucket[j]
                current = bucket[right]
                if _is_candidate(candidate, current, topology, window_ms):
                    results.append(_pair_record(
                        candidate, current,
                        "approx:same-fingerprint-bucket+time-adjacent+topology-related"))
    results.sort(key=lambda record: (record["pair"], record["reason"]))
    return results


def pair_set(pairs: list[dict]) -> set[tuple[str, str]]:
    return {tuple(record["pair"]) for record in pairs}  # type: ignore[misc]


@dataclass(frozen=True)
class PathDecision:
    selected_path: str
    reason: str
    alert_count: int
    fingerprint_cardinality: int
    exact_alert_threshold: int
    fingerprint_cardinality_threshold: int
    boundary: dict

    def to_dict(self) -> dict:
        return {
            "selected_path": self.selected_path,
            "reason": self.reason,
            "alert_count": self.alert_count,
            "fingerprint_cardinality": self.fingerprint_cardinality,
            "exact_alert_threshold": self.exact_alert_threshold,
            "fingerprint_cardinality_threshold": self.fingerprint_cardinality_threshold,
            "boundary": self.boundary,
        }


def select_path(alerts: list[Alert], parameters: Parameters) -> PathDecision:
    alert_count = len(alerts)
    cardinality = len({alert.fingerprint for alert in alerts})
    alert_triggered = alert_count > parameters.exact_alert_threshold
    cardinality_triggered = (
        cardinality > parameters.fingerprint_cardinality_threshold)
    boundary = {
        "alert_threshold_breach": alert_triggered,
        "alert_headroom": parameters.exact_alert_threshold - alert_count,
        "cardinality_threshold_breach": cardinality_triggered,
        "cardinality_headroom": (
            parameters.fingerprint_cardinality_threshold - cardinality),
    }
    if alert_triggered or cardinality_triggered:
        reasons = []
        if alert_triggered:
            reasons.append(
                f"alert_count {alert_count} > "
                f"exact_alert_threshold {parameters.exact_alert_threshold}")
        if cardinality_triggered:
            reasons.append(
                f"fingerprint_cardinality {cardinality} > threshold "
                f"{parameters.fingerprint_cardinality_threshold}")
        return PathDecision(
            selected_path="approximate",
            reason="; ".join(reasons),
            alert_count=alert_count, fingerprint_cardinality=cardinality,
            exact_alert_threshold=parameters.exact_alert_threshold,
            fingerprint_cardinality_threshold=(
                parameters.fingerprint_cardinality_threshold),
            boundary=boundary)
    return PathDecision(
        selected_path="exact",
        reason=(
            f"alert_count {alert_count} <= "
            f"{parameters.exact_alert_threshold} and fingerprint_cardinality "
            f"{cardinality} <= {parameters.fingerprint_cardinality_threshold}; "
            "no scale rule breached, exact pairwise path stays default"),
        alert_count=alert_count, fingerprint_cardinality=cardinality,
        exact_alert_threshold=parameters.exact_alert_threshold,
        fingerprint_cardinality_threshold=(
            parameters.fingerprint_cardinality_threshold),
        boundary=boundary)


@dataclass(frozen=True)
class ErrorBounds:
    measured_false_positives: int
    measured_misses: int
    structural_false_positive_upper_bound: int
    structural_miss_upper_bound: int
    excluded_pairs: int
    evaluated_bucket_pairs: int
    exact_pair_count: int
    rationale: str

    def to_dict(self) -> dict:
        return {
            "measured_false_positives": self.measured_false_positives,
            "measured_misses": self.measured_misses,
            "structural_false_positive_upper_bound": (
                self.structural_false_positive_upper_bound),
            "structural_miss_upper_bound": self.structural_miss_upper_bound,
            "excluded_pairs": self.excluded_pairs,
            "evaluated_bucket_pairs": self.evaluated_bucket_pairs,
            "exact_pair_count": self.exact_pair_count,
            "rationale": self.rationale,
        }


def compare_paths(alerts: list[Alert], topology: Topology,
                  window_ms: int, exact_pairs: list[dict],
                  approximate_pairs: list[dict]) -> ErrorBounds:
    exact = pair_set(exact_pairs)
    approximate = pair_set(approximate_pairs)
    false_positives = approximate - exact
    if false_positives:
        first = sorted(false_positives)[0]
        raise DivergeError(
            f"approximate path produced pair {first} absent from exact path",
            pair=first)
    misses = exact - approximate
    buckets: dict[str, list[Alert]] = {}
    for alert in alerts:
        buckets.setdefault(alert.fingerprint, []).append(alert)
    evaluated = sum(
        len(bucket) * (len(bucket) - 1) // 2
        for bucket in buckets.values())
    total_pairs = len(alerts) * (len(alerts) - 1) // 2
    excluded_pairs = max(0, total_pairs - evaluated)
    rationale = (
        "Approximate evaluation is a subset restriction of the exact predicate: "
        "a same-bucket pair runs the identical time/topology test, so its false "
        "positive upper bound is 0. Every exact pair is either re-evaluated in a "
        "shared fingerprint bucket or excluded as a cross-bucket pair, hence the "
        "miss upper bound equals the exact pair count; a tighter combinatorial "
        "bound is the number of cross-bucket candidate pairs reported as "
        "excluded_pairs, and the tightest bound is the measured miss count. Bounds "
        "are recomputed per stream alongside the path thresholds.")
    return ErrorBounds(
        measured_false_positives=0,
        measured_misses=len(misses),
        structural_false_positive_upper_bound=0,
        structural_miss_upper_bound=min(len(exact), excluded_pairs),
        excluded_pairs=excluded_pairs,
        evaluated_bucket_pairs=evaluated,
        exact_pair_count=len(exact),
        rationale=rationale)
