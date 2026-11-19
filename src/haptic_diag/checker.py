"""Device receipt reconciliation with typed first-mismatch causes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .constants import CHANNEL_INDEX, PROTOCOL_V2
from .frames import BatchDigest, FrameRecord
from .quantization import DEFAULT_QUANT_PROFILE, QuantProfile
from .transport import DeviceReceipt, ReceiptChannel

CAUSE_QUANTIZATION = "quantization_difference"
CAUSE_PROTOCOL = "protocol_field_difference"
CAUSE_TRANSPORT = "transport_missing"


@dataclass(frozen=True)
class CheckIssue:
    seq: int
    cause: str
    field: str
    expected: Any
    actual: Any
    within_tolerance: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "cause": self.cause,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "within_tolerance": self.within_tolerance,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CheckResult:
    status: str
    first_mismatch: CheckIssue | None
    issues: tuple[CheckIssue, ...]
    confirmed_sequences: tuple[int, ...]
    pending_sequences: tuple[int, ...]
    unexpected_receipt_sequences: tuple[int, ...]
    digest_consistent: bool | None
    digest_mismatch_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "first_mismatch": (
                None if self.first_mismatch is None else self.first_mismatch.to_dict()
            ),
            "issues": [issue.to_dict() for issue in self.issues],
            "confirmed_sequences": list(self.confirmed_sequences),
            "pending_sequences": list(self.pending_sequences),
            "unexpected_receipt_sequences": list(self.unexpected_receipt_sequences),
            "digest_consistent": self.digest_consistent,
            "digest_mismatch_only": self.digest_mismatch_only,
        }


def _issue(
    seq: int,
    cause: str,
    field: str,
    expected: Any,
    actual: Any,
    within_tolerance: bool,
    detail: str,
) -> CheckIssue:
    return CheckIssue(
        seq=seq,
        cause=cause,
        field=field,
        expected=expected,
        actual=actual,
        within_tolerance=within_tolerance,
        detail=detail,
    )


def _compare_channel(
    frame: FrameRecord,
    receipt: DeviceReceipt,
    profile: QuantProfile,
) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    expected_channels = frame.channel_by_name
    actual_channels = receipt.channel_by_name
    if frame.protocol_version != receipt.protocol_version:
        issues.append(
            _issue(
                frame.seq,
                CAUSE_PROTOCOL,
                "protocol_version",
                frame.protocol_version,
                receipt.protocol_version,
                False,
                "receipt protocol version differs from the compiled frame",
            )
        )
    if frame.timestamp_us != receipt.timestamp_us:
        issues.append(
            _issue(
                frame.seq,
                CAUSE_PROTOCOL,
                "timestamp_us",
                frame.timestamp_us,
                receipt.timestamp_us,
                False,
                "frame metadata differs; timestamp is an exact wire field",
            )
        )
    if frame.mask != receipt.mask:
        issues.append(
            _issue(
                frame.seq,
                CAUSE_PROTOCOL,
                "channel_mask",
                frame.mask,
                receipt.mask,
                False,
                "wire channel mask differs",
            )
        )

    for channel in frame.channels:
        actual = actual_channels.get(channel.channel)
        if actual is None:
            issues.append(
                _issue(
                    frame.seq,
                    CAUSE_PROTOCOL,
                    f"channel:{channel.channel}",
                    channel.amplitude_code,
                    None,
                    False,
                    "expected channel is absent from the receipt",
                )
            )
            continue
        if actual.amplitude_code != channel.amplitude_code:
            amplitude_step = profile.amplitude_step
            difference = abs(actual.amplitude - channel.amplitude)
            within_tolerance = difference <= amplitude_step + 1e-15
            issues.append(
                _issue(
                    frame.seq,
                    CAUSE_QUANTIZATION,
                    f"channel:{channel.channel}:amplitude",
                    channel.amplitude_code,
                    actual.amplitude_code,
                    within_tolerance,
                    (
                        f"amplitude codes differ by {abs(actual.amplitude_code - channel.amplitude_code)}; "
                        f"declared half-step is {profile.amplitude_max_error}"
                    ),
                )
            )
        if frame.protocol_version == PROTOCOL_V2:
            if actual.frequency_code is None or actual.frequency_hz is None:
                issues.append(
                    _issue(
                        frame.seq,
                        CAUSE_PROTOCOL,
                        f"channel:{channel.channel}:frequency_hz",
                        channel.frequency_code,
                        None,
                        False,
                        "v1 receipt is missing the v2 frequency field",
                    )
                )
            elif actual.frequency_code != channel.frequency_code:
                difference = abs(actual.frequency_hz - channel.frequency_hz)
                within_tolerance = difference <= profile.frequency_step + 1e-15
                issues.append(
                    _issue(
                        frame.seq,
                        CAUSE_QUANTIZATION,
                        f"channel:{channel.channel}:frequency_hz",
                        channel.frequency_code,
                        actual.frequency_code,
                        within_tolerance,
                        (
                            f"frequency codes differ by {abs(actual.frequency_code - channel.frequency_code)}; "
                            f"declared half-step is {profile.frequency_max_error}"
                        ),
                    )
                )
    for channel in receipt.channels:
        if channel.channel not in expected_channels:
            issues.append(
                _issue(
                    frame.seq,
                    CAUSE_PROTOCOL,
                    f"channel:{channel.channel}",
                    None,
                    channel.amplitude_code,
                    False,
                    "receipt contains a channel not present in the compiled frame",
                )
            )
    return issues


def reconcile_receipts(
    frames: Sequence[FrameRecord],
    receipts: Iterable[DeviceReceipt],
    expected_batch_digests: Sequence[BatchDigest] = (),
    observed_batch_digests: Sequence[str] = (),
    observed_batch_ranges: Sequence[tuple[int, int]] = (),
    profile: QuantProfile = DEFAULT_QUANT_PROFILE,
) -> CheckResult:
    expected_frames = {frame.seq: frame for frame in sorted(frames, key=lambda item: item.seq)}
    receipt_by_seq: dict[int, DeviceReceipt] = {}
    duplicate_receipts: set[int] = set()
    for receipt in receipts:
        if receipt.seq in receipt_by_seq:
            duplicate_receipts.add(receipt.seq)
        else:
            receipt_by_seq[receipt.seq] = receipt

    issues: list[CheckIssue] = []
    confirmed: list[int] = []
    pending: list[int] = []
    for seq in sorted(expected_frames):
        frame = expected_frames[seq]
        receipt = receipt_by_seq.get(seq)
        if receipt is None:
            pending.append(seq)
            issues.append(
                _issue(
                    seq,
                    CAUSE_TRANSPORT,
                    "frame",
                    frame.content_hash,
                    None,
                    False,
                    "no receipt received; frame remains pending confirmation",
                )
            )
            continue
        frame_issues = _compare_channel(frame, receipt, profile)
        if frame_issues:
            issues.extend(frame_issues)
        else:
            confirmed.append(seq)
    for seq in sorted(duplicate_receipts):
        issues.append(
            _issue(
                seq,
                CAUSE_TRANSPORT,
                "receipt",
                1,
                2,
                False,
                "duplicate receipt for the same frame sequence",
            )
        )
    unexpected = tuple(sorted(set(receipt_by_seq) - set(expected_frames)))
    for seq in unexpected:
        issues.append(
            _issue(
                seq,
                CAUSE_PROTOCOL,
                "seq",
                None,
                seq,
                False,
                "receipt sequence was not part of the expected frame batch",
            )
        )

    digest_consistent: bool | None = None
    if expected_batch_digests and observed_batch_digests:
        if observed_batch_ranges:
            expected_by_range = {
                (item.first_seq, item.last_seq): item.digest
                for item in expected_batch_digests
            }
            observed_by_range = dict(zip(observed_batch_ranges, observed_batch_digests))
            if set(expected_by_range) == set(observed_by_range):
                digest_consistent = all(
                    expected_by_range[key] == observed_by_range[key]
                    for key in expected_by_range
                )
        elif len(expected_batch_digests) == len(observed_batch_digests):
            expected_values = [item.digest for item in expected_batch_digests]
            digest_consistent = expected_values == list(observed_batch_digests)
    issues_sorted = tuple(sorted(issues, key=lambda item: (item.seq, item.field, item.cause)))
    first_mismatch = issues_sorted[0] if issues_sorted else None
    digest_mismatch_only = digest_consistent is False and first_mismatch is None
    non_missing_issues = [issue for issue in issues_sorted if issue.cause != CAUSE_TRANSPORT]
    if non_missing_issues:
        status = "mismatch"
    elif pending:
        status = "pending_confirmation"
    elif duplicate_receipts or unexpected:
        status = "mismatch"
    else:
        status = "confirmed"
    return CheckResult(
        status=status,
        first_mismatch=first_mismatch,
        issues=issues_sorted,
        confirmed_sequences=tuple(confirmed),
        pending_sequences=tuple(pending),
        unexpected_receipt_sequences=unexpected,
        digest_consistent=digest_consistent,
        digest_mismatch_only=digest_mismatch_only,
    )
