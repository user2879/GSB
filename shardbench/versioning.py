"""配置/环/依赖图/预算四类版本的识别与单调递增。

版本由两部分构成：内容摘要（判断是否真的变化）与单调递增序号（参与缓存校验）。
新摘要与上一版本不同则序号 +1，相同则保持，保证序号永不倒退。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ShardConfig


@dataclass(frozen=True)
class VersionState:
    config_version: int
    ring_version: int
    dependency_version: int
    budget_version: int
    config_digest: str
    ring_digest: str
    dependency_digest: str
    budget_digest: str

    def as_dict(self) -> dict[str, object]:
        return {
            "config_version": self.config_version,
            "ring_version": self.ring_version,
            "dependency_version": self.dependency_version,
            "budget_version": self.budget_version,
            "config_digest": self.config_digest,
            "ring_digest": self.ring_digest,
            "dependency_digest": self.dependency_digest,
            "budget_digest": self.budget_digest,
        }

    def numeric_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.config_version,
            self.ring_version,
            self.dependency_version,
            self.budget_version,
        )


def initial_versions() -> VersionState:
    return VersionState(1, 1, 1, 1, "", "", "", "")


def derive_versions(previous: VersionState | None, config: ShardConfig) -> VersionState:
    base = previous or initial_versions()
    return VersionState(
        config_version=_advance(base.config_version, base.config_digest, config.config_digest),
        ring_version=_advance(base.ring_version, base.ring_digest, config.ring_digest),
        dependency_version=_advance(
            base.dependency_version,
            base.dependency_digest,
            config.dependency_digest,
        ),
        budget_version=_advance(base.budget_version, base.budget_digest, config.budget_digest),
        config_digest=config.config_digest,
        ring_digest=config.ring_digest,
        dependency_digest=config.dependency_digest,
        budget_digest=config.budget_digest,
    )


def _advance(old_number: int, old_digest: str, new_digest: str) -> int:
    if not old_digest:
        return 1
    if old_digest == new_digest:
        return old_number
    return old_number + 1


def diff_versions(old: VersionState, new: VersionState) -> dict[str, bool]:
    return {
        "config": old.config_digest != new.config_digest,
        "ring": old.ring_digest != new.ring_digest,
        "dependency": old.dependency_digest != new.dependency_digest,
        "budget": old.budget_digest != new.budget_digest,
    }
