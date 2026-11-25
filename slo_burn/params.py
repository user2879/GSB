"""参数版本演进：编辑应用、失效传播与受影响集合推导。"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ServiceSpec


@dataclass(frozen=True)
class ParamEdit:
    """一次参数编辑（参数版本号严格递增）。"""

    service_id: str
    param_version: int
    effective_after_minute: int
    changes: dict[str, int]
    line: int


@dataclass(frozen=True)
class VersionedSpec:
    """带生效区间的服务声明：version 在 (effective_from, effective_to] 分钟内有效。"""

    spec: ServiceSpec
    effective_from: int
    effective_to: int | None  # None 表示当前版本，仍生效

    def covers(self, minute: int) -> bool:
        return minute > self.effective_from and (
            self.effective_to is None or minute <= self.effective_to
        )


def build_version_chain(
    base_specs: list[ServiceSpec],
    edits: list[ParamEdit],
    max_minute: int | None = None,
) -> dict[str, list[VersionedSpec]]:
    """把基础声明与编辑脚本编译为每服务的有序版本链。

    版本 v 的生效区间为 (effective_after_minute, 下一版本 effective_after]。
    """
    chains: dict[str, list[VersionedSpec]] = {
        spec.service_id: [
            VersionedSpec(
                spec=spec,
                effective_from=-1,
                effective_to=None,
            )
        ]
        for spec in base_specs
    }
    for edit in edits:
        chain = chains[edit.service_id]
        current = chain[-1]
        if edit.param_version != current.spec.param_version + 1:
            raise ValueError(
                f"参数版本号必须连续递增: service={edit.service_id} "
                f"current={current.spec.param_version} got={edit.param_version}"
            )
        closed = VersionedSpec(
            spec=current.spec,
            effective_from=current.effective_from,
            effective_to=edit.effective_after_minute,
        )
        chain[-1] = closed
        new_spec = current.spec.with_version(edit.param_version, **edit.changes)
        chain.append(
            VersionedSpec(
                spec=new_spec,
                effective_from=edit.effective_after_minute,
                effective_to=None,
            )
        )
    return chains


def spec_at_minute(chain: list[VersionedSpec], minute: int) -> VersionedSpec:
    for versioned in chain:
        if versioned.covers(minute):
            return versioned
    raise ValueError(f"分钟 {minute} 没有生效的参数版本")


def affected_minutes(chain: list[VersionedSpec], minute_horizon: int) -> list[int]:
    """参数演进后需要整体替换的受影响分钟集合（旧版本结论标陈旧的范围）。"""
    affected: set[int] = set()
    for versioned in chain[1:]:
        start = versioned.effective_from + 1
        end = minute_horizon
        affected.update(range(start, end + 1))
    return sorted(affected)
