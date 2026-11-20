"""领域数据类型：通道声明、分片事件、组结局、估计版本、窗口事件等。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChannelSpec:
    """单条重组通道声明，全部为已校验的整数/字面量字段。"""

    channel_id: str
    host: str
    port: int
    payload_limit: int          # 单包负载上限（字节），正整数
    window_ms: int              # 重组窗口宽度（毫秒），正整数
    max_inflight_ms: int        # 允许的最大在途偏移（毫秒），非负整数
    slot_count: int             # 并发窗口槽位数，非负整数
    min_estimate_samples: int   # 偏差估计最小样本数，正整数


@dataclass(frozen=True)
class FragmentEvent:
    """逐片记录。

    send_ms     应用侧标注的发送时刻（毫秒，发送端时钟）
    arrive_ms   接收侧时钟记录的到达时刻（毫秒，唯一可复核时间来源）
    payload     该片负载字节
    """

    channel_id: str
    group_seq: int
    frag_offset: int
    frag_length: int
    total_frags: int
    send_ms: int
    arrive_ms: int
    payload: bytes


@dataclass(frozen=True)
class EstimateVersion:
    """偏差估计的一个版本。

    version        单调递增版本号；0 表示未定（样本不足）
    skew_ms        整数毫秒定点偏差估计：local = send + skew
    sample_count   截至该版本累计样本组数
    qerror_ms      偏差量化误差上界（整数毫秒，保守取 1ms）
    """

    version: int
    skew_ms: int | None
    sample_count: int
    qerror_ms: int


@dataclass(frozen=True)
class WindowEvent:
    """窗口开闭账本事件。"""

    channel_id: str
    window_seq: int
    action: str          # OPEN / CLOSE
    estimate_version: int
    base_ms: int         # 窗口基准（首个落入组的折算时刻）
    close_ms: int | None
    anchor_group: int


@dataclass(frozen=True)
class FlipRecord:
    """偏差版本变化导致同一组窗口内外判定翻转的记录。"""

    channel_id: str
    group_seq: int
    from_version: int
    to_version: int
    from_adjusted_ms: int
    to_adjusted_ms: int
    from_window: int | None      # None 表示落在窗口外
    to_window: int | None
    flip_kind: str              # IN_TO_OUT / OUT_TO_IN / WINDOW_CHANGE


@dataclass(frozen=True)
class GroupOutcome:
    """组结局。

    status:
      COMPLETE       重组成功且摘要一致
      E_DIGEST       摘要与登记值不一致
      SLOT_DROPPED   槽位耗尽被丢弃
      OFFSET_DROPPED 超出最大在途偏移被丢弃
      INCOMPLETE     窗口关闭时仍缺片
      UNKNOWN        强制终止时窗口内未落定，需要重跑
    """

    channel_id: str
    group_seq: int
    status: str
    estimate_version: int
    window_seq: int | None
    adjusted_ms: int | None
    digest: str | None = None
    expected_digest: str | None = None
    reason: str | None = None


@dataclass
class ReassemblyResult:
    """单通道完整裁决结果（纯函数输出，可直接序列化进账本）。"""

    channel_id: str
    estimates: list[EstimateVersion] = field(default_factory=list)
    window_events: list[WindowEvent] = field(default_factory=list)
    flips: list[FlipRecord] = field(default_factory=list)
    outcomes: list[GroupOutcome] = field(default_factory=list)
    fragments: list[FragmentEvent] = field(default_factory=list)
