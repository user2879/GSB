"""隧道重组窗口裁决领域组件。

模块划分：
  errors    业务错误码与异常
  model     通道、分片事件、组结局等领域数据类型
  config    通道声明与发送脚本 JSON 的强校验加载
  protocol  UDP 数据报片头编解码（真实回环套接字）
  clock     单调时钟与确定性时钟抽象
  engine    偏差估计、窗口判定、翻转、槽位/偏移裁决、重组摘要
  ledger    sqlite3 账本与会话快照（强制终止后恢复）
  replay    恢复后一致性重放比对
  faults    异常注入（声明损坏/样本不足/快照摘要不符）
  logutil   结构化 JSONL 日志
  ui        Tkinter 诊断界面（无 tkinter 时守护式降级）
"""

from .errors import (
    BusinessError,
    E_CHANNEL,
    E_ESTIMATE,
    E_DIGEST,
    E_SNAPSHOT,
    E_REPLAY,
    E_PROTOCOL,
)
from .model import ChannelSpec, FragmentEvent, GroupOutcome

__all__ = [
    "BusinessError",
    "E_CHANNEL",
    "E_ESTIMATE",
    "E_DIGEST",
    "E_SNAPSHOT",
    "E_REPLAY",
    "E_PROTOCOL",
    "ChannelSpec",
    "FragmentEvent",
    "GroupOutcome",
]

__version__ = "1.0.0"
