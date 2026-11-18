"""全局常量：版本号、标志位、定点比率与成因标签。"""

from __future__ import annotations

MODEL_VERSION = "3.0.0"
ARBITRATION_RULE_VERSION = "1.0.0"
SUPPORTED_EVENT_VERSIONS = ("v1", "v2", "v3")

# 标志位（单字节位掩码）
FLAG_SYN = 0x01
FLAG_ACK = 0x02
FLAG_FIN = 0x04
FLAG_RST = 0x08
FLAG_RETX = 0x10
FLAG_RESERVED_MASK = 0xE0

# 线格式
WIRE_MAGIC = b"NDF1"
HEADER_SIZE = 36
TRAILER_SIZE = 4
MAX_PAYLOAD = 1400
MAX_FRAME_SIZE = HEADER_SIZE + MAX_PAYLOAD + TRAILER_SIZE

# 成因标签
CAUSE_TRUE_LOSS = "true_loss"
CAUSE_RETRANSMIT = "retransmit_fill"
CAUSE_OOO = "ooo_late"
CAUSE_PENDING = "undetermined"

# 仲裁规则
ARBITRATION_WIRE_PREFERRED = "wire_preferred"
ARBITRATION_EVENT_PREFERRED = "event_preferred"

# 定点比率：比率 = round(1_000_000 * lost / expected)，单位 BPS（基点的百分之一）
RATE_SCALE = 1_000_000
RATE_UNIT = "BPS"  # 1 BPS = 1e-6
RATE_ROUNDING = "half-up"

# 来源
SOURCE_WIRE = "wire"
SOURCE_EVENT = "event"

# 默认会话参数
DEFAULT_WINDOW_MS = 100
DEFAULT_MIN_SAMPLES = 4
