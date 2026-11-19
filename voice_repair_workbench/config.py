"""全局固定常量：STFT 规格、版本戳与通道/位深策略。

STFT 窗长与帧移是谱减实现的契约，任何改动都会改变修复结果，
因此被固化进引擎版本戳 ENGINE_VERSION，作为缓存失效与快照新旧判定的依据。
"""
from __future__ import annotations

# ---- 音频规格 ----
SAMPLE_RATE = 48_000
SUPPORTED_BIT_DEPTHS = (16, 24)
SUPPORTED_CHANNELS = (1, 2)

# ---- 谱减 STFT 契约（固化，禁止随工程文件改动）----
STFT_WINDOW = "hann"
STFT_NFFT = 2048  # 窗长（样本）
STFT_HOP = 512  # 帧移（样本）
STFT_ENGINE_REVISION = 1

# ---- schema ----
SCHEMA_V1 = 1
SCHEMA_V2 = 2
CURRENT_SCHEMA = SCHEMA_V2

# ---- 引擎版本戳：算法 + STFT 契约 + 阶段定义 ----
# 任何会改变修复输出的实现改动都必须提升此版本戳。
ENGINE_VERSION = (
    f"engine-v{STFT_ENGINE_REVISION}"
    f"-stft{STFT_NFFT}x{STFT_HOP}-{STFT_WINDOW}"
    f"-pipeline4(denoise,gate,normalize,limit)"
)

# ---- 通道处理顺序（v2 字段 channel_order）----
ORDER_MONO_FIRST = "mono_first"  # 先混合为单声道，再跑全部阶段
ORDER_PER_CHANNEL = "per_channel"  # 降噪/门限按声道处理，再混合，再归一/限幅
CHANNEL_ORDERS = (ORDER_MONO_FIRST, ORDER_PER_CHANNEL)

# ---- 位深策略（v2 字段 bit_depth_policy.output_bit_depth）----
BIT_DEPTH_POLICIES = ("preserve", "force16", "force24")
DEFAULT_OUTPUT_BIT_DEPTH = 16

# ---- 本地有损格式标记 ----
LOSSY_MARKER = "LOSSY-LRX1"
LOSSY_EXTENSION = ".lrx"
LOSSY_FORMAT_NAME = "LRX1"  # Local deterministic codec, revision 1

# ---- 持久化文件名 ----
PROJECT_FILE = "project.json"
CACHE_DIR = "cache"
SNAPSHOT_FILE = "snapshot.json"
CLOSED_MARKER = "CLOSED.flag"
EXIT_REPORT_FILE = "exit_report.json"
ORPHAN_LOG_FILE = "orphans.log"
