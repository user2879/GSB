"""说话人音色分析可集成组件。

五个确定性阶段：
    segment   音频读取与发音段切分
    feature   段级声学特征汇聚
    quantize  段级音色描述向量逐维自适应量化
    attach    段间归属一致判定
    aggregate 说话人级描述向量加权聚合

所有产物均由内容指纹 + 配置指纹派生缓存键，支持精确增量重算、
首响预算降级、线程池并发、阶段快照整体替换与中断恢复。
"""

__all__ = [
    "errors",
    "utils",
    "config",
    "synthesize",
    "audio",
    "features",
    "quantize",
    "attach",
    "aggregate",
    "cache",
    "pipeline",
    "snapshot",
    "budget",
    "inject",
    "report",
]
