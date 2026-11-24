"""回执归并诊断组件（本地确定性推理实验账本）。

模块划分：
  spec      本地 JSON 声明加载与强校验（E_SPEC）
  codec     r1/r2/r3 回执编解码与归一
  deterministic 确定性参数与样本构造
  worker    回环子进程工作器
  scheduler 双重并发配额调度
  ledger    SQLite 权威账本
  preview   快速视图 / 权威重算（E_PREVIEW）与定点延迟统计
  cli       命令行入口
"""

__version__ = "1.0.0"
LEDGER_FORMAT_VERSION = 1

