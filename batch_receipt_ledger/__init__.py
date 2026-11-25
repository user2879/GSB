"""本地批推理调度与多版本回执系统（演示版）。

模块划分：
  errors    统一错误码（E_SPEC/E_FIELD/E_REF/E_PORT/E_TIMEOUT/E_LEASE ...）
  clock     单调时钟封装（毫秒）
  spec      声明/请求流/配额事件 JSON 强校验与归并
  receipts  r1/r2/r3 回执编解码与归一
  inference 整数定点 NumPy 前向计算（确定性）
  plan      确定性批切分计划与稳定序列化
  protocol  JSON 行套接字协议
  worker    工作进程子进程
  leases    租约签发/续约/真实过期与配额回收
  scheduler 双重上限并行批调度（线程池）
  invalidation 失效传播与失效子集重算
  ledger    SQLite 原子账本、快照替换与崩溃恢复
"""

__version__ = "1.0.0"
