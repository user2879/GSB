"""连续域约束求解与数值误差传播诊断实验环境。

模块划分：
  errors    业务错误码与异常
  util      确定性序列化、内容哈希、确定性浮点比较口径
  spec      YAML 规格解析与校验（双精度数值模型）
  seeding   固定种子的子种子派生
  continuous 连续域候选布局求解
  quantize  连续量 -> 网格图块量化与误差度量
  grid      离散关卡网格
  constraints 约束判定与误差区间
  ledger    误差传播账本与边界报告
  generator 生成入口（写工件）
  recompute 独立复算器（只消费工件）
  incremental 增量重生成与重算闭包
  demo      可重复演示入口
"""

__all__ = ["errors"]
