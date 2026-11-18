# 推荐排序模型离线评测与 A/B 实验归因工作台

本地、确定性、可复现的离线评测工作台。仅依赖 Python 3.11 标准库与 NumPy，无需容器、
外部账号或在线服务。

## 快速开始

```bash
python3 -m venv --without-pip .venv311
.venv311/bin/python /tmp/get-pip.py
.venv311/bin/python -m pip install numpy

.venv311/bin/python scripts/run_demo.py                       # 生成样本 + 并发/串行演示
.venv311/bin/python -m unittest discover -s tests             # 25 个测试
.venv311/bin/python scripts/equivalence_check.py              # 增量 vs 全量等价矩阵
```

## 能力一览

- 确定性分桶（BLAKE2b 哈希 + 权重连续区间）与严格日志校验。
- CTR/CVR 分组统计，定点整数口径，请求 ID→物品 ID 固定累加序。
- 绝对/相对提升与固定方差式置信区间，最小样本量边界显式标注，胜负判定。
- 多属性键分层归因、最大余数权重配平、漂移检测与确定性逐层修正。
- 内容指纹缓存：精确到分组的失效与复用，增量与全量逐字段等价。
- 阶段原子快照、SIGTERM 优雅退出、强制终止按完整快照恢复。
- 四类真实故障注入与业务错误码命中复现。

## 文档

- `docs/architecture.md`：架构、确定性口径、缓存键与错误码。
- `docs/runbook.md`：环境准备、运行命令与实际验证记录。
- `docs/requirements_traceability.md`：八条需求的实现与验证对照。
