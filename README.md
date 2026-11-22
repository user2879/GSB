# mlimgedit：本地确定性多标签图像分类数据编辑应用

面向程序生成 PNG 合成图像的多标签分类数据编辑与操作应用，核心解决标签集合、
分类阈值、训练样本反复编辑时的增量更新与精确依赖失效判定。纯 Python 实现，
仅依赖 NumPy 与 Pillow，不使用深度学习框架、GPU 或容器。

## 环境要求

- Python 3.10+
- NumPy、Pillow（运行）；pytest（测试）

## 快速开始

```bash
# 1. 生成确定性合成样本、配置、增量编辑计划与故障注入配置
python3 scripts/bootstrap_demo.py demo

# 2. 单次全量运行
python3 -m mlimgedit.cli run \
  --config demo/config.json \
  --samples demo/samples \
  --store tmp/store \
  --report tmp/report.json \
  --injections demo/injections.json

# 3. 回放增量编辑计划
python3 -m mlimgedit.cli plan \
  --plan demo/edit_plan.json \
  --workdir tmp/work \
  --store tmp/inc_store \
  --reports-dir tmp/reports \
  --injections demo/injections.json

# 4. 增量最终状态 vs 空白工作区全量重建逐字段比对
python3 -m mlimgedit.cli verify \
  --plan demo/edit_plan.json \
  --workdir tmp/work \
  --store tmp/inc_store \
  --reports-dir tmp/reports \
  --injections demo/injections.json

# 5. 查看快照清单
python3 -m mlimgedit.cli inspect --store tmp/inc_store
```

## 一键验收

```bash
python3 acceptance/run_acceptance.py
```

覆盖：增量/全量一致、预算待办与快照恢复、并发/串行一致、异常注入隔离。
结果写入 `acceptance/runs/acceptance/acceptance_summary.json`。

## 测试

```bash
python3 -m pytest tests -q
```

## 目录

- `mlimgedit/`：管线、分类器、缓存、预算、快照、并发与 CLI
- `scripts/bootstrap_demo.py`：确定性演示数据生成
- `tests/`：41 个单元与集成测试
- `acceptance/run_acceptance.py`：四类验收场景驱动
- `docs/DESIGN.md`：架构、缓存键、失效规则与验证结果说明

## 业务码

- `E_SAMPLE_FORMAT`：PNG 容器/解码或标注结构故障
- `E_LABEL_UNKNOWN`：标注引用配置外标签
- `E_BBOX_OUT_OF_RANGE`：整数边界框越界
- `E_FEATURE_BUDGET_PENDING`：单批特征点预算不足，任务保留待办
- `E_STAGE`：阶段内非输入类异常，单样本隔离不中断整体运行
