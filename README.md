# 离线特征流水线缓存命中诊断组件

本项目实现分层特征处理链（原始样本 → 标准化 → 分桶/交互乘积 → 训练视图）的确定性定点计算、三版缓存回执归一、内容摘要失效判定、预算调度、原子快照与故障注入报告。

## 环境准备

要求 Python 3.11 或 3.12：

```bash
python --version
python -m pip install -r requirements.txt
```

运行时核心包只依赖标准库；`numpy` 与 `pytest` 用于开发测试依赖约定。

## 一键复现

```bash
python scripts/generate_fixtures.py
python -m pytest -q
python -m fcdiag.cli run --output output/run-report.json
python -m fcdiag.cli incremental-check --output output/incremental-report.json
python -m fcdiag.cli injections --output output/injection-report.json
```

`scripts/generate_fixtures.py` 使用固定整数种子生成：

- `samples/samples.jsonl`：确定性样本；
- `samples/pipeline.yaml`：分层流水线定义；
- `samples/config.json`：定点、预算与降级策略；
- `samples/commands.jsonl`：样本、参数、量化位宽、档位和回执版本编辑流；
- `samples/receipts/v1.txt`、`v2.txt`、`v3.txt`：三版定宽回执；
- `samples/injections.json`：五类独立故障注入清单。

## 核心模块

- `fcdiag/loader.py`：JSONL/YAML/配置/命令流加载与全量违规收集；
- `fcdiag/receipts.py`：v1/v2/v3 定宽解析、归一、双向转换与旁路元数据；
- `fcdiag/pipeline.py`：定点整数域确定性特征计算与内容摘要；
- `fcdiag/invalidation.py`：摘要、依赖图和降级保守策略；
- `fcdiag/scheduler.py`：批次预算、阶段配额、稳定排序键与并发隔离；
- `fcdiag/snapshot.py`：临时目录写入、整体替换和版本恢复判定；
- `fcdiag/runner.py`：端到端账本、事件、缓存键和快照编排；
- `fcdiag/experiment.py`：增量/全量一致性与故障注入实验。

详细设计见 `docs/TECHNICAL_NOTES.md`。
