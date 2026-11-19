# 本地确定性关系与事件抽取编辑应用（rexapp）

面向本地确定性样本的关系/事件抽取数据编辑应用，核心解决交互延迟与计算结果一致性的冲突：每条交互命令在延迟预算内返回最新可用结果或显式 pending 中间态，多次交互后的增量结果与冷启动全量执行逐字段一致。

## 快速开始

```bash
python3.10 -m venv --without-pip .venv         # 系统 ensurepip 缺失时的引导方式
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install -r requirements.txt
./run.sh                                        # 生成样本 + 全量 + 交互 + 验收 + 单测
```

单独运行：

```bash
export PYTHONPATH=src
.venv/bin/python scripts/generate_samples.py
.venv/bin/python -m rexapp full samples output
.venv/bin/python -m rexapp interactive samples output
.venv/bin/python scripts/acceptance_check.py
.venv/bin/python -m unittest discover -s tests
```

## 输入与输出

- `samples/documents.jsonl`：文档 ID、空格分隔词序列与词位置（含 4 条显式非法样本）。
- `samples/config.json`：实体/关系/事件规则、双路置信度阈值、定点位宽、预算、并发数、快照间隔、窗口。
- `samples/commands.json`：触发词修改、文档替换/删除、事件阈值与关系窗口调整共 7 条命令。
- `output/full_results.json` / `incremental_results.json`：全量与增量抽取结果。
- `output/interaction_responses.json`：每条命令的预算内响应（latest/pending 标记）。
- `output/diagnostic_report.json`：分档分布、边界样本与漂移、复用/失效审计、耗时。
- `output/acceptance.json`：逐命令增量 vs 冷启动全量一致性验收。
- `output/snapshots/snapshot.json`：临时文件写盘 + 原子替换的阶段快照。

## 模块与设计

详见 `docs_notes.md`。核心模块位于 `src/rexapp/`：加载校验（`config.py`、`documents.py`）、确定性抽取（`extraction.py`）、定点判定（`fixedpoint.py`）、指纹缓存（`fingerprint.py`、`cache.py`）、并发与预算引擎（`engine.py`）、原子快照（`snapshots.py`）、命令与编排（`commands.py`、`app.py`）。

## 验证结果

- 23 个 unittest 全部通过：业务码、定点边界、确定性、增量/全量一致、并发顺序、预算 pending、快照版本、SIGTERM 退出码 0。
- `acceptance_check.py`：7 条命令全部 `results_match=true`、`failures_match=true`。
- 重复生成样本与全量执行逐字节一致（固定种子 20261120）。
