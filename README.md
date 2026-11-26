# GSB：中文客服工单否定作用域账本

本仓库在空目录上自建了确定性中文客服工单样本、加载校验、否定作用域求解、定点抵消/惩罚/加权、float64 误差追踪、内容指纹增量账本、预算并发调度和阶段化快照恢复。

## 快速开始

机器上的系统 `python3 -m venv` 因 Ubuntu 缺少 `ensurepip/python3.11-venv` 不可用；实际验证使用本机已安装的 `virtualenv` 创建了 `.venv`。标准 WSL Ubuntu 若已安装 venv 组件，可直接用 `python3 -m venv .venv`。

```bash
virtualenv .venv                 # 或 python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt  # 等价于任务要求的 pip install numpy

python -m compileall -q main.py gsb tests
python -m unittest discover -s tests -v
python -m gsb.cli verify
python -m gsb.cli apply-edits
```

也可以运行一键检查：

```bash
bash scripts/run_checks.sh
```

## 常用 CLI

```bash
python -m gsb.cli generate-sample --dir samples
python -m gsb.cli validate --sample-dir samples
python -m gsb.cli run --sample-dir samples --budget 1000 --workers 4
python -m gsb.cli verify
python -m gsb.cli resume-snapshot --snapshot-dir artifacts/snapshot-demo
```

校验失败时输出稳定错误码 `E_SCHEMA_VIOLATION` 和逐字段 `issues`；快照不可用时输出 `E_SNAPSHOT_UNAVAILABLE`，不会伪造恢复结论。

## 目录

- `gsb/schema.py`：样本与配置加载、聚合 Schema 校验。
- `gsb/samplegen.py`：固定种子 `383` 的语料、词表、权重、配置和编辑流生成器。
- `gsb/scanner.py`：字面命中、作用域生成、唯一生效作用域和压制依据。
- `gsb/settlement.py`：定点抵消、饱和轨迹、隔离和长度惩罚。
- `gsb/composition.py`：渠道加权、极性/风险阈值、半步边界标记和同分排序。
- `gsb/reference.py`：NumPy float64 参考路径、四环节差值和翻转归因。
- `gsb/cache.py`、`gsb/incremental.py`：内容指纹键、精确失效和复用账本。
- `gsb/scheduler.py`：作用域点数预算、pending 队列和有限并发。
- `gsb/snapshot.py`：临时目录、原子 symlink 提交和已提交快照恢复。
- `tests/test_gsb.py`：20 个聚焦测试、真实 SIGKILL 恢复测试和端到端验收。
- `docs/DESIGN.md`：规则、公式、失效依赖和一致性设计。
- `artifacts/validation-results.txt`：实际 compileall、unittest、verify 输出。
- `artifacts/acceptance-report.json`：结构化验收证据。

## 本地样本

`samples/` 已生成：

- 18 张固定种子工单；
- 64 个受控词项；
- 5 个否定触发词；
- 5 个极性冲突词；
- 4 个渠道权重；
- 6 条编辑命令流。

刻意覆盖：

- `TK-0001`：嵌套作用域和 2 条压制依据；
- `TK-0015`：交叠作用域和 1 条压制依据；
- `TK-0005`：冲突抵消连续饱和，零分命中保留轨迹；
- `TK-0013`：极性阈值量化半步边界；
- `TK-0002`：风险阈值边界，Q6/Q2 档位产生可归因的阈值翻转。

## 最新验证摘要

实际结果见 `artifacts/validation-results.txt`：

- `python -m compileall -q gsb tests`：PASS。
- `python -m unittest discover -s tests -v`：20/20 通过，包含任意预算断点续跑和子进程 SIGKILL 后恢复。
- `python -m gsb.cli verify`：7 组验收检查全部通过。
- Q2 粗档位首次超容差计数：抵消 6 单、结论合成 3 单。
- 增量验收：替换 `TK-0004` 仅失效该单；修改 `NEG_BIE` 仅失效 `TK-0003`；未受影响 17 单均复用。
- 预算并发：`workers=1/2/4/8`、`budget=24/1000` 下结论与规范账本均与串行逐字段一致。
- 快照：`hit_scan` 和 `composition` 提交点恢复均与空白重建 JSON 一致；损坏/缺失快照明确报 `E_SNAPSHOT_UNAVAILABLE`。

三类风险翻转来源在验收报告中各有 1 个真实样本：

```json
{"scope_attribution": 1, "cancellation_saturation": 1, "threshold_comparison": 1, "total": 3}
```
