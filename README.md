# obsflow — 本地观测记录分层滑动聚合组件

面向本地 Windows/Linux 环境的可集成 Python 包：归一三种历史记录格式、
按指标键维护定点整数滑动窗口、精确失效传播与仅子集重算、双配额并行调度、
租约生命周期、SQLite 账本快照、强制终止恢复与逐窗口重放校验，并附带
Tkinter 诊断界面、本地 socket 状态端点与异常注入链路。

## 环境准备（Windows 命令提示符）

```
py -m venv .venv
.venv\Scripts\activate
python --version          # 需 3.10+
pip install -r requirements.txt   # numpy>=1.24
```

仅使用标准库 `sqlite3 / tkinter / socket / threading / subprocess` 加 NumPy，
无需任何系统服务或容器，样本由固定整数种子本地生成。

## 快速开始

```
python -m obsflow.cli generate --output-dir samples/local
python -m obsflow.cli run --workdir artifacts/run
python -m obsflow.cli replay --workdir artifacts/run
python -m obsflow.cli ui --workdir artifacts/run
```

异常注入与崩溃恢复：

```
python -m obsflow.cli inject truncate --workdir artifacts/inj   # E_INTEGRITY
python -m obsflow.cli inject order --workdir artifacts/inj      # E_ORDER
python -m obsflow.cli inject snapshot --workdir artifacts/snap  # E_SNAPSHOT
python -m obsflow.cli crash --workdir artifacts/crash
python -m obsflow.cli recover --workdir artifacts/crash --rerun
```

## 包接口

```python
from obsflow import (
    load_source_set, load_engine_config, load_stream,
    ObsFlowEngine, BoundedScheduler, LeaseManager, Ledger,
    FormatError, IntegrityError, OrderError, SnapshotError,
    LeaseError, ReplayError,
)

engine = ObsFlowEngine.from_files("samples/local/sources.json",
                                  "samples/local/window_params.json",
                                  "artifacts/run/obsflow.db")
engine.read_records()                 # 读取/校验/归一记录流
report = engine.advance()             # 聚合并发推进 + 租约 + 账本提交
windows = engine.query_windows()      # 查询窗口聚合
engine.save_snapshot()                # 保存快照（版本+摘要）
engine.recover()                      # 恢复账本（未提交窗口标 unknown）
engine.replay_verify()                # 逐窗口逐字段重放校验
```

## 目录

- `obsflow/` 可集成包：codec、aggregation、propagation、scheduler、
  lease、ledger、engine、diagnostics、injection、process_runner、cli
- `tests/` pytest 套件（27 个用例）
- `examples/run_demo.py` 端到端示例
- `docs/DESIGN.md` 设计说明，`docs/VERIFICATION.md` 实际验证记录

## 验证

```
python -m pytest tests
```

实测 `27 passed`；详细验收矩阵见 `docs/VERIFICATION.md`。
