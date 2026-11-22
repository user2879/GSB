# GPU 配额与分片抢占调度诊断工作台

面向 Windows 本地的确定性 GPU 资源池/任务/分片调度诊断应用：三重配额约束、
分片边界优先级抢占、逐轮定点水位账本、SQLite 强制终止恢复、版本化快照重放，
以及 Tkinter 快速视图与独立权威重算。仅依赖 Python 3.11+ 标准库、NumPy、pytest。

## 快速开始

```bash
python3 -m virtualenv .venv          # Windows: py -3 -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m gpu_diag.cli init-samples
python -m gpu_diag.cli run
python -m gpu_diag.cli recompute
python -m gpu_diag.cli ui
python -m pytest -q
```

Windows 可直接运行 `run_demo.bat`；设计细节、业务码与验证结果见 `DESIGN.md`。

## 业务码

| 码 | 触发场景 |
| --- | --- |
| `E_DECL` | 声明字段非法/标识重复/依赖引用缺失，整体拒绝并列违规 |
| `E_MEMORY` | 显存预算越界或任务显存超过池容量 |
| `E_QUOTA` | 配额份额之和超上限 / 全局 GPU 配额约束 |
| `E_CYCLE` | 分片或任务依赖成环 |
| `E_LEDGER` | 账本内容摘要不符/损坏 |
| `E_REPLAY` | 快照恢复重放与账本首个偏差 |
| `E_PREVIEW` | 界面权威重算与缓存账本首个偏差 |
| `E_STATE` | 强制终止后未提交分片标记为未知 |

## 验收对照

- 三重约束并行调度：`tests/test_scheduler.py`
- 分片边界确定性抢占与恢复复用：`tests/test_scheduler.py::test_preemption_*`
- 水位账本持久化与逐字节权威重算：`tests/test_persistence.py`
- 强制终止未知分片、快照整体替换与 `E_REPLAY`：`tests/test_persistence.py`
- 四类异常注入结构化日志：`tests/test_cli_ui.py`
