# 慢查询多源证据诊断（本地可运行版）

基于本地 SQLite 账本与三类异构观测源（库侧执行统计、应用日志摘要、慢日志
摘录）的确定性慢查询诊断工具。支持多版本帧编解码、稳定仲裁、定点误差归因、
受限并行回环回放、方案演进失效传播与强制终止后恢复重放。

## 快速开始

```powershell
python --version              # Windows，需 Python 3.11+
python -m pip install -r requirements.txt
python scripts/generate_samples.py
python -m unittest discover -s tests
python demos/demo_full_pipeline.py
python demos/demo_crash_recovery.py
python demos/demo_error_paths.py
```

CLI 回放：

```powershell
python -m slowquery_diag.cli --declaration examples/declaration.json `
  --ledger workdir/cli_ledger.db --replay --output workdir/cli_report.json
```

可选 Tk 界面（Windows 标准 Python 自带 Tk）：

```powershell
python -m slowquery_diag.gui --declaration examples/declaration.json
```

## 目录导航

- `slowquery_diag/`：核心包（校验、编解码、仲裁、量化、诊断、回放、账本、引擎）
- `scripts/generate_samples.py`：NumPy 固定种子样本生成
- `examples/`：示例声明、二进制帧、回放目录
- `demos/`：完整流水线、崩溃恢复、错误码三类演示
- `tests/`：30 个 unittest 用例
- `docs/`：架构、定点误差界、需求追溯、验证记录

## 文档

- 架构与模块：`docs/ARCHITECTURE.md`
- 定点量化与误差界：`docs/FIXED_POINT_ERROR_BOUNDS.md`
- 需求追溯矩阵：`docs/REQUIREMENTS_TRACEABILITY.md`
- 验证记录与 Windows 说明：`docs/VALIDATION.md`

## 错误码

`E_SOURCE`（声明违规）、`E_FORMAT`（帧/版本/字段违规，定位行号）、
`E_PORT`（回环端口）、`E_TIMEOUT`（真实套接字超时）、`E_ACK`（回执缺字段）、
`E_REPLAY`（恢复重放不一致，定位查询与字段）、`E_LEDGER`（账本兜底）。
