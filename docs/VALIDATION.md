# 本地验证记录与 Windows 运行说明

## 1. 环境准备（Windows）

```powershell
python --version          # 需 3.11+
python -m pip install numpy
```

SQLite、Tkinter、socket、subprocess 均为标准库；无需 Docker、账号或网络。
本仓库在 Linux + Python 3.11.0rc1 + NumPy 1.26.4 下完成同样验证（该沙箱
缺少 libtk，GUI 导入被惰性保护，Windows 标准 Python 自带 Tk 可直接打开）。

## 2. 一键复现步骤

```powershell
# 1) 生成确定性样本（固定随机种子 20261125）
python scripts/generate_samples.py

# 2) 全量单元测试（30 个用例）
python -m unittest discover -s tests -v

# 3) 三个演示
python demos/demo_full_pipeline.py
python demos/demo_crash_recovery.py
python demos/demo_error_paths.py

# 4) CLI（含真实回环回放）
python -m slowquery_diag.cli `
  --declaration examples/declaration.json `
  --ledger workdir/cli_ledger.db `
  --replay --global-concurrency 2 --per-source-quota 1 `
  --socket-timeout-ms 2000 --output workdir/cli_report.json

# 5) 可选 Tk 界面
python -m slowquery_diag.gui --declaration examples/declaration.json
```

## 3. 实测结果（本环境）

- `python -m unittest discover -s tests`：**30/30 通过**，原始输出保存在
  `docs/validation/unittest_output.txt`。
- `demo_full_pipeline.py`：6 来源、3 查询、8 个 (候选,来源) 回放全部成功；
  全局在途峰值 2、每来源峰值 1，配额排队被单调时钟记录。
- `demo_crash_recovery.py`：模拟硬杀后意图行保持 `unknown`，重开仅重跑该
  候选，全部已提交回执逐字段比对通过（未触发 E_REPLAY）。
- `demo_error_paths.py`：E_SOURCE、E_FORMAT、E_PORT、E_TIMEOUT、E_ACK、
  E_REPLAY 全部按预期抛出，且洗牌来源顺序后报告逐字节一致。
- CLI 回放报告 `workdir/cli_report.json`：8 条回放 `ok=true`。

## 4. 关键设计约束的自证方式

- **不用人为延迟模拟配额**：调度器只用有界线程池与 `Condition` 阻塞；
  服务端唯一 sleep 是 `stall_ms` 专用超时测试条目。
- **硬杀一致性**：意图与回执分属两个事务，未提交事务在回滚日志模式下
  整体丢弃，已提交内容由全表 SHA-256 摘要在重开时同口径校验。
- **顺序确定性**：所有集合输出前显式排序，JSON 固定 `sort_keys`，
  测试对正序/逆序输入做字节级断言。
