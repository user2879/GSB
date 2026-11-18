# 本地数据目录

- `configs/`：由 `scripts/generate_samples.py` 生成的 v1、v2、v3 配置及初始升级检查点。首次运行会原子升级并创建只读 `configs/backups/`。
- `events/`：确定性 NDJSON 事件流。
- `metadata/`：派生关系声明与预期仲裁说明。
- `logs/`：运行后写入 JSONL 结构化日志。
- `ledger.sqlite3`：首次执行 `run`/`edit`/`recover` 时创建；不预置数据库，避免把演示运行误当成用户数据。

交付样本保持为未升级的三个历史版本，便于重复验收自动升级与中断恢复。
