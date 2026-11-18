# 默认运行目录

运行 UI/CLI 时，默认生成：

- `session.db`：SQLite 会话、当前代、审计账本与快照注册表
- `snapshots/`：不可覆盖的版本化快照 JSON
- `logs/drill.jsonl`：结构化运行日志

演示脚本使用 `artifacts/demo-data/`，不在此目录生成固定数据。
