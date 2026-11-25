# 预期快照目录（占位）

运行 `python -m arbiter.cli snapshot --out decls/snapshots/expected/snapshot.json`
后在此生成带 `format_version` 与 `content_digest` 的会话快照。
整体替换恢复只接受完整快照；摘要或版本不一致时拒绝增量合并。
