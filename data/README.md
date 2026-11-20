# 本地数据

本目录由以下命令生成，不依赖网络：

```powershell
python -m trace_diag generate-samples --data-dir data --records 600
python -m trace_diag demo
```

- `services.json`：3 个服务的确定性记录声明。
- `policies.json`：2 条候选采样策略。
- `ledger/trace_diag.sqlite`：本地 SQLite 账本，可删除后由 `demo` 或 `analyze` 重建。
- `logs/trace_diag.jsonl`：结构化运行和异常日志。
- `diagnostic_report.json`：完整统计、归因、翻转和重放结果。
