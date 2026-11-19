# 实际验证记录

本目录保留可复核的本机执行结果。

- `pytest.txt`：21 个测试全部通过。
- `main_validation.json`：8 条主编辑命令的增量/冷全量逐状态比较结果。
- `budget_validation.json`：195/234 点预算序列、待匹配队列和快照恢复结果。
- `final_snapshot.json`：主序列最终全局版本 5 的整文档快照。
- `initial_incremental_report.json`：初始增量运行完整报告，含逐行/逐字符、切分记录、候选得分、预算、缓存和并发诊断。

主序列最终结果：`passed=true`，`first_overall_divergence=null`，快照版本 5，4 个阶段、8 个阶段摘要，缓存审计 22 个对象无缺失，关闭回读恢复语义一致，4 线程与单线程语义一致。

预算序列最终结果：`passed=true`，快照版本 3，初始 195 点消费完成 5 个字符并留下 20 个 `E_POINT_BUDGET_PENDING`；提升到 234 点后复用 195 点、重算 1 个字符，并保留 19 个待匹配字符。
