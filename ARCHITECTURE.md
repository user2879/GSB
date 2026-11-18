# 架构与关键设计

## 1. 分层与数据流
```
JSON 规则文件
  -> ruleset.load_rules        字段强校验(E_RULE) + 引用校验 + 成环检测(E_DEPEND)
  -> graph.DependencyGraph     稳定拓扑 / 下游传递闭包 / 传播轨迹
  -> changeset.apply_changeset 原子变更 + 失效集合（E_REF/E_RULE/E_DEPEND）
  -> compiler.PlanCompiler     规范化产物 + 计划整体替换 + 单调版本号
  -> ledger.Ledger             SQLite 持久化（规则集/图/计划/失效/观测/快照/pending）
  -> shaper / traffic_sim      虚拟时钟与真实回环 TCP 两种整形验证
  -> snapshot / replay         版本识别、整体替换、重放比对
```

## 2. 校验与整体拒绝
- `models.validate_rule_dicts` 收集全部字段级违规，返回 `FieldViolation(rule_id, field_path, reason)`；
  只要存在任意违规，`ruleset.load_rules` 即抛 `E_RULE`，不产生部分加载。
- `bool` 被显式排除在整数之外（`isinstance(True, int)` 为 True 的陷阱）。
- 引用存在性与自引用在 `DependencyGraph.validate_references` 中以字段路径 `upstream` 报告；
  成环由 Kahn 算法检测，剩余节点即环上节点，抛 `E_DEPEND`。

## 3. 失效传播
- 边语义为 `upstream -> downstream`；`transitive_downstream` 求传递闭包。
- 失效序列按全图稳定拓扑序过滤得到，保证下游规则一定在其上游之后失效。
- 传播轨迹 `TraceEntry(step, rule_id, triggered_by, via_edge)` 通过 BFS 记录直接父来源；
  删除场景在旧图上 BFS，因此被删规则的间接下游也能标注 `delete:<id>` 触发源与 `->>` 边。
- 删除前置检查 `E_REF`：只要仍有 downstream 引用即拒绝，并列出每个待删规则的引用者。

## 4. 编译、版本与整体替换
- 单规则产物 `build_artifact` 是纯数据规范化（排序后的端口前缀、上游、动作），不含全局版本号。
- 计划优先级键：`(最长前缀长度降序, 端口前缀元组, 地址, 规则标识)`，完全确定。
- 计划摘要 `plan_hash = SHA-256(canonical(entries + artifact_hashes))`。
- 计划版本号由账本 `MAX(plan_version)+1` 单调供给，编译器再校验严格递增；
  每次版本变化都用新 `Plan` 对象整体替换当前计划，不存在逐条修补路径。
- 编译器缓存（规则指纹、产物、产物哈希、产物版本号）持久化在 `compiler_cache` 表，
  因此跨 CLI 进程与崩溃重启后，未失效规则仍逐字节复用、产物版本号保持不变。

## 5. 增量=全量一致性
- `assert_incremental_matches_full` 用无缓存 `full_recompile` 构造对照基准，
  按规则标识字典序逐条比较产物哈希，首个差异即抛 `E_INCR`，并定位到具体不一致字段。
- 复用判定：规则不在失效集合且规则指纹（规范化规则体哈希）未变。

## 6. 整形模型
- 令牌桶：速率 `rate_kbps*1000/8` 字节/秒，容量 `burst_bytes`，按经过时间补充令牌。
- 令牌不足时按缺口向上取整计算排队延迟；`queue_limit_packets>0` 且排队深度超限时丢包。
- 匹配采用最长端口前缀，同长按规则标识决胜。
- `verify_throughput` 以平均吞吐不超过 `committed*(1+tolerance)` 判定，
  突发容量体现在首包可线速发送 `burst_bytes`；违规抛 `E_SHAPE` 并附承诺/实际整数对比。
- 回环帧：`magic(u32) seq(u32) dst_port(u32) payload_len(u32) + payload`，
  负载为序号的确定性大端重复字节。`run` 子命令以就绪文件编排 sink/shaper 子进程，
  避免健康探针抢占单次 `accept`。

## 7. 账本与强制终止恢复
- 业务写入先登记 `pending_operations(status=OPEN)`，完整提交后置 `COMMITTED`。
- 重启 `recover()` 只依据 `COMMITTED` 的计划/图记录恢复版本；
  所有仍为 OPEN 的操作，其 scope 规则并入 UNKNOWN 集合，要求重新编译或重跑。
- 系统明确不生成强制终止时刻的“退出报告”，恢复报告只列未知规则、OPEN 操作与账本版本。
- SQLite 启用 `WAL` 与 `synchronous=FULL`，提升崩溃后的提交一致性。

## 8. 快照与重放
- 快照内容含规则集版本、依赖图版本、计划版本、计划/产物摘要、失效序列与轨迹、观测摘要、变化清单。
- 恢复时先比较规则集版本与依赖图版本：不一致即 `full-replace`，整体替换规则集、图与计划状态，
  并稳定输出规则级 `added/modified/deleted`；禁止增量合并。
- 重放按快照规则集与变化清单重建变更集，重新推导失效集合并全量编译，
  逐项比较失效序列、轨迹、计划摘要与每规则产物哈希，首个差异抛 `E_REPLAY` 并给出依据。

## 9. 异常注入
注入只改变输入文件或参数，不改业务代码路径：
- `--fault corrupt` 选择 JSON 损坏文件 -> `E_RULE`
- `--fault cycle` 选择成环样本 -> `E_DEPEND`
- `--fault snapshot` 强制版本识别读取缺字段快照 -> `E_SNAP`
所有注入结果写入 `logs/*.jsonl` 结构化日志。

## 10. 实际验证结果（本机）
- `python -m unittest discover -s tests`：28 个测试全部通过。
- `python demo.py`：14 项端到端检查全部 PASS，报告写入 `data/diagnostic_report.json`。
- 真实回环 TCP：24 包全部由 sink 接收（字节守恒）；压力场景 30 个 4096 字节包全部命中 R8，
  最大整形延迟约 409ms，令牌桶限速生效。
- 确定性：同规则集重复编译计划逐字节一致；打乱输入规则顺序后计划摘要不变。
- 环境限制：本机 Linux 无 Tkinter 且无 root 安装，GUI 已验证可优雅降级；
  Windows 官方 Python 自带 Tkinter，可直接 `WorkbenchGUI().run()`。
