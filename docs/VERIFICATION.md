# 实际验证结果

验证环境：Ubuntu 22.04 容器，Python 3.11.0rc1，NumPy 1.26.4（目标环境为
Windows 10/11 + Python 3.11，代码仅用标准库与 NumPy，无平台专有调用）。
当前容器无 tkinter，GUI 模块按设计降级，测试不依赖显示。

## 自动化测试

`python3 -m unittest discover -s tests` 共 34 个用例全部通过，且以
`-W error::ResourceWarning` 运行无资源泄漏：

| 测试模块 | 用例数 | 覆盖点 |
| --- | --- | --- |
| `test_fixedpoint.py` | 4 | 向零截断、int64 饱和、定点乘、延迟统计/P95/不剔除 |
| `test_manifest.py` | 10 | E_MANIFEST 各违规分支、无环、端口/版本/键/类型校验 |
| `test_compiler.py` | 5 | 未变产物逐字节不变、派生失效闭包、稳定拓扑、聚合确定性 |
| `test_protocol_errors.py` | 4 | E_PORT/E_HANDSHAKE/E_FIELD、失败节点不计入成功 |
| `test_retry.py` | 1 | 首次连接 RST → 重试成功，尝试序号与语义不变 |
| `test_integration.py` | 5 | 真实子进程发布、增量下发、双路径一致、E_PREVIEW、E_LATENCY、重开恢复 |
| `test_recovery.py` | 5 | 摘要链篡改/格式版本整体替换与只读备份、E_REPLAY、强制 kill 恢复 |

## 端到端 CLI（实测）

`python -m configbench.cli --ledger workspace/bench.db run`：

- v1：直接变化 base-rate/composite/threshold，3/3 节点成功。
- v2：仅 base-rate 变化，受影响 node-alpha、node-beta，2/2 成功，
  node-gamma 保留 v1 不产生新任务（增量下发）。
- v3：composite/threshold 变化（含派生系数与目标集合），3/3 成功。
- 三个版本 `compare_paths` 全部 `match=true`（聚合摘要逐字节一致，
  耗时误差界 0ms）。
- 延迟事件 10 条全部入帐，缩放因子 1000、向零截断，无违规（预算 500ms）。

## 逐字节确定性（实测）

对每个 (版本, 节点) 连续聚合 3 次，摘要 SHA256 完全相同，结果存于
`workspace/determinism.json`。例如 v1/node-alpha 摘要 digest 前缀
`dd40759e8c70df0d`，规范化字节长度 1147。

## 强制终止恢复（实测）

`python scripts/demo_crash_recovery.py` 输出：

1. v1 发布并完整入帐；
2. 三个节点子进程被 `kill()` 强制终止，账本直接关闭，无退出报告；
3. 新进程 `recover()` 仅依据已提交事件恢复 3 个节点，`unknown_tasks=[]`，
   重放 3 个已完成任务全部匹配；
4. 重启节点后发布 v2（2/2 成功），双路径一致性 `match=true`。

## 账本损坏处理（实测）

- 篡改已提交事件 payload → 链式 digest 校验失败 → 旧库复制为
  `bench.db.bak-<时间戳>` 只读备份并整体替换，新库事件为空（不增量合并）。
- 改写 meta 中 `format_version` → 同样整体替换。
