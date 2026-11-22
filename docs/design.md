# 设计说明：模型服务弹性伸缩决策工作台

## 1. 范围与依赖
- 仅依赖 Python 3.10+ 标准库与 NumPy；Tkinter 随 python.org Windows 安装包分发。
- 无外部服务、无网络、无容器；SQLite 为本地文件，日志为 JSON Lines。
- 所有序列化走 `canonical_bytes`（`sort_keys=True`、固定分隔符），保证逐字节可比。

## 2. 模块分层
| 模块 | 职责 |
| --- | --- |
| `workbench/loader.py` | JSON 强校验，失败统一 `E_SCHEMA`，定位首个出错服务与字段 |
| `workbench/models.py` | 不可变领域模型、规范化 JSON、SHA-256 摘要 |
| `workbench/estimator.py` | 整数副本规则、全量估算器、NumPy 环形缓冲增量估算器、`E_DIVERGE` |
| `workbench/allocator.py` | 统一裁剪提交、配额/预算约束、拒绝量、稳定排序 |
| `workbench/preview.py` | 单次读取的增量预览缓存、`E_PREVIEW` 比对（不改账本） |
| `workbench/latency.py` | 单调时钟采样、定点均值/P95/最大值、`E_LATENCY` |
| `workbench/ledger.py` | SQLite 事务、版本与陈旧标记、快照摘要、恢复游标 |
| `workbench/engine.py` | 版本演进、分钟推进、权威重放 `E_REPLAY`、崩溃恢复 |
| `workbench/faults.py` | 三类确定性异常注入 |
| `workbench/gui.py` | Tkinter 五标签页 + 无头可测视图模型 |
| `workbench/cli.py` | 命令行入口与稳定退出码 |

## 3. 整数估算规则（rule_version=`replica-int-v1`）
- 单副本每分钟承载量 `capacity = concurrency_per_replica * 60000 // latency_ms`；
  `latency_ms=0` 时定义为 `concurrency * 60000`（零成本即时完成）。
- 未夹取需求 `raw_demand = ceil(requests / capacity)`，零请求为 0，零容量时取服务上限。
- 夹取需求 `demand = clamp(raw_demand, min_replicas, max_replicas)`。
- 利用率为基点整数 `requests*10000 // (demand*capacity)`，配合扩缩容阈值给出
  `scale_up/scale_down/hold/saturate` 标记。
- 全量与增量共用同一个纯函数，天然消除算法分歧；增量器额外维护 NumPy
  定长环形窗口的滚动和与定点均值，仅作诊断量，不参与需求结论。

## 4. 路径选择与一致性
- 显式边界 `incremental_threshold_minutes`：`minutes > boundary` 选增量，否则全量；
  边界分钟本身留在全量侧，选择理由与边界值写入每分钟决策。
- 每条分钟在提交前同时跑两条路径并逐标量比对，首个分歧以 `E_DIVERGE`
  报告分钟、服务、字段。

## 5. 统一裁剪
- 有效上限 `effective_cap = min(max_replicas, concurrency_quota)`。
- 稳定全序：需求降序、服务标识升序决胜；按序消费全局预算，一次性提交。
- 裁剪原因优先级：`capacity_overflow > concurrency_quota > min_floor > none`，
  预算不足覆盖为 `global_budget`。
- 拒绝量 `max(0, requests - allocated*capacity)`，恒非负。

## 6. 预览/权威分离与延迟
- 预览只读内存缓存（一次读取），权威重放走账本重放；差异以 `E_PREVIEW`
  稳定列出分钟/服务/字段，且故障注入不会回写账本。
- 时钟基准 `local.monotonic_ns//1_000_000`，真值来源为账本 `latency_samples`。
- 均值定点误差上界 1 个定点单位（`1/scale` 毫秒）；P95 为最近秩、无插值，
  最大值为精确整数样本；聚合与刷新顺序无关。超预算样本保留并登记 `E_LATENCY`。

## 7. 版本、事务与恢复
- 声明/预算/配额/缩放因子变化即新版本号；旧版本写入 `stale_versions`，当前视图排除。
- 每分钟在一个 SQLite 事务内写 `decisions`、`snapshots`、`progress` 与延迟样本。
- 崩溃后仅完整提交分钟可恢复；未提交分钟进入 `unknown_minutes` 并要求重跑，
  重跑跳过已提交分钟，绝不覆盖。
- 恢复后按声明版本重放并逐项比对，分歧报 `E_REPLAY`；快照摘要不符报 `E_SNAPSHOT`。

## 8. 错误码与退出码
`E_SCHEMA=2, E_LENGTH=3, E_DIVERGE=4, E_PREVIEW=5, E_LATENCY=6, E_REPLAY=7, E_SNAPSHOT=8`。
