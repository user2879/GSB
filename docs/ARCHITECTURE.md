# 实现说明

## 范围与确定性语义

本项目是完全本地运行的首版复核工作台：

- 输入为本地 JSON 服务声明、请求序列、预算与租约策略。
- 不访问网络、账号、GPU 或容器。
- 时间裁决使用 `time.monotonic_ns()` 换算的单调毫秒，不使用可人工调整的虚拟时钟。
- 所有跨服务结论均按 `service_id` 字典序、分钟升序输出；规范 JSON 使用排序键和紧凑分隔生成 SHA-256 摘要。
- Tkinter 是 Windows 官方 Python 安装包自带的可选界面；当前 Linux 验证环境缺少 `libtk`，因此 GUI 只做编译与缺依赖提示验证，核心验收由 CLI/SQLite 完成。

“每服务并发配额”的量纲是请求并发槽 `Q_s`。每个副本提供 `C_s=concurrency_per_replica` 个槽，因此服务的副本配额为：

```text
service_replica_quota_s = floor(Q_s / C_s)
```

全局预算直接限制同时持有的副本租约数。账本同时记录请求并发槽消耗/余量和副本槽消耗/余量。

## 模块映射

- `scale_review/spec.py`：JSON schema、强类型校验、首个非法字段 `E_SPEC`。
- `scale_review/estimator.py`：整数定点向上取整、夹取规则和 NumPy 批量交叉校验。
- `scale_review/algorithms.py`：全量逐分钟路径、固定长度环形缓冲增量路径、工作量模型和稳定路径选择。
- `scale_review/allocation.py`：全局预算 + 服务配额的确定性分配、租约占用 ID 和规范结论字节。
- `scale_review/consistency.py`：逐服务、逐分钟、逐字段比较；首个差异以 `E_DIVERGE` 给出，并保存两路径输入视图。
- `scale_review/leases.py`：单调时钟租约、版本号乐观续约、到期/撤销唯一裁决、事件重放。
- `scale_review/scheduler.py`：固定优先级队列、`max_inflight_tasks`、单任务副本上限和单调毫秒等待区间。
- `scale_review/propagation.py`：方案版本、滚动追加/预算/因子失效闭包、脏子集重算和旧字节复用。
- `scale_review/ledger.py`：SQLite 事务轮次、摘要链、运行尝试标记、快照版本/摘要整体替换。
- `scale_review/replay.py`：已提交轮次重算、`E_REPLAY` 字段级定位、未完成轮次游标恢复。
- `scale_review/acceptance.py`：一键验收和 `artifacts/acceptance-report.json` 证据生成。
- `scale_review/gui.py`：Tkinter 工作台。
- `scripts/verify.py`：单元测试 + 一键验收 + 机器可读总报告。
- `scripts/Verify-Windows.ps1`：Windows PowerShell 安装依赖、验证、启动 GUI。

## 路径选择规则

设序列长度为 `n`，窗口长度为 `W`：

```text
full_work        = n + sum(min(W, t + 1)), t=0..n-1
incremental_work = W + 4n
work_delta       = abs(full_work - incremental_work)
```

- `n < path_switch_length`：选择 `full`，原因 `length_below_switch_threshold`。
- `work_delta <= effort_tie_boundary`：选择 `tie_break_path`，原因 `estimated_effort_within_explicit_tie_boundary`。
- 否则选择预估工作量更小的路径。

样本 `n=18, W=4` 时，`full_work=84`、`incremental_work=76`、差值 `8 > 2`，因此选择 `incremental`。

## 唯一归属规则

每个全局副本槽和服务副本槽在锁内分配给最低空闲槽。终止裁决规则为：

1. 租约已有终态：后续 `expire` 或 `batch_revoke` 只记录无效竞争事件，不再次释放配额。
2. 观察时刻 `now >= expires_at_monotonic_ms`：到期裁决具有客观优先权；即使先调用批次撤销，生效事件的 `source` 仍是 `expiry`。
3. `now < expires_at_monotonic_ms`：批次撤销可以生效；之后到期检查只记录 `already_revoked`。

因此同一租约/配额只会有一条 `effective=true` 的 `quota_released` 事件。

## 失效传播

- 追加分钟 `t`：仅新增 `t` 为脏；其 trailing window 会引用 `[max(0,t-W+1), t]`，但旧分钟的 trailing window 不包含未来点，旧规范字节不变。
- 全局预算 `B_old -> B_new`：分钟 `m` 的配额上限需求总量 `E_m > min(B_old,B_new)` 时，该分钟分配可能变化。
- 服务配额变化：仅当 `min(clamped, floor(Q/C))` 实际变化时标记该分钟；分钟内采用稳定服务序，因此同分钟下游分配一起重算。
- 缩放因子变化：先找整数估算结果变化的种子分钟，再沿窗口引用传播到 `seed..seed+W-1`，并向动作基线传播到 `seed+1`。

## 账本与恢复

每个提交轮次在同一 SQLite 事务写入：

- 需求中间项与分配结论；
- 租约归属和槽位；
- 配额消耗；
- 等待起止；
- 进度游标；
- 路径选择依据。

轮次负载和记录均有 SHA-256 摘要，账本维护 `previous_digest + payload_digest` 摘要链。强杀前可提交 `round_attempts(status='running')` 标记；恢复时将其改为 `unknown`，写入释放事件，且绝不把它当作已完成轮次。
