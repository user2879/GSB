# 设计与算法说明

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `leaseaudit/scenario.py` | 场景 JSON 强校验、规范化摘要、外部配置摘要 |
| `leaseaudit/clock.py` | 真实单调时钟、测试用手动单调时钟 |
| `leaseaudit/engine.py` | 签发、续约、到期、释放、观察、配额回收和状态不变量 |
| `leaseaudit/ledger.py` | SQLite 事务、摘要链、快照/恢复、签发意图和未知租约 |
| `leaseaudit/replay.py` | 独立只读状态机重放与首差异定位 |
| `leaseaudit/view.py` / `gui.py` | 稳定只读投影和 Tkinter 展示 |
| `leaseaudit/mock_service.py` / `driver.py` | 确定性回环服务与客户端请求 |
| `leaseaudit/faults.py` | 三类文件级故障注入 |
| `leaseaudit/crash_recovery.py` | `os._exit` 强制终止与恢复演示 |

## 2. 场景校验顺序

校验器不依赖 Python 字典的偶然顺序，而是按固定 schema 顺序检查。关键顺序为：

1. 根对象、必填根字段、未知根字段；
2. 格式版本、场景版本、参数版本；
3. 容量、租约时长、续约间隔及其严格大小关系；
4. 回收阈值、回收轮数、观察窗口、缩放因子、压测时长；
5. `sources` 非空数组；
6. 每个来源的必填/未知字段、标识唯一性、回环地址、端口范围与唯一性；
7. 份额/保留配额整数性、`reserve_quota <= lease_share`、累计保留配额上界；
8. 模式、处理时间、请求间隔、初始申请量等模拟参数。

变量不会在失败前暴露为可继续使用的 `Scenario` 对象，CLI 也不会在加载前创建账本。

## 3. 状态机

初始状态：

```text
retained_i = reserve_i
active_i = 0
global = capacity - Σ reserve_i
```

### 3.1 签发

给定来源 `i` 和数量 `a > 0`：

```text
active_i + a <= share_i
need = max(0, active_i + a - retained_i)
need <= global
retained_i += need
global -= need
active_i += a
```

签发事件中的 `quota_from_global` 等于 `need`。因此即使多个来源声明的 `lease_share` 总和大于容量，实际授予仍由全局额度和每源份额共同约束。

### 3.2 续约

```text
lease in active
lease.source = caller_source
now < lease.expires_at_ms
lease.expires_at_ms = now + lease_duration_ms
lease.renew_count += 1
```

到期时 `now >= expires_at_ms`。续约入口会先执行到期扫描，因此过期租约必然进入 `RACE_REJECTED`，不会延长。

### 3.3 到期/释放

```text
active_i -= lease.amount
retained_i unchanged
global unchanged
low_streak_i = 0
```

活跃租约转为来源空闲保留额度。其他来源能否同时申请，取决于其份额余量和全局可分配额度；低占用空闲额度若要转为全局额度，需要经过声明的低水位观察与回收流程。

### 3.4 观察

```text
ratio_ppm = floor(active_i * watermark_scale / share_i)
below = ratio_ppm < reclaim_threshold
low_streak_i = low_streak_i + 1 if below else 0
```

每次观察写一个 `OBSERVED` 事件，事件保存活跃量、份额、整数定点比例、阈值、连续轮数、缩放因子和截断方向。观察按 `source_id` 排序提交，使同 tick 来源处理顺序不影响结论。

### 3.5 配额回收

```text
eligible = low_streak_i > reclaim_window_rounds
amount = max(0, retained_i - max(reserve_i, active_i))
retained_i -= amount
global += amount
```

只回收空闲部分，保证回收不触碰有效租约。回收事件嵌入最近 `reclaim_window_rounds + 1` 个观察样本。没有可回收空闲额度或连续轮数不足时，显式重复回收请求写 `RACE_REJECTED`，状态不变并以 `E_RACE` 返回。

## 4. 时序与确定性

- 所有公共写操作在 `RLock` 内串行；
- 到期租约按 `(expires_at_ms, lease_id)` 排序；
- 同一 tick 的观察与回收按 `source_id` 排序；
- `issue_many` 对任意输入顺序先按 `(source_id, amount)` 规范化；
- 归属快照按来源、租约标识排序后计算摘要；
- 显式 `now_ms` 小于账本最后事件时刻时，钳制到最后事件时刻，避免回退。

这些规则使“来源处理顺序”不改变归属集合、裁决依据和归属摘要。直接逐一调用 `issue()` 的事件序号仍反映真实串行调用顺序；需要同 tick 竞争安全时使用 `issue_many()` 或观察/到期自动入口。

## 5. 账本事务与摘要

事件摘要输入包括：

```text
seq, event_id, event_type, source_id, lease_id, amount, mono_ms,
params_version, params_digest, progress_cursor, payload
```

链摘要：

```text
content_digest_n = SHA256(content_digest_{n-1} + event_hash_n)
```

事务顺序：

1. `BEGIN IMMEDIATE`；
2. 读取 meta 中最后序号和摘要；
3. 写入事件、参数版本、`progress_cursor = seq`；
4. 更新 meta 的事件数、最后序号和内容摘要；
5. 若为签发事件，同事务将对应 `PENDING` 意图更新为 `COMMITTED`；
6. `COMMIT`。

任何异常均回滚。打开时重算所有链值、序号连续性、游标和 meta 摘要；发现异常以 `E_SNAPSHOT` 报首个定位字段。

## 6. 强制终止窗口

```text
T0: COMMIT issuance_intent(PENDING)
    <- 强制终止发生在此处
T1: BEGIN; INSERT lease event; UPDATE intent=COMMITTED; UPDATE meta; COMMIT
```

若在 T1 前终止，恢复只看到 `PENDING`，不看到租约事件。恢复器写 `LEASE_UNKNOWN`，并保持活跃租约集合不包含该意图。若 T1 已完整提交，意图和租约事件同时可见。不存在“租约已生效但意图未提交”或相反状态。

## 7. 独立重放

重放器不导入引擎状态，而是使用账本 `params_json` 初始化自己的初始状态。每个事件先验证参数版本/摘要与时间不回退，再推导预期状态与预期载荷字段，最后统一比较：

- `decision` 与 `basis`；
- 事件数量/来源/租约/时间字段；
- `source_state_after`；
- `total_active_after`、`retained_total_after`、`global_allocatable_after`；
- 完整 `ownership` 数组和 `ownership_digest`；
- 配额回收的连续轮数、阈值、观察窗口和精确回收量。

比较使用确定性的结构遍历顺序，因此报告的是首个差异。重放只读打开 SQLite，不执行迁移、修复或写操作。

## 8. 外部漂移分类

账本保存三类摘要：

- `params_digest`：容量、时间参数、阈值、来源份额与保留配额；
- `external_digest`：场景版本、来源地址/端口/模式/处理时间等外部配置；
- `scenario_digest`：完整规范化场景。

审计时先比较当前场景与账本场景。若不同，仍使用账本参数进行独立重放；重放成功则输出 `EXTERNAL_DRIFT`，`ledger_defect=false`，列出字段差异。重放失败才以 `E_REPLAY` 报告账本缺陷，因此模拟服务配置或版本变化不会被误判为账本问题。

## 9. 故障注入

- `scene_corrupt`：写一个地址非法的场景副本，调用同一严格加载器；
- `quota_overflow`：写一个保留配额总和越界的场景副本；
- `ledger_digest`：从完整账本复制数据库，仅修改事件 JSON 载荷，不更新摘要链，随后只读打开。

三类均不修改源码；前两类不会创建审计账本，第三类只读取受污染副本。

## 10. Tk 只读投影

`build_audit_view()` 先执行完整只读重放，再读取 meta、事件和 unknown 表，生成按来源和事件序号稳定排序的普通字典。Tk 层只消费该投影，不做时钟推进、不调用引擎、不写 SQLite。因此刷新按钮不会改变账本内容或摘要。
