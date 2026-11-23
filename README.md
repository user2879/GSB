# leaseaudit：并发租约签发与配额回收审计组件

`leaseaudit` 是一个只面向本机回环网络的容量压测审计组件。它从本地 JSON 场景加载确定性来源，按单调时钟签发/续约并发租约，真实回收到期权益和长期低占用额度，并把每一次归属变化写入带内容摘要链的 SQLite 账本。账本可快照、整体替换恢复，也可由独立只读审计器从提交事件重新推导并逐项比对。

实现不依赖 Docker、不访问外部网络；演示服务只绑定 `127.0.0.1` 高位端口。

## 快速开始（Windows 11 / Windows 10）

1. 安装 Python 3.11 或更高版本，并选择官方安装包中的 Tcl/Tk 组件（官方 Windows 安装包默认包含）。
2. 在工作区根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # 可选：运行 pytest
```

3. 校验样本场景：

```powershell
python -m leaseaudit validate --scenario scenarios\demo_scene.json
```

4. 运行约 6 秒的三阶段真实回环演示：

```powershell
python scripts\run_demo.py --workdir artifacts\demo --base-port 55441
```

演示脚本会动态选择可用高位端口、启动本仓库的 `scripts/mock_source.py` 子进程，依次完成：

- 冷启动签发；
- `beta` 停止续约，到期后释放并发权益并由 `alpha`、`delta` 重签发；
- `beta` 连续低水位后回收 2 个单位到全局可分配额度；
- 独立重放；
- 账本快照、故意追加事件改变摘要、再整体替换恢复。

输出示例：

```text
SCENARIO cold_start code=OK issued=alpha:3,beta:3,gamma:1
SCENARIO expiry_reclaim code=OK expired=beta:3;reissued=alpha:1,delta:2
SCENARIO quota_reclaim code=OK reclaimed=beta:2
INVARIANT retained+global=capacity code=OK retained=9 global=3 capacity=12
WATERMARK code=OK samples=88 truncation=FLOOR_TOWARD_ZERO
AUDIT code=OK events=178 digest=5ad47a234f2c29421566e52a9eb85f6e16aef7ad1e27804d9c4a9c24c6aeed02
SNAPSHOT_RESTORE code=OK status=RESTORED digest=5ad47a234f2c29421566e52a9eb85f6e16aef7ad1e27804d9c4a9c24c6aeed02
NETWORK source=alpha requests=23 outcomes=success:23
NETWORK source=beta requests=10 outcomes=success:10
NETWORK source=delta requests=15 outcomes=success:15
NETWORK source=gamma requests=24 outcomes=connection_refused:24
OVERALL code=OK
```

网络请求数量会随本机调度略有变化，但阶段顺序、业务码、归属结论与最终不变量稳定。

## 常用命令

```powershell
# 运行本地演示
python -m leaseaudit demo --workdir artifacts\demo --base-port 55441

# 独立只读重放审计
python -m leaseaudit audit --ledger artifacts\demo\audit.sqlite --scenario artifacts\demo\scenario.json

# 稳定顺序的只读文本视图
python -m leaseaudit view --ledger artifacts\demo\audit.sqlite

# Tkinter 审计界面；刷新不会写账本
python -m leaseaudit gui --ledger artifacts\demo\audit.sqlite

# 快照与整体替换恢复
python -m leaseaudit snapshot --ledger artifacts\demo\audit.sqlite --out artifacts\demo\snapshots
python -m leaseaudit restore --snapshot artifacts\demo\snapshots\snap-XXXXXXXX-XXXXXXXXXXXX --ledger artifacts\demo\audit.sqlite

# 强制终止后恢复：未提交签发意图会变成 unknown，要求重签发
python -m leaseaudit crash-demo --workdir artifacts\crash-recovery
python -m leaseaudit recover --ledger artifacts\crash-recovery\crash-ledger.sqlite --scenario artifacts\crash-recovery\crash-scenario.json

# 三类 CLI 故障注入
python -m leaseaudit fault-demo --workdir artifacts\faults --kind scene_corrupt
python -m leaseaudit fault-demo --workdir artifacts\faults --kind quota_overflow
python -m leaseaudit fault-demo --workdir artifacts\faults --kind ledger_digest

# 测试
python -m pytest -q
$env:RUN_DEMO_INTEGRATION=1; python -m pytest -q
```

## 场景 JSON

完整可运行样本见 `scenarios/demo_scene.json`。根字段：

| 字段 | 要求 |
| --- | --- |
| `cluster_capacity` | 正整数，集群总并发容量 |
| `lease_duration_ms` | 正整数，租约时长 |
| `renew_interval_ms` | 非负整数，必须严格小于租约时长 |
| `reclaim_threshold` | 非负整数，定点水位阈值 |
| `reclaim_window_rounds` | 非负整数，默认 3；低水位必须**严格超过**该轮数 |
| `observation_window_ms` | 正整数，默认 250，观察窗口 |
| `watermark_scale` | 正整数，默认 1000000；水位定点缩放因子 |
| `load_duration_ms` | 正整数，压测时长 |
| `scenario_version` | 非空字符串，参与外部漂移识别 |
| `lease_params_version` | 正整数，默认 1，写入每个事件 |
| `sources` | 非空数组 |

每个来源：

| 字段 | 要求 |
| --- | --- |
| `source_id` | 非空字符串且全局唯一 |
| `address` | 必须是精确字面量 `127.0.0.1` |
| `port` | 1..65535 的整数，且全局唯一；演示只选高位端口 |
| `lease_share` | 非负整数；同一来源所有有效租约授予量之和不得超过它 |
| `reserve_quota` | 非负整数；不得大于该来源 `lease_share`；所有来源之和不得超过总容量 |
| `mode` | `success`、`fixed_delay` 或 `refuse` |
| `processing_ms` | 非负整数；`fixed_delay` 模式必须大于 0 |
| `request_interval_ms` | 正整数，默认 200 |
| `initial_issue` | 0..`lease_share`，演示首轮申请量 |
| `stop_renew_at_ms` | `null` 或非负整数，演示中停止续约的相对时刻 |
| `seed` / `request_timeout_ms` | 确定性来源辅助参数 |

强校验按固定顺序读取字段，发现第一个违规即整体拒绝，不创建或继续写账本。普通场景错误为 `E_SCENE`；保留配额总和越过总容量使用更具体的 `E_QUOTA`。错误对象包含 `field_path`、`details`，例如：

```json
{
  "code": "E_QUOTA",
  "field_path": "sources[1].reserve_quota",
  "details": {
    "sum_reserve_quota": 10,
    "cluster_capacity": 8,
    "excess": 2
  }
}
```

## 额度模型与唯一归属

每个来源有两个额度概念：

- `reserve_quota`：不可回收的保留底线；
- `retained_quota`：当前划给该来源的额度，初始等于保留底线，可因签发而从全局额度划入，也可因低水位回收而归还全局。

核心恒等式在每次事件提交后断言：

```text
Σ retained_quota + global_allocatable = cluster_capacity
0 <= source.active_granted <= source.lease_share
total_active <= cluster_capacity
retained_quota >= reserve_quota
```

一份额度在任一事件后只处于一种状态：

```text
global_allocatable -> retained_quota(idle) -> active lease
active lease -> retained_quota(idle)             # 到期/显式释放
retained_quota(idle, 高于保留线) -> global       # 配额回收
```

配额回收只处理 `retained_quota - max(reserve_quota, active_granted)` 的空闲部分，因此不会与有效租约归属冲突。租约到期会清零该来源低水位连续计数，避免同一 tick 中“租约到期释放”和“配额回收”重复归属同一份权益。

## 单调时钟与续约

- 默认使用 `time.monotonic_ns() // 1_000_000`，不使用墙上时钟。
- 签发事件记录 `issued_at_ms = now`、`expires_at_ms = now + lease_duration_ms`。
- 续约必须由当前持有者发起，且 `now < expires_at_ms`；成功后从续约时刻重新计算到期。
- 到期判定为 `now >= expires_at_ms`，到期后旧租约不能再续约；重复续约或释放会写 `RACE_REJECTED` 并以 `E_RACE` 返回。
- 引擎会把显式传入的旧 `now` 钳制到账本最后事件时刻，账本中的单调时间序列永不回退。
- 同一来源可以持有多个租约，但授予总量不会超过其 `lease_share`。

## 定点水位与配额回收

水位按以下整数公式计算，截断方向固定为 `FLOOR_TOWARD_ZERO`：

```text
ratio_ppm = floor(active_granted * watermark_scale / lease_share)
```

`lease_share == 0` 时比例为 0。低水位条件为 `ratio_ppm < reclaim_threshold`。只有当连续低水位轮数**严格大于** `reclaim_window_rounds` 时，才允许回收空闲额度；事件中记录：

- 最近观察窗口样本；
- 缩放因子与截断方向；
- 低水位连续轮数；
- 回收前后 `retained_quota`、`global_allocatable`；
- `reclaim_only_idle_above_reserve` 裁决依据。

`numpy` 用于演示侧的确定性请求间隔抖动（由来源 `seed` 派生）和独立重算定点水位，不参与 SQLite 事务或核心状态机。

## SQLite 审计账本

账本实现位于 `leaseaudit/ledger.py`，表包括：

- `ledger_meta`：整数格式版本、内容摘要、事件数、进度游标、租约参数版本/摘要、场景/外部配置摘要；
- `events`：事件序号、类型、来源、租约、数量、单调毫秒、参数版本、进度游标、前序摘要、事件摘要、JSON 裁决载荷；
- `issuance_intents`：签发意图的 `PENDING/COMMITTED/UNKNOWN` 状态；
- `unknown_leases`：强制终止时未提交签发意图的未知标记。

每个事件在一个 `BEGIN IMMEDIATE` 事务中同时写入事件、参数版本、进度游标并推进账本内容摘要。摘要链为：

```text
GENESIS -> hash(event_1) -> hash(event_2) -> ... -> ledger_meta.content_digest
```

打开账本时会重算全部事件摘要、游标、参数版本和链摘要；任何内容漂移均以 `E_SNAPSHOT` 定位到首个异常事件字段。

快照使用 SQLite 在线备份 API，生成不可变目录、数据库 SHA-256 和清单。恢复时先验证清单版本、文件 SHA-256、数据库内容和摘要链，再通过同目录临时文件 + `os.replace` 整体替换；失败会回退到 `.pre-restore` 备份。

## 强制终止恢复

签发前先将意图提交为 `PENDING`，随后租约事件与该意图的 `COMMITTED` 状态在同一事务提交。若进程在两者之间被强制终止，租约事件必然未提交；重启时：

1. 只从已提交事件恢复有效租约、`retained_quota` 和 `global_allocatable`；
2. 将残留 `PENDING` 意图写为 `LEASE_UNKNOWN`；
3. 在 `unknown_leases` 中标记“需重签发”；
4. 恢复状态永远报告 `exit_report_available = false`，不伪造强制终止退出报告。

## 独立重放与外部漂移

`leaseaudit/replay.py` 只以只读 URI 打开账本，并使用账本中记录的参数版本、参数摘要和事件顺序重新执行状态机：

- 重新计算每次签发的全局额度划转与到期时刻；
- 重新计算续约、到期/释放、观察和配额回收；
- 对每个事件比较来源状态、总授予、`retained_total`、`global_allocatable`、完整归属集合及归属摘要；
- 首个差异以 `E_REPLAY` 返回 `event_seq`、`source_id`、`field_path`、`expected`、`actual` 和 `basis`。

原始场景仅用于版本/外部配置关联。若场景版本或模拟服务地址、端口、模式等发生变化，结果标为 `EXTERNAL_DRIFT` / `E_EXTERNAL_DRIFT` 且 `ledger_defect = false`；重放本身仍使用账本记录的参数，不会把外部漂移混入账本缺陷。

## 故障注入

三类注入只改输入文件、配额参数或账本副本，不修改源码：

| `--kind` | 业务码 | 注入方式 | 可定位信息 |
| --- | --- | --- | --- |
| `scene_corrupt` | `E_SCENE` | 场景来源地址改为 `0.0.0.0` | `sources[0].address` |
| `quota_overflow` | `E_QUOTA` | 来源保留配额总和超出容量 | `sources[1].reserve_quota`、超额数额 |
| `ledger_digest` | `E_SNAPSHOT` | 修改账本事件载荷但不更新摘要链 | `$.events[1].event_hash`、期望/实际摘要 |

结果写入 `fault-*.json`，错误同时以 JSONL 写入 `structured.jsonl`。预期故障路径由 `fault-demo` 包装并正常退出。

## Tkinter 审计界面

```powershell
python -m leaseaudit gui --ledger artifacts\demo\audit.sqlite
```

界面按 `source_id` 稳定排序显示：有效租约及到期时刻、并发租约份额、保留配额、当前保留额度、有效授予、水位比例、已回收额度、回收依据、全局可分配额度、未知租约标记、账本格式/参数版本和按 `seq` 排序的事件列表。刷新按钮只调用只读投影函数，不追加事件、不改变账本摘要。

当前 Linux 验收容器的 Python 缺少 `libtk8.6`，无法实际拉起窗口；界面代码通过无 Tk 的只读投影测试验证。Windows 官方 Python 默认带 Tk，按上述命令可直接运行。

## 验证与文档

- 算法与状态机细节：`docs/design.md`
- 实际运行记录与复现命令：`docs/verification.md`
- 样本场景：`scenarios/demo_scene.json`
- 模拟来源脚本：`scripts/mock_source.py`
- 完整演示：`scripts/run_demo.py`
- 原样运行摘要：`reports/`
- 测试：`tests/`

最终本地验收结果：`34 passed`（含 5.6 秒真实回环网络演示），三类故障码分别为 `E_SCENE`、`E_QUOTA`、`E_SNAPSHOT`，强制终止恢复得到 1 个 `LEASE_UNKNOWN`、`exit_report_available=false`，重放为 `OK`。
