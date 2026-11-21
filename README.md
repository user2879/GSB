# 中间件配额仲裁工作台（quota_workbench）

面向 Windows 本地环境的服务注册与健康探测配额仲裁工作台：多个租户共享一组
以回环 TCP 子进程表示的确定性模拟服务实例；真实套接字探测驱动实例健康判定，
健康判定驱动请求仲裁；探测判定、仲裁结局与配额账本演进全部固化为可复核记录。

仅依赖 Python 3.11+ 标准库（`sqlite3` / `socket` / `subprocess` / `threading` /
`tkinter`）与 NumPy（确定性样本与聚合验证）。不使用容器、不访问外部站点，
全部流量限制在 `127.0.0.1` 回环高位端口。

## 环境准备（Windows）

```bat
py -3.11 --version          :: 需要 Python 3.11 及以上
py -3.11 -m pip install numpy
```

Linux/macOS 对应命令为 `python3 -m pip install numpy`。验证前请预留若干空闲
回环端口（样本默认使用 18301–18304）。

## 快速开始

```bat
py -3.11 scripts\gen_sample.py                              :: 生成确定性样本
py -3.11 -m quota_workbench run --spec samples\spec.json --requests samples\requests.ndjson --db work\ledger.db
py -3.11 -m quota_workbench replay --spec samples\spec.json --requests samples\requests.ndjson --db work\ledger.db
py -3.11 -m quota_workbench snapshot-verify --spec samples\spec.json --snapshot work\snapshots\latest.json --db work\ledger.db
py -3.11 -m quota_workbench gui --db work\ledger.db         :: Tkinter 诊断界面
py -3.11 scripts\demo_gui.py                                :: 一键样本+运行+界面
```

## 声明与请求流格式

`samples/spec.json`（JSON）：

- `tenants[]`：`tenant_id`、整数 `share`（配额份额）、整数 `max_concurrency`、整数 `burst`
- `instances[]`：`instance_id`、`host`（必须为字面量 `127.0.0.1`）、整数 `port`
  （全局唯一）、整数 `weight`、`behavior`（`normal|refused|delay|error`）、`hosts`（承载租户）
- `probe_policy`：`interval_ms`、`timeout_ms`、`failure_threshold`、`eject_threshold`
- `rule_version`、`global_concurrency`、`total_share_cap`、`duration_factor_ms`
- 集合版本：`tenant_set_version` / `instance_set_version` / `probe_policy.policy_version`

`samples/requests.ndjson`（逐行 JSON）：`request_id`、`tenant_id`、正整数 `demand`、
单调不降的非负整数 `arrival_ms`。

任一违规整体拒绝加载：业务码 `E_SPEC`，稳定报告首个违规字段与（请求流的）行号；
租户份额之和超出 `total_share_cap` 报 `E_QUOTA`。

## 仲裁规则 v1（显式判决顺序）

每条请求在每次求值时按以下顺序判决并记录依据（basis）：

1. 租户剩余配额不足 → 拒绝 `R_QUOTA_EXHAUSTED`
2. 租户在途数达到 `max_concurrency + burst` → 排队 `R_TENANT_SLOTS_FULL`
3. 在途数超过 `max_concurrency` 部分计入突发额度（授予依据记录 `burst_used`）
4. 全局在途数达到 `global_concurrency` → 排队 `R_GLOBAL_SLOTS_FULL`
5. 目标实例健康：`ejected` → 拒绝 `R_INSTANCE_EJECTED`；非 `healthy` → 排队
   `R_INSTANCE_NOT_HEALTHY`；`healthy` → 授予 `R_GRANT`

授予实时扣减配额并占用在途；排队记录等待起止（虚拟毫秒时钟）；事件流结束时仍
排队（实例始终不健康）的请求以 `R_END_OF_STREAM` 拒绝。调度器为确定性离散事件
模拟：同一输入 + 同一规则版本多次执行，结局与依据序列逐字节一致（内容摘要相同）。

## 健康探测与状态机

探测只来自真实套接字：`connect` / `sendall` / `recv` 配合真实 `settimeout` 超时，
绝不以人为等待推断成败。实例在 `unknown → healthy | unhealthy → ejected` 间迁移
（连续失败达 `failure_threshold` 转 unhealthy，达 `eject_threshold` 转 ejected），
每次迁移记录探测轮次、连续失败累计与单调时钟毫秒。

## 账本、恢复与快照

- 每个仲裁批次 / 探测迁移集合在同一 SQLite 事务中写入请求结局、依据、租户配额
  余量、在途数、实例健康与进度游标；账本携带整数 `format_version` 与内容摘要。
- 强制终止（`--crash-after-batch N` 以 `os._exit` 模拟）后，`recover` 仅信任已
  提交记录：已提交但仍 `in_flight` 的请求标记为 `unknown` 并要求重跑，不生成
  强制终止时的退出报告；同时清理被终止运行遗留的实例子进程。
- 快照持久化四个版本号、配额终态、完成游标与内容摘要；重新打开时与当前声明
  比较，任何不一致以 `E_SNAPSHOT` 报出稳定差异列表，并整体替换（而非增量合并）
  租户、实例与账本状态（`snapshot-verify --replace`）。
- `replay` 按声明重新执行真实探测与仲裁，逐请求比较结局、依据序列、等待/授予/
  完成时刻、实例健康迁移与配额终态；首个不一致以 `E_REPLAY` 定位请求标识、
  租户与字段。

## 依赖失效传播

`quota_workbench/propagation.py` 建有向依赖图（`health:<实例>` → `tenant:<租户>` →
`request:<请求>`，另有 `rule` / `probe_policy` / `stats` 节点）并校验引用完整。
实例探测迁移、实例集合变化或规则版本演进时，按稳定（排序）顺序输出受影响集合
与每步传播依据；未受影响请求的结局与依据序列逐字节不变，可由
`verify_unaffected_unchanged` 逐项校验。

## 异常注入（仅改变输入或参数）

```bat
py -3.11 -m quota_workbench run ... --inject port-occupied     :: 占用声明端口 -> E_PORT
py -3.11 -m quota_workbench run ... --inject quota-overflow    :: 份额和超限   -> E_QUOTA
py -3.11 -m quota_workbench run ... --inject snapshot-corrupt  :: 快照摘要不符 -> E_SNAPSHOT
```

所有业务错误以 JSON 输出到 stderr（退出码 2），并写入结构化日志
（`work/run.log`，JSON Lines）。

## Tkinter 诊断界面

`python -m quota_workbench gui --db work/ledger.db` 展示：租户配额余量与在途数、
实例健康状态与迁移理由、探测轮次、请求结局分布、受影响集合、当前版本与快照
列表。渲染顺序稳定（排序键 / ORDER BY），刷新为只读操作，不改变账本内容。

## 测试

```bat
cd tests
py -3.11 -m unittest discover -v
```

覆盖：规格/请求流强校验、仲裁确定性与判决顺序、依赖传播、完整运行+重放一致、
强制终止恢复、三类异常注入、快照版本识别与整体替换、NumPy 聚合校验。

实际验证结果见 `VERIFICATION.md`；任务书原文见 `docs/TASK.md`。
