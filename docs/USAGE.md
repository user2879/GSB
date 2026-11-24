# 本地 GPU 池调度、账本恢复与诊断界面

本交付不依赖 Docker、外部账号、网络 GPU 或云服务。资源池端口均为 `127.0.0.1`，执行器是由调度器拉起的本机 Python 子进程，调度客户端通过标准库 HTTP/JSON 发起真实回环连接，执行器使用 NumPy `int64` 做确定性整数定点计算。

## 1. 环境与自检

要求 Python 3.11+，安装 NumPy：

```bash
python -m pip install -r requirements.txt
# 或 python -m pip install numpy
python -m gpu_scheduler doctor
```

`doctor` 会输出 Python、NumPy、Tkinter 和回环地址信息。当前无图形库的 Linux 容器可能缺少 `libtk8.6.so`；核心 CLI、执行器、账本和验收不依赖显示环境，Tkinter 界面在官方 Windows Python/Tk 环境或安装 Tk 运行库后可启动。

## 2. 一键验收

```bash
python -m unittest discover -s tests -v
python -m gpu_scheduler verify --workdir runtime/acceptance
```

验收覆盖：

- 定点水位、截断方向和误差上界；
- 真实执行器子进程与回环调用；
- 高优先级任务触发的稳定抢占、分片边界挂起和游标恢复；
- 分片/任务依赖失效传播、稳定拓扑序和未受影响回执逐字节复核；
- `SIGKILL` 后 open attempt/unknown 分片识别与重跑；
- `SIGTERM` 安全边界停止和完整退出报告；
- `E_SPEC`、`E_CYCLE`、`E_PORT`、`E_SNAPSHOT` 业务码。

完整 JSON 报告写入命令输出中的 `full_report_path`，摘要写入 `runtime/acceptance/latest-report.json`。

## 3. 常用命令

全局 `--workdir` 放在子命令前：

```bash
python -m gpu_scheduler --workdir runtime/demo validate samples/platform.json
python -m gpu_scheduler --workdir runtime/demo run samples/platform.json
python -m gpu_scheduler --workdir runtime/demo snapshot-verify \
  samples/platform.json runtime/demo/session-snapshot.json
```

也可使用根目录包装入口：

```bash
python main.py --workdir runtime/demo run samples/platform.json
```

反复调整全局 GPU 配额：

```bash
python -m gpu_scheduler --workdir runtime/quota-6 run samples/platform.json --global-gpu-quota 6
```

在已有账本上追加一次配额调整水位轮，并同步刷新快照：

```bash
python -m gpu_scheduler --workdir runtime/demo quota samples/platform.json \
  --global-gpu-quota 6 --write-snapshot
```

## 4. Tkinter 诊断界面

```bash
python -m gpu_scheduler --workdir runtime/ui ui samples/platform.json
```

界面功能：

- 展示全局与每池 GPU、显存、活动任务水位和定点误差；
- 展示运行、排队、unknown 分片；
- 展示抢占序列、稳定被抢占集合、选择规则和挂起游标；
- 点击“权威重算并逐字节比较”后，在独立线程从 SQLite 重算视图，并与 `ui-cache.json` 的规范字节比较；
- 不一致时显示首个差异的 JSON 路径和字段，并在顶部标红；
- “运行演示调度”会在后台线程拉起真实执行器子进程。

## 5. 三类异常注入

短参数和业务码别名均可：

```bash
# E_CYCLE：分片依赖成环
python -m gpu_scheduler --workdir runtime/e-cycle validate \
  samples/cycle_injection.json
python -m gpu_scheduler --workdir runtime/e-cycle validate \
  samples/platform.json --inject E_CYCLE

# E_PORT：占用资源池回环端口
python -m gpu_scheduler --workdir runtime/e-port run \
  samples/preemption.json --inject port

# E_SNAPSHOT：运行后篡改 SQLite 轮次摘要并重放
python -m gpu_scheduler --workdir runtime/e-snapshot run \
  samples/propagation.json --inject snapshot
```

错误以单行结构化 JSON 输出到 stdout，同时写入 `<workdir>/logs/cli.jsonl`。普通字段校验失败返回 `E_SPEC`，并在 `locator` 中给出首个非法 `entity_type/entity_id/field`；依赖环按异常契约返回 `E_CYCLE`，`locator.cycle` 给出稳定环路径。

## 6. 强制终止与可处理信号

手动强杀演练：

```bash
python -m gpu_scheduler --workdir runtime/kill run samples/recovery.json \
  --crash-ready-file runtime/kill/crash-ready.json &

# 等待 runtime/kill/crash-ready.json 出现，其中有 pid 和 active_shards
kill -9 <crash-ready.json 中的 pid>

python -m gpu_scheduler --workdir runtime/kill recover samples/recovery.json
python -m gpu_scheduler --workdir runtime/kill run samples/recovery.json
```

恢复规则：

- `attempts.state=open` 或最后轨迹为 `dispatched/running` 但没有回执的分片标记为 `unknown`；
- unknown 分片必须重跑；
- 已和轮次在同一 SQLite 事务提交的回执、水位、抢占、挂起状态和游标才可作为权威状态；
- `SIGKILL` 无法生成退出报告，验收会断言不存在该进程写出的 exit report。

可处理信号：

```bash
kill -TERM <scheduler-pid>
```

调度器停止派发新分片，等待在途分片到达安全边界，提交账本后写入：

- `<workdir>/reports/exit-report-<pid>-sigterm-<ts>.json`
- `<workdir>/reports/latest-exit-report.json`

CLI 以 `128 + 15 = 143` 退出；`SIGINT` 对应 `130`。

## 7. 快照整体替换与重放

每次成功运行写 `<workdir>/session-snapshot.json`，内容包括：

- 整数快照/账本格式版本；
- 资源池版本、任务版本、缩放口径版本和完整声明；
- 各轮水位；
- 抢占序列和传播集合；
- 分片终局、回执摘要、资源摘要；
- 账本游标和链摘要；
- `content_sha256`。

重新打开时先校验外层内容摘要。若当前声明的 `spec_digest` 或版本摘要与快照不一致，不做字段级增量合并，而是把快照内嵌的完整声明写到 `<workdir>/restored-spec.json`，返回 `merge_mode=whole_replace`。随后逐轮重放 SQLite 账本，并逐项比较快照中的终局状态、回执摘要、终局摘要和资源摘要；首个差异定位到 `shard_id + field`。

## 8. 关键实现约定

### 资源口径

每个运行分片占用任务声明的 `gpu_count` 张 GPU，以及 `gpu_count * memory_mib_per_gpu` MiB 显存。调度同时检查：

- 全局 `global_gpu_quota`；
- 池物理卡数和池 `quota_share`（取两者较小值）；
- 池显存总容量；
- 池 `max_concurrent_tasks`；
- 任务自身每卡显存需求形成的显存预算。

### 定点水位

设缩放因子 `S = 10^scale_power`，`scale_power ∈ {0,1,2,3}`：

```text
ratio_fixed = floor(used * S / capacity)
真实比例    ∈ [ratio_fixed / S, (ratio_fixed + 1) / S)
误差上界    < 1 / S，显式记录为 1 个 fixed unit
截断方向    = 非负值向下 floor
```

### 稳定抢占规则

候选牺牲任务按以下键全局排序，不依赖池字典或哈希遍历顺序：

```text
priority ASC,
submitted_monotonic_ms DESC,
completed_shard_count ASC,
task_id ASC
```

先在高优先级任务所属池内选择候选以满足本地 GPU/显存/并发约束，再跨池选择候选以满足全局 GPU 配额。已发出的抢占不会取消正在执行的执行器调用，而是等当前分片回执后，在下一分片边界挂起，并把 `next_shard_id` 写入 `cursor_rows` / `suspension_rows`。

### 回执不可变复核

```bash
python -m gpu_scheduler receipts-export runtime/propagation runtime/before.json
# 执行 retry/outcome change 后
python -m gpu_scheduler receipts-verify runtime/propagation \
  --before runtime/before.json \
  --affected etl#0 --affected etl#1 --affected etl#2 --affected mart#0
```

未受影响分片的原始回执行、规范 JSON 字节和 SHA-256 摘要必须保持不变，且不会产生新的最新回执。
