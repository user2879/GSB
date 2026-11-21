# QoS 定点误差传播诊断组件

`qos-fixed-diag` 是一个面向本地 Windows/Python 环境、完全离线运行的 QoS
流量整形诊断包。它从 JSON 加载整形策略与级联图，在本机 `127.0.0.1` 高位
UDP 端口上产生确定性数据报，按级联关系逐级执行整数令牌桶，记录每个数据报
在每一级的放行、排队和丢弃决策，并沿级联合成定点误差上界。

组件同时提供：

- 可集成的 Python 包 API；
- SQLite 事务账本、强杀恢复、版本化快照与确定性重放；
- Tkinter 桌面诊断界面；
- 单级整形、三级级联、快照恢复重放三个稳定顺序的示例场景。

不依赖 Docker、外部服务、真实网卡或其他授权目标。

## 快速开始

Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"

qos-fixed-diag demo --output artifacts/demo
pytest
qos-fixed-diag ui
```

Tkinter 随 Windows 官方 Python 安装器提供。安装时需启用
`tcl/tk and IDLE`。Debian/Ubuntu 可使用 `sudo apt install python3-tk`。

如果不想安装包，也可在源码目录使用：

```powershell
$env:PYTHONPATH = "src"
python -m qosfdiag demo --output artifacts/demo
python -m pytest
```

生成全部本地 JSON 样本：

```powershell
python examples/generate_samples.py
```

样本写入 `examples/data/`，生成过程不访问网络。

## 示例场景

`python -m qosfdiag demo --output artifacts/demo` 顺序执行：

1. `single_stage`：单级整形器，展示基础放行统计和误差项；
2. `three_stage`：`edge -> core -> tail` 三级级联，产生真实排队、丢弃、
   延迟误差，并把一个仅速率相差 1 kbps 的配置判为误差内不可区分；
3. `snapshot_restore_replay`：保存三级快照，以旧单级会话打开快照，触发整体
   替换和只读备份，再按快照记录的策略版本、缩放因子和包序列重放校验。

稳定输出摘要写入 `artifacts/demo/summary.json`。每次演示会清理自己生成的
`artifacts/demo/*.sqlite*` 与快照目录，避免增量残留。

本机实际运行摘要：

```text
single_stage: generated=24 released=24 bytes=11860
              bytes_bound=24000 avg_delay=0 ns
three_stage:  generated=40 released=25 bytes=21771
              bytes_bound=30700 avg_delay=31594936 ns delay_bound=6000000 ns
              dominant=edge indistinguishable=true diff=0 bound=61424
snapshot:    applied=true backup_read_only=true
              restored_ledger_cursor=40 decisions=184
              replay_ok=true events_compared=184
```

## 策略 JSON

完整样本见：

- `examples/data/single_stage_policy.json`
- `examples/data/three_stage_policy.json`
- `examples/data/three_stage_near_equal_policy.json`

示例：

```json
{
  "policy_version": "three-stage-v1",
  "max_committed_rate_kbps": 1000000,
  "shapers": [
    {
      "shaper_id": "edge",
      "committed_rate_kbps": 2000,
      "burst_bytes": 3000,
      "queue_limit_packets": 8,
      "match_dst_port_prefix": 530
    },
    {
      "shaper_id": "core",
      "committed_rate_kbps": 1200,
      "burst_bytes": 3000,
      "queue_limit_packets": 8,
      "match_dst_port_prefix": 530
    }
  ],
  "edges": [
    {"upstream": "edge", "downstream": "core"}
  ]
}
```

校验规则：

- `shaper_id` 必须唯一且为非空字符串；
- `committed_rate_kbps` 必须是正整数，且不超过 `max_committed_rate_kbps`
  或 API 参数 `max_rate_kbps`；
- `burst_bytes` 与 `queue_limit_packets` 必须为非负整数；
- `match_dst_port_prefix` 必须为 `1..65535` 的整数；
- 边端点必须存在，不能自环或重复，每个整形器至多一个上游；
- 级联图必须无环；
- 不允许空整形器集合；
- 所有错误一次性以 `E_POLICY` 抛出，按 `(shaper_id, field, message)` 稳定
  排序，不进行部分加载。

目的端口前缀采用十进制字符串前缀语义。例如前缀 `530` 匹配 `53000` 和
`53099`，不匹配 `53100`；前缀 `5000` 可精确匹配端口 `5000`，也可匹配
`50000..50009`。数据报从所有匹配的根整形器进入；根集合和有向边都按稳定
顺序遍历，分叉时每个下游分支接收一份数据报。

## 整数定点模型

会话参数 `scale = D` 必须为 `0..6` 的整数。令：

```text
S = 10^D
1 个令牌单位 = 1/S 字节
1 个时间 tick = 1/S 秒 = 10^(9-D) 纳秒
每 tick 令牌数 = kbps * 1000 / 8 = kbps * 125
```

因此速率、令牌、时间使用同一套定点单位。整数 kbps 在
`S = 10^D` 下，每个 tick 的令牌数 `kbps*125` 是精确整数，单速率量化误差为
`0 bytes/s`；误差主要来自时间边界量化和队列延迟取整。

显式规则：

- 虚拟到达时间：向零方向截断到 tick，非负输入等价于向下取整；
- 队列释放：向上取整到下一个 tick，保证不会提前放行；
- 队列：严格 FIFO，队长单位是包，不能超过 `queue_limit_packets`；队列非空时，
  新到包不得绕过队首，即使其字节数小于当前令牌余量；
- 令牌补充：`tokens += rate_units_per_tick * elapsed_ticks`；
- 令牌饱和：无论补充多少，令牌余量钳制到 `burst_bytes * S`；
- 溢出：Python 任意精度整数，不存在机器字溢出，最终由 burst 容量钳制；
- 包成本：完整 UDP 数据报字节数 `packet.size_bytes * S`；
- 包成本大于 burst 容量时直接 `DROP`，因为该包永远不可能合规放行。

队列释放时间：

```text
deficit_units = packet_cost_units - current_tokens
delay_ticks   = ceil(deficit_units / rate_units_per_tick)
release_tick  = enqueue_tick + delay_ticks
```

### 误差上界推导

对第 `i` 级整形器：

1. 速率量化误差：

   ```text
   E_rate_i = |represented_bytes_per_s - exact_bytes_per_s| = 0
   ```

2. 单次令牌补充边界误差：

   一次到达或释放事件的时间量化最多损失不到 1 tick，即最多
   `r_i = committed_rate_kbps_i * 125` 个令牌单位。若该级发生 `R_i` 次
   arrival/release 边界，则保守累积上界为：

   ```text
   E_refill_units_i = R_i * r_i
   E_refill_bytes_i = ceil(E_refill_units_i / S)
   ```

3. 队列延迟量化误差：

   到达时间向下取整误差 `< tick_ns`，释放时间向上取整误差 `< tick_ns`，
   因而每个发生排队的包的延迟误差上界为：

   ```text
   E_delay_i = 2 * tick_ns
   ```

4. 沿拓扑顺序合成：

   ```text
   B_i = E_rate_i + E_refill_bytes_i
   B_total = sum(B_i)
   D_total = sum(E_delay_i)
   share_bytes_i = B_i / B_total
   share_delay_i = D_total == 0 ? 0 : E_delay_i / D_total
   ```

结果会输出每一级的推导中间项、贡献值、占比和累计上界。主导级规则为：选择
字节误差占比最大的级；只有该最大占比不小于阈值 `0.50` 时才报告该级，否则
报告 `NONE`。占比并列时先比较延迟占比，再按整形器标识升序决胜。

### 不可区分判定

两个配置先按 `config_fingerprint` 规范化排序，判定不依赖调用参数顺序：

```text
abs(released_bytes_a - released_bytes_b)
    <= bytes_error_bound_a + bytes_error_bound_b
```

成立时结果为 `INDISTINGUISHABLE`，并设置 `error_affected=true`。不成立时放行
字节更多者胜；字节相同的情况已被上式覆盖。平均延迟及其误差界作为佐证信息
输出，但不会暗中放宽题目要求的字节判定。

## 确定性发生器与回环 UDP

`DeterministicUdpGenerator` 使用固定 seed 的 NumPy PCG64 序列生成：

- 包字节数；
- 目的端口；
- 确定性 payload；
- 整数包间隔和虚拟时间戳。

二进制数据报格式为固定的 34 字节头加 payload：

```text
magic="QFD1", seq, timestamp_ns, total_size, dst_port, payload_len
```

`packet_id` 由 seq 稳定格式化为 `p00000001`。序列摘要会持久化到账本和快照，
重放前先重新生成并比较；重生成失败会被标为不可复现，而不是误报账本缺陷。

`LoopbackTransport` 在 `127.0.0.1` 上从高位端口池（默认 `49152+`）为每个整形器
绑定独立 UDP ingress socket，并额外绑定最终 sink。根数据报、逐级放行/释放后的
数据报和最终接收都走真实 UDP `sendto/recvfrom`。虚拟时间只用于令牌桶与确定性
调度，不依赖墙钟休眠；因此同一输入在本地可重复得到同一决策序列。

每级事件都记录：

- `PASS`：立即放行；
- `QUEUE`：令牌不足但队列未满，进入 FIFO 尾部；
- `DROP`：队列满、包大于 burst，或入口无匹配根；
- `RELEASE`：排队包到达释放 tick 后真实转发。

事件包含当时令牌余量、决策前后队长、时间 tick、延迟、释放 tick、转发目标和
原因。

## SQLite 账本与强杀恢复

默认每批 `batch_size` 个生成数据报。处理一批前，先单独提交一个 `inflight`
标记；该批的全部逐级决策、令牌/队列状态、进度游标、整数账本格式版本与内容
摘要再在同一个 SQLite 事务中提交。事务成功时同时删除标记。

若进程在标记之后、批次提交之前被强杀：

- 已提交批次的记录完整可见；
- 未提交批次的决策不存在；
- 重启看到遗留 marker，把这些数据报写入 `unknown_packets`，标记
  `requires_rerun=true`；
- 恢复只加载最后一个完整提交的状态快照，从游标重跑 unknown/后续批次；
- 重跑成功提交后自动把未知记录标为已解决，并记录解决的批次号；
- 不生成或伪造“强杀瞬间退出报告”。

账本内容摘要覆盖已提交决策、进度游标和语义状态；非语义的堆调度计数器被排除。
同一输入在“不中断”和“强杀后恢复”两条路径上会得到相同的决策序列、状态和
内容摘要。

实际强杀验证：

```powershell
qos-fixed-diag run `
  --policy examples/data/three_stage_policy.json `
  --generator examples/data/three_stage_generator.json `
  --scale 3 --ledger artifacts/crash.sqlite `
  --crash-after-marker-batch 2

# 上一条命令以 os._exit(137) 结束，且无退出报告
qos-fixed-diag run `
  --policy examples/data/three_stage_policy.json `
  --generator examples/data/three_stage_generator.json `
  --scale 3 --ledger artifacts/crash.sqlite
```

第二次运行会报告 recovered unknown 包，并完成到 `cursor=40`。

## 快照、整体替换与只读备份

`SnapshotManager.save(...)` 写入：

- 快照 schema 版本与策略版本；
- 缩放因子；
- 整形器集合摘要和级联图摘要；
- 完整策略与图；
- 发生器参数、完整包序列及序列摘要；
- 逐级决策账本、进度状态、汇总和账本内容摘要；
- 误差传播结论及可选不可区分结论；
- 最终快照内容摘要。

`SnapshotManager.open(path, current=...)` 比较快照策略版本和内容摘要与当前
会话。若不一致，返回整组替换结果：

```text
cascade_graph, error_conclusions, ledger, policy_set, scale, shaper_set
```

它不会把新账本增量合并到旧状态。若当前会话指向一个旧快照文件，旧文件先复制
到 `snapshots/backups/` 并设置为只读；否则将当前会话描述写成只读备份。替换
结果是完整的 `policy_set/shaper_set/cascade_graph/scale/ledger/error_conclusions`。

若需要把替换结果真正落成 SQLite 会话，可调用：

```python
document.restore_ledger("artifacts/restored.sqlite", overwrite=True)
```

该方法先在临时数据库写入快照策略、包序列、逐级决策、状态和游标，重算并校验
账本内容摘要，再原子替换目标文件；旧数据库不会发生行级合并。目标已有文件时
必须显式传入 `overwrite=True`。

快照列表按 `(policy_version, content_digest, filename)` 稳定排序。

## 重放校验

```python
from qosfdiag import verify_snapshot_replay

result = verify_snapshot_replay(
    "artifacts/demo/snapshots/three-stage.json",
    current_policy_version="three-stage-v1",
    current_scale=3,
)
assert result.ok
```

校验顺序：

1. 快照 schema、快照内容摘要、包序列摘要；
2. 当前策略版本、缩放因子、发生器摘要；
3. 重新生成完整包序列，逐包比较记录；
4. 用快照策略和图重新驱动整数整形引擎；
5. 比较每个事件的决策、令牌、队长、时间、延迟、释放 tick、转发目标和原因；
6. 比较最终放行字节数、各级统计和误差上界。

事件查找顺序稳定为 `(packet_seq, stage_id, kind)`。首个不一致会报告：

```text
E_REPLAY: first_deterministic_mismatch
packet_id=... shaper_id=... field=... expected=... actual=...
classification=LEDGER_DEFECT
```

策略版本、scale、包序列或重生成摘要不一致时，报告
`classification=UNREPRODUCIBLE`，并说明原因，不误报为账本缺陷。部分快照缺少
已提交 summary/误差结论时，要求重跑，同样归类为不可复现。

## Python API

```python
import json

from qosfdiag import (
    CascadeEngine,
    DecisionLedger,
    DeterministicUdpGenerator,
    SnapshotManager,
    compare_configurations,
    load_policy,
    propagate_fixed_errors,
    verify_snapshot_replay,
)

policy = load_policy("examples/data/three_stage_policy.json")
with open("examples/data/three_stage_generator.json", encoding="utf-8") as handle:
    generator = DeterministicUdpGenerator(json.load(handle))

with DecisionLedger("artifacts/api.sqlite") as ledger:
    with CascadeEngine(
        policy,
        generator,
        scale=3,
        use_loopback=True,
        base_port=49200,
    ) as engine:
        run = engine.run(ledger=ledger)

    errors = propagate_fixed_errors(
        policy,
        run.summary,
        run.events,
        scale=3,
        generator_digest=generator.sequence_digest,
    )

    manager = SnapshotManager("artifacts/snapshots")
    snapshot = manager.save(
        filename="three-stage.json",
        policy=policy,
        scale=3,
        generator=generator,
        ledger=ledger,
        errors=errors,
    )

    # 整体物化替换账本（不会与旧库增量合并）
    snapshot.restore_ledger("artifacts/restored.sqlite", overwrite=True)

result = verify_snapshot_replay(snapshot)
print(run.summary.final_released_bytes, errors.bytes_error_bound)
```

对另一组结论调用 `compare_configurations(errors_a, errors_b)` 可得到规范化、
参数顺序无关的不可区分结果。

## Tkinter 诊断界面

```powershell
qos-fixed-diag ui
# 或加载指定 JSON
qos-fixed-diag ui --policy examples/data/three_stage_policy.json `
  --generator examples/data/three_stage_generator.json
```

界面分为：

- `Overview`：根、级联边、sink、各级放行/排队/丢弃统计、误差摘要、不可区分判定；
- `Timeline`：按虚拟时间、包序、拓扑序稳定排序的逐事件令牌余量和队列长度；
- `Error heat`：各级字节/延迟误差贡献、占比和主导级高亮；
- `Snapshots`：稳定排序的快照列表；可保存当前快照并重放选中快照。

界面逻辑通过 `qosfdiag.ui_model.build_view` 与 Tk 解耦，视图模型可在无图形环境
下测试。缺少 Tk 或显示服务时，`launch_ui` 给出明确错误，不影响包导入和非 UI
API。

## CLI

```text
qos-fixed-diag demo [--output DIR] [--json]
qos-fixed-diag generate-samples [--output DIR]
qos-fixed-diag validate --policy FILE [--max-rate-kbps N]
qos-fixed-diag run --policy FILE --generator FILE [options]
qos-fixed-diag replay --snapshot FILE [--current-policy-version V]
qos-fixed-diag ui [--policy FILE] [--generator FILE]
```

`run` 支持 `--ledger` 继续已提交游标、`--reset` 清理、`--in-memory` 跳过网络、
`--snapshot-dir` 自动保存快照，以及用于故障注入的
`--crash-after-marker-batch N` / `--crash-before-commit-batch N`。

## 测试

```powershell
python -m pytest -q
```

测试覆盖：

- `E_POLICY` 聚合校验、无环/单上游、端口和类型校验；
- 定点 tick/令牌公式、截断、向上取整、容量饱和；
- FIFO、队列上限、丢弃、三级真实 UDP sink；
- 误差项、主导级、参数顺序无关的不可区分判定；
- 批次强杀、unknown 标记、恢复后摘要逐字节一致；
- 快照内容摘要、整体替换字段、只读备份；
- 从快照原子物化替换 SQLite 账本；
- 重放成功、首个字段差异定位、不可复现分类；
- UI 视图稳定顺序、三场景可重复性；
- 子进程真实 `os._exit(137)` 强杀后重启。

本机实际结果见 `VALIDATION.md`。

## 设计边界

- 这是诊断器，不是内核 qdisc；延迟是虚拟整形队列延迟，不包含操作系统网络栈。
- 发生器使用真实回环 UDP，但虚拟时间不按墙钟休眠；保证可重复性优先。
- 目的端口“前缀”按十进制字符串解释，不按 IP 前缀或位掩码解释。
- 账本用 SQLite WAL + `synchronous=FULL`。强杀后可恢复到最后完整提交批次。
- 快照替换 API 同时提供完整替换视图与原子账本物化；禁止逐字段合并。
