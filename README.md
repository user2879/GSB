# Meshwork · 本地服务网格流量治理诊断工作台

Meshwork 是一个面向 Windows 本地环境的确定性服务网格诊断应用。它在 `127.0.0.1` 高位端口上启动独立的上游模拟子进程，使用真实 TCP 套接字发送 JSON 行协议请求，在全局并发上限与每服务配额的双重预算下并行调度，维护权重路由、熔断器、整数定点 SLO、SQLite 尝试账本、版本快照、强制终止恢复和一致性重放。

项目只使用本地声明文件、本地回环套接字、SQLite 与 Tkinter；不访问真实站点、账号、容器或付费资源。

## 快速开始（Windows）

要求 Python 3.10+，并安装 NumPy。Windows 官方 Python 安装包默认包含 Tkinter。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m meshwork.cli check-ports --spec samples\sample_spec.json --log-file artifacts\ports.jsonl
python -m meshwork.cli run --spec samples\sample_spec.json --requests samples\requests.json `
  --ledger artifacts\ledger.db --snapshot-dir artifacts\snapshots --revision 1 --reset `
  --log-file artifacts\workbench.jsonl

python -m meshwork.cli ui --spec samples\sample_spec.json --requests samples\requests.json `
  --ledger artifacts\ui.db --snapshot-dir artifacts\ui-snapshots --revision 1
```

在项目根目录执行。`check-ports` 会真实尝试绑定样本中声明的 4 个端口；若被占用会返回 `E_PORT`。

## 目录

- `meshwork/model.py`：声明加载与强校验，错误字段和 JSON 位置。
- `meshwork/selector.py`：显式累计权重区间与 SHA-256 稳定位置选择。
- `meshwork/breaker.py`：CLOSED / OPEN / HALF_OPEN 熔断状态机，真实单调时钟冷却。
- `meshwork/budget.py`：全局与每服务双重预算、优先级等待与单调等待时钟。
- `meshwork/protocol.py`：真实 TCP JSON 行协议、握手、应答字段校验和套接字超时。
- `meshwork/upstream_manager.py`：Windows/POSIX 子进程编排、端口预检与生命周期清理。
- `meshwork/ledger.py`：SQLite 尝试/结局/预算事务账本、整数格式版本、内容摘要和恢复态重建。
- `meshwork/snapshot.py`：快照版本、摘要、整体替换与稳定差异列表。
- `meshwork/stats.py`：整数定点成功率、P95、吞吐、误差界和稳定决胜。
- `meshwork/engine.py`：并行分发、重试、熔断、预算、账本提交和快照状态。
- `meshwork/replay.py`：逐请求、逐尝试一致性比较与首个差异定位。
- `meshwork/viewmodel.py` / `meshwork/ui.py`：只读诊断视图模型与 Tkinter 界面。
- `meshwork/cli.py`：`validate`、`check-ports`、`run`、`recover`、`replay`、`ui` 命令。
- `scripts/upstream_instance.py`：独立本地上游模拟器，按整数种子和声明参数确定性产出结果。
- `samples/`：服务声明、普通流量、熔断、重放、崩溃恢复请求样本。
- `demo/run_demos.py`：顺序执行三场景演示并写出 `artifacts/demo/verification.json`。
- `tests/`：36 项自动化测试。

## 声明格式

`samples/sample_spec.json` 是完整样例。核心结构：

```json
{
  "schema_version": 1,
  "session": {
    "scale": 1000000,
    "global_concurrency": 4,
    "attempt_timeout_ms": 150,
    "retry_backoff_ms": [4, 12],
    "breaker": {
      "failure_threshold": 2,
      "cooldown_ms": 180,
      "half_open_probes": 1
    }
  },
  "services": [
    {
      "id": "checkout",
      "quota": 2,
      "instances": [
        {
          "id": "checkout-a",
          "host": "127.0.0.1",
          "port": 47321,
          "weight": 3,
          "seed": 101,
          "behavior": {
            "fast_ms": 5,
            "slow_ms": 60,
            "success_percent": 85,
            "slow_percent": 10,
            "error_percent": 5,
            "rules": []
          }
        }
      ]
    }
  ],
  "route_revisions": [
    {
      "revision": 1,
      "description": "balanced baseline",
      "rules": [
        {"service": "checkout", "weights": {"checkout-a": 3}}
      ]
    }
  ]
}
```

校验规则包括：服务与实例标识唯一；`host` 必须是字面量 `127.0.0.1`；端口全局唯一；权重为非负整数；每个服务至少一个正权重；修订号在声明顺序中严格递增；退避元素为非负整数；超时、配额和并发上限为正整数；行为百分比之和为 100。任何非法声明都会以 `E_SPEC` 返回 `field` 与 `location`。

路由修订中的 `weights` 是有效权重覆盖：规则内的实例可省略，省略即权重 0；若服务没有规则或 `weights` 缺失，则使用实例自身声明权重。`--revision` 未指定时使用最高修订。

请求文件格式：

```json
{"requests": [{"id": "r-001", "service": "checkout", "payload": {"cart": "c-1"}}]}
```

## 确定性选择与调度

选择算法不依赖字典遍历、Python `hash`、请求到达顺序或墙钟：

1. 实例按声明顺序展开为累计权重区间 `[start, end)`。
2. 对 `service + request_id + attempt_no` 计算 SHA-256，取前 8 字节作为无符号整数。
3. 对总权重取模得到位置，落入第一个 `start <= position < end` 的区间。
4. 零权重实例区间为空，永不会被选中。

每次尝试先经过熔断判定，再加锁申请全局与每服务预算。预算槽获得后才写入“未提交意图”并启动真实套接字连接。重试退避在释放预算之后执行，不占用并发槽；尝试次数为 `1 + len(retry_backoff_ms)`，耗尽后终结码为 `E_EXHAUST`。

熔断阈值为连续失败次数。OPEN 期间真实单调时钟冷却，选择到该实例时快速失败且不申请预算；冷却结束后进入 HALF_OPEN，按 `half_open_probes` 放行；探测成功关闭，失败重新计时 OPEN。

## 账本、快照与恢复

SQLite 账本格式版本为整数 `1`，并在每次写事务后重新计算内容 SHA-256 摘要。一次尝试完成时，以下写入在同一事务内提交：

- `attempts`：尝试轨迹、结局、实例选择序号、规则修订和熔断后状态。
- `budget_usage`：全局/服务预算槽、限制、等待与运行时间。
- `outcomes`：请求最终结局（仅最终尝试）。
- 删除 `attempt_intents`、更新 `requests` 与账本摘要。

运行中的套接字尝试在 `attempt_intents` 中留有意图。进程被强制终止后，重启会将遗留意图写入 `unknown_attempts` 并标记请求待重跑；预算与熔断只从已完整提交的 `attempts` 重建，且报告固定包含 `exit_report_available=false` 和 `pre_crash_exit_report=null`，不伪造崩溃前退出报告。

快照持久化规则修订号、实例集合版本、预算已用量、熔断状态、已完成尝试游标、账本格式版本和账本摘要。重新打开时若格式、摘要、规则修订或实例版本不一致，会以 `E_SNAPSHOT` 记录并按账本权威状态整体替换；差异以稳定排序列出 `added`、`removed`、`changed` 和 `scalar_changed`。

完整重放会在全新账本中重新启动真实上游并分发同一请求序列，比较请求服务、最终结局、尝试次数、每次尝试的实例、选择序号、结局、错误码、预算使用标志、规则修订和上游声明延迟。首个差异以 `E_REPLAY` 返回，定位到 `request_id`、`service`、`field` 和尝试序号。

## 整数定点 SLO

会话缩放因子 `S = session.scale`。所有非负比例与定点值使用向下截断（floor）：

- 成功率定点：`successes * S // requests`；单值量化误差上界为 1 个缩放单位。
- P95：nearest-rank，秩为 `ceil(0.95 * n)`，再乘 `S`；延迟来源先量化为整数毫秒，固定点单值误差上界为 `S` 个缩放单位（即 1 ms）。
- 吞吐：`requests * 1000 * S // observed_window_ms`，单位为“请求/秒 × S”；窗口采用整数毫秒，定点除法误差上界为 1 个缩放单位。
- 两服务差值比较误差界为各单值误差上界之和：比例 2、P95 为 `2S`、吞吐 2。

差值落在误差界内时按服务标识升序稳定决胜，并在排名和两两比较结果中标注 `within_error_bound`。统计输入按稳定键排序，聚合与请求到达顺序无关。

## 错误码

| 代码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明、请求或账本配置不匹配 |
| `E_PORT` | 回环端口被占用或无法绑定 |
| `E_HOST` | 子进程启动、连接或握手失败；套接字层主机错误 |
| `E_REPLY` | 握手或应答缺少必需字段、字段非法或不匹配 |
| `E_TIMEOUT` | 单次真实套接字超时（尝试级） |
| `E_UPSTREAM` | 上游声明返回错误结果（尝试级） |
| `E_CIRCUIT_OPEN` | 熔断冷却或半开额度耗尽的快速失败 |
| `E_EXHAUST` | 重试耗尽后的请求最终错误码 |
| `E_SNAPSHOT` | 快照/账本版本、摘要或状态不一致 |
| `E_REPLAY` | 重放首个语义不一致 |
| `E_INTERNAL` | 未分类内部错误 |

## 异常注入

注入只改变临时输入文件或进程环境，不修改源码，随后仍走正常校验/检测路径：

```powershell
# 复制声明并把第二个修订号改坏，正常加载器返回 E_SPEC
python -m meshwork.cli run --spec samples\sample_spec.json --requests samples\requests.json `
  --ledger artifacts\inject-spec.db --reset --inject corrupt-rules `
  --log-file artifacts\inject-spec.jsonl

# 工作台进程先占用首个声明端口，正常预检返回 E_PORT
python -m meshwork.cli run --spec samples\sample_spec.json --requests samples\requests.json `
  --ledger artifacts\inject-port.db --reset --inject port-occupied `
  --log-file artifacts\inject-port.jsonl

# 生成快照后修改 payload 而保留旧摘要，正常加载返回 E_SNAPSHOT
python -m meshwork.cli run --spec samples\sample_spec.json --requests samples\requests.json `
  --ledger artifacts\inject-snapshot.db --snapshot-dir artifacts\inject-snapshots --reset `
  --inject snapshot-digest --log-file artifacts\inject-snapshot.jsonl
```

## 恢复与重放命令

```powershell
python -m meshwork.cli recover --spec samples\sample_spec.json --ledger artifacts\ledger.db `
  --snapshot-dir artifacts\snapshots --log-file artifacts\recover.jsonl

python -m meshwork.cli replay --spec samples\sample_spec.json --source-ledger artifacts\ledger.db `
  --replay-ledger artifacts\replay.db --snapshot-dir artifacts\replay-snapshots `
  --log-file artifacts\replay.jsonl
```

## 界面

`python -m meshwork.cli ui` 打开 Tkinter 诊断界面，展示：

- 服务、实例、声明权重、规则有效权重、选择次数和最近选择；
- 每个实例熔断状态、连续失败、阈值、半开试探和剩余冷却；
- 灰/彩色并发时间线，标注全局与服务在飞数、预算等待和尝试结果；
- 成功率、P95、吞吐、结局分布、定点公式、单值误差和比较误差界；
- 账本格式/内容摘要、实例集合版本、规则修订和快照列表。

`Refresh` 只执行 `SELECT` 和内存计算；界面显示本次刷新前后的账本摘要是否一致。点击 `Reset + Run sample` 才会替换账本并真实运行。视图模型的刷新摘要稳定性有自动化测试。

## 测试与演示

```powershell
python -m unittest discover -v tests
python demo\run_demos.py
```

演示脚本在独立进程中顺序执行：

1. 冷启动 18 个请求的权重分发与双预算并发。
2. 熔断触发、真实冷却、快速失败、半开探测恢复。
3. 快照恢复/版本替换/摘要不符、强制终止恢复、完整重放与 `E_REPLAY` 篡改定位。

实际测试结果与发展环境记录见 `VERIFICATION.md`，更深层的实现说明见 `TECHNICAL.md`。

## 本地安全边界

- 所有监听和连接均只使用字面量 `127.0.0.1` 与声明的高位端口。
- 上游脚本只接受 JSON 行协议，不访问外部网络。
- 不使用 Docker、容器、云服务、账号或凭据。
- 进程清理使用每个上游独立的 stdin 看门狗；父工作台被强制终止后，子进程在管道 EOF 时退出，避免遗留端口。
