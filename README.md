# 云原生集群租约持有权水位组件

这是一个面向本地 Windows 环境的确定性集群租约仲裁组件。它为每个声明集群启动独立 Python 子进程，仅绑定 `127.0.0.1` 回环端口，通过 JSON Lines 协议裁决申请、续约和主动释放；当前裁决、有效租约、水位与进度持久化在 SQLite 中。

核心性质：

- 当前参数版本下，同一 `(cluster_id, event_seq)` 只能有一条有效裁决。
- 同一事件序列在 `logical` 时钟模式下产生逐字节一致的裁决回执与账本摘要。
- 容量满时只允许严格更高优先级的持有者以稳定规则接管最低优先级租约。
- 每批裁决、有效租约、水位和进度在同一 SQLite 事务中提交。
- 参数演进、强制终止恢复和快照恢复采用整体视图替换或提交游标恢复，不做增量猜测。
- Tkinter 诊断界面以只读连接打开 SQLite，刷新不会改写账本。

## 环境准备（Windows）

1. 安装 Python 3.11 或以上，官方安装包默认包含 `tkinter`。
2. 安装 NumPy：`py -3.11 -m pip install -r requirements.txt`（等价于安装 `numpy`）。
3. 确认 tkinter：`py -3.11 -c "import tkinter; print(tkinter.TkVersion)"`。
4. 检查样本端口是否被占用：

   ```powershell
   Get-NetTCPConnection -LocalPort 39171,39172 -ErrorAction SilentlyContinue
   ```

5. 若被其他本地进程占用，先关闭该进程再运行。组件不会访问非回环地址。

SQLite 使用 Python 标准库 `sqlite3`，不需要额外驱动。项目不使用 Docker 或其他容器。

## 快速运行

```powershell
python -m cluster_lease validate --clusters samples/clusters.json --holders samples/holders.json --events samples/events.ndjson

python -m cluster_lease run --clusters samples/clusters.json --holders samples/holders.json --events samples/events.ndjson --db build/state.db --runtime-dir build/runtime --batch-size 4 --clock-mode logical --snapshot-out snapshots/session-v1.json

python -m cluster_lease snapshot-open --clusters samples/clusters.json --holders samples/holders.json --events samples/events.ndjson --db build/state.db --snapshot snapshots/session-v1.json

python -m cluster_lease evolve --clusters samples/clusters_evolved.json --holders samples/holders_evolved.json --events samples/events.ndjson --db build/state.db

python -m cluster_lease recover --clusters samples/clusters_evolved.json --holders samples/holders_evolved.json --events samples/events.ndjson --db build/state.db

python -m cluster_lease snapshot-create --clusters samples/clusters_evolved.json --holders samples/holders_evolved.json --events samples/events.ndjson --db build/state.db --out snapshots/session-v2.json

python -m cluster_lease inspect --db build/state.db

python -m cluster_lease gui --db build/state.db
```

每个命令均可使用 `--log logs/command.jsonl` 写结构化 JSONL 日志，并用 `--report artifacts/report.json` 落盘最终 JSON。

## 输入格式

集群声明支持根数组或 `{"clusters": [...]}`：

```json
{
  "clusters": [
    {
      "cluster_id": "alpha",
      "host": "127.0.0.1",
      "port": 39171,
      "lease_duration_ms": 5000,
      "renewal_interval_ms": 1000,
      "capacity": 1
    }
  ]
}
```

持有者声明支持根数组或 `{"holders": [...]}`：

```json
{
  "holders": [
    {"holder_id": "h-high", "priority": 9, "targets": ["alpha"]}
  ]
}
```

事件流为 NDJSON，序号从 1 连续递增，时间戳单调不降：

```json
{"event_seq":1,"timestamp_ms":0,"cluster_id":"alpha","holder_id":"h-high","event_type":"apply"}
{"event_seq":2,"timestamp_ms":1000,"cluster_id":"alpha","holder_id":"h-high","event_type":"renew"}
{"event_seq":3,"timestamp_ms":1100,"cluster_id":"alpha","holder_id":"h-high","event_type":"release"}
```

`event_type` 接受 `apply/renew/release`，也接受中文标签 `申请/续约/主动释放`。字段定义详见 `docs/DESIGN.md`。

正常规格错误统一以 `E_SPEC` 拒绝整个加载过程，并给出首个违规字段，例如 `clusters[1].port`。三类受控注入使用额外业务码：`E_REF`、`E_TIMING`、`E_SNAPSHOT`。

## 状态机与时钟

租约状态为 `granted -> renewed -> released`，未续约时进入 `expired`。到达 `expires_ms` 的租约在服务处理下一个请求时先回收到期租约，再裁决当前请求。容量已满时，严格更高优先级的申请者可接管当前最低优先级租约；同优先级不抢占。

时钟模式：

- `logical`：只使用事件时间戳，适用于确定性重放和验收。
- `hybrid`：事件时间或服务端单调时钟任一到期即回收，是 `run` 默认值。
- `real`：只使用服务端单调时钟。

快照重放和参数演进重放强制使用 `logical`。样本租约时长为 5000 ms，普通本地运行不会触发混合时钟误判；真实计时路径另有自动化测试。

## 账本、演进与恢复

SQLite 账本包含参数版本、裁决、当前有效租约、持有者持有权起止区间、逐事件水位、进度游标、批运行记录和快照目录。水位按固定整数缩放因子 `1_000_000` 表示，截断方向为 `floor`；每条记录显式携带缩放因子和截断方向。

裁决摘要由集群标识、事件序号、裁决结论、原因和稳定排序的有效租约集合按规范 JSON 计算。账本整体摘要不依赖字典插入顺序或 Python 哈希遍历顺序。

批处理流程：

1. 先提交 `inflight` 批标记。
2. 向对应集群子进程逐行发送请求。
3. 在同一个 SQLite 事务中提交该批裁决、最终有效租约、逐事件水位、进度游标和 `committed` 批状态。

强制终止后，遗留 `inflight` 批次会被标记为 `unknown` 并要求重跑；恢复只使用已完整提交的游标，不生成“强制终止时退出报告”。已持有租约按账本记录的租约时长重算到期时间，报告中区分 `true_renewal_continuation` 与 `recovery_recomputed_continuation`。

验收脚本可通过 `--crash-after-inflight-batch N` 注入强制终止；这是测试用故障入口，不改变业务源码。

参数演进比较容量、租约时长、续约间隔、端口及接管优先级摘要。受影响集群参数版本加一，旧裁决/水位/有效租约整体标记为陈旧，然后按新参数完整重放事件并原子替换当前视图。

快照携带整数格式版本、参数版本、事件流摘要、有效租约、水位、裁决账本和内容摘要。打开时先校验快照自身摘要，再比较当前会话；不一致时整体替换当前视图并返回 `E_SNAPSHOT`。随后用快照参数版本和游标重新驱动子进程，逐事件比较裁决与水位；首个差异以 `E_REPLAY` 定位到集群、事件序号和字段。

## 异常注入

```powershell
python -m cluster_lease validate ... --inject missing-ref   # E_REF
python -m cluster_lease validate ... --inject timing        # E_TIMING
python -m cluster_lease snapshot-open ... --inject snapshot # E_SNAPSHOT
```

注入只修改载入内存的输入参数，不改源码或磁盘输入文件。所有错误同时写入 `--log` 指定的结构化日志。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_SPEC` | 输入规格不合法，整体拒绝加载 |
| `E_REF` | 持有者目标集群引用缺失注入 |
| `E_TIMING` | 租约时长与续约间隔关系错误注入 |
| `E_PORT` | 回环端口被占用 |
| `E_HANDSHAKE` | 子进程握手真实套接字超时 |
| `E_REPLY` | 回执字段缺失、格式错误或超时 |
| `E_SNAPSHOT` | 快照摘要/参数版本/会话内容不一致 |
| `E_REPLAY` | 快照恢复后重放与账本不一致 |
| `E_STATE` | 数据库状态与命令不匹配 |
| `E_ENV` | 本地运行环境不满足要求，例如缺少 tkinter/显示 |

## 测试与实测

```powershell
python -m unittest discover -s tests -t . -v
python scripts/run_acceptance.py
```

当前工作区实际结果：

- 8 项自动化测试全部通过，覆盖规格校验、注入、状态机、真实时钟到期、端口/握手/回执错误、确定性、演进、快照重放、崩溃恢复。
- 端到端验收 13 项全部通过；两次全新运行的裁决回执摘要均为 `82dc9afc080900e26ab798867b29a48d2e81002b792a81ee5b0725f04fe97db0`，初始账本摘要均为 `2ca12e136d69a92f7430447879af4f45e6402ae2892013c886be565e4a5616eb`。
- 当前测试容器是 Linux Python 3.11.0rc1 且未安装 tkinter；核心功能、SQLite、TCP 子进程已实测，GUI 未在该容器渲染，仅验证了缺 tkinter 时返回 `E_ENV`。Windows 官方 Python 自带 tkinter，可直接运行 `gui` 子命令。

详细协议、事务和校验规则见 `docs/DESIGN.md`，实测命令与输出摘要见 `docs/VALIDATION.md`。
