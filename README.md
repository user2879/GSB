# 推理网关路由诊断台

这是一个面向 Windows 本地环境的确定性推理网关路由诊断组件。它以本地 JSON/NDJSON 为领域数据，以真实 `127.0.0.1` 子进程为推理后端，提供按客户端键稳定分流的灰度路由、NumPy 前向推理、配额调度、SQLite 权威账本、快速视图/权威重算逐字节校验，以及强制终止后的恢复重放。

## 交付内容

- `gateway_diag/`：Python 包，包含数据校验、参数构造、路由、子进程协议、调度、账本、恢复、CLI 和 Tkinter 操作台。
- `examples/`：可直接运行的后端/规则规格与 NDJSON 请求样例。
- `tests/`：39 项自动化验收测试，覆盖真实回环子进程、并发配额、逐字节一致性、版本替换、强制终止恢复与异常码。
- `docs/TECHNICAL.md`：算法、事务、协议和一致性设计说明。

## 环境准备（Windows）

要求 Windows 10/11，Python 3.11 或 3.12，并且 `python` 命令位于 `PATH`。

```powershell
cd <项目目录>
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

也可在项目根目录直接以 `python -m gateway_diag ...` 运行，不要求先安装包；安装后可使用 `gateway-diag ...` 入口。

应用只连接 `127.0.0.1`，不访问外部网络。若 Windows 首次弹出防火墙提示，允许专用网络上的 Python 进程监听/连接回环高位端口即可。所有后端主机字段必须严格为字面量 `127.0.0.1`。

## 领域数据

### 规格 JSON

```json
{
  "revision": 1,
  "limits": {
    "global_concurrency": 2,
    "backend_concurrency": {
      "stable-a": 1,
      "canary-b": 1
    }
  },
  "backends": [
    {
      "id": "stable-a",
      "host": "127.0.0.1",
      "port": 43171,
      "model_dim": 4,
      "hidden_dim": 3,
      "seed": 17,
      "quantization": "int8"
    },
    {
      "id": "canary-b",
      "host": "127.0.0.1",
      "port": 43172,
      "model_dim": 4,
      "hidden_dim": 5,
      "seed": 29,
      "quantization": "int4"
    }
  ],
  "rules": [
    {
      "id": "alpha",
      "prefix": "alpha:",
      "stable_backend": "stable-a",
      "canary_backend": "canary-b",
      "canary_weight_bp": 3000,
      "batch_size": 2,
      "version": 1
    }
  ]
}
```

约束：

- `revision` 是规格整体版本，修改任意后端或规则时必须严格递增。
- `backends[].id` 与 `rules[].id` 分别唯一；全部端口在规格内唯一且范围为 `1..65535`。
- `host` 必须为 `127.0.0.1`；缺省时也取该值。
- `model_dim`、`hidden_dim`、`batch_size`、规则 `version` 为正整数；`seed` 为整数；布尔值不会被当作整数。
- `quantization` 仅支持 `fp32`、`int8`、`int4`。
- `canary_weight_bp` 为 `0..10000` 的整数。
- 每条规则的两个目标后端必须存在。
- `limits.global_concurrency`、`limits.default_backend_concurrency` 与每个后端配额均为正整数。缺省全局/单后端并发均为 `1`。
- 修改同一规则内容时，该规则 `version` 必须严格递增；新增/删除条目及后端内容变化通过规格 `revision` 递增表达。

### 请求 NDJSON

```json
{"request_id":1,"client_key":"alpha:customer-001","input":[0.1,-0.2,0.3,0.4],"arrival_ms":1000}
```

- `request_id` 为唯一正整数；`client_key` 为非空字符串；`input` 为长度等于实际路由后端 `model_dim` 的有限数值数组；`arrival_ms` 为不小于 0 的整数。
- 规则选择：可选请求字段 `prefix`、`request_prefix`、`path` 或 `rule_id` 用于匹配。否则将 `client_key` 当作待匹配文本，选择最长匹配的规则前缀。只有一条规则且无匹配时，将其作为该规则的显式回退。
- 请求按 `(arrival_ms, request_id)` 排序，因此文件行顺序和字典遍历顺序不影响结果。

## 运行

```powershell
python -m gateway_diag validate --spec examples\sample_spec.json --requests examples\sample_requests.ndjson
python -m gateway_diag run --spec examples\sample_spec.json --requests examples\sample_requests.ndjson --db artifacts\ledger.sqlite3 --report artifacts\run.json
python -m gateway_diag preview --db artifacts\ledger.sqlite3
python -m gateway_diag preview --db artifacts\ledger.sqlite3 --request-id 1
python -m gateway_diag verify --mode both --spec examples\sample_spec.json --requests examples\sample_requests.ndjson --db artifacts\ledger.sqlite3
python -m gateway_diag recover --spec examples\sample_spec.json --requests examples\sample_requests.ndjson --db artifacts\ledger.sqlite3
python -m gateway_diag versions --db artifacts\ledger.sqlite3
python -m gateway_diag gui --spec examples\sample_spec.json --requests examples\sample_requests.ndjson --db artifacts\ledger.sqlite3
```

安装后也可使用 `gateway-diag ...`。样例端口 `43171/43172` 被占用时会按规格返回 `E_PORT`，请改用本机空闲高位端口。

### 交互操作台

`gui` 子命令打开 Tkinter 桌面应用，包含：

- 规格 JSON 编辑、保存与强校验。
- 后端/规则结构化表格，支持新增、双击编辑、删除；结构化修改会自动递增规格 revision，规则内容修改还会自动递增规则 version。
- NDJSON 请求流编辑器，保存前执行完整校验。
- 执行/续跑、快速视图、权威与重放校验、恢复、版本历史和日志面板。
- 快速视图单次读取缓存后立即展示去向、分桶、权重版本、批归属、输出摘要和批次耗时；双击可查看详情。

## 一致性模型

- 稳定分桶：`SHA-256("gateway-route-v1\0" + rule_id + "\0" + client_key)`，取前 8 字节按大端整数解释后 `mod 10000`，得到 `0..9999`；当 `bucket < canary_weight_bp` 时去灰度版，否则去稳定版。权重 `0` 全稳定，权重 `10000` 全灰度。
- 批次归属：按 `(rule_id, backend_id)` 分组，以 `(arrival_ms, request_id)` 稳定排序，再按规则 `batch_size` 切块。批次 ID 由 revision、规则/后端摘要和组内序号确定。
- 参数：以 SHA-256 计数器流生成浮点矩阵，按 `fp32`、对称 `int8` 或对称 `int4` 量化；参数摘要绑定后端标识、维度、种子和量化口径版本。
- 权威重算会重新执行分流、批次规划和真实推理，输出摘要必须与账本一致。单调时钟耗时是已提交的不可变观测事实，权威统计从账本耗时事实重算并再次逐字节比较，不把无法复现的新一次墙钟耗时误当成同一事实。
- 快速视图与权威视图都以规范 JSON 序列化字节比较；首个差异报告为 `E_PREVIEW`，包含请求序号与字段。

## 持久化与恢复

- SQLite 开启 WAL、`synchronous=FULL`、`foreign_keys=ON`。
- 每完成一个批次，批次记录、请求账本、回执摘要、参数摘要、配额消耗、进度游标和账本内容摘要在一个 `BEGIN IMMEDIATE` 事务中提交。
- 重启后，规划中缺失的批次写入 `recovery_unknowns`，状态为 `UNKNOWN`，报告 `needs_rerun=true`；重跑成功后在批次提交事务中改为 `RESOLVED`。
- `recover` 只跳过完整提交的批次，重跑缺失批次，然后从真实后端重新执行分流与推理，与账本逐项比较；首个差异报告 `E_REPLAY`，包含请求序号、后端和字段。
- 新规格 revision 激活时，旧 revision 的批次与请求账本整体标记 `stale=1`，不计入当前快速视图和汇总。当前视图整体替换，并持久化稳定的新增/删除/版本变更清单。

## 错误码

| 错误码 | 含义 | CLI 退出码 |
|---|---|---:|
| `E_SPEC` | 规格/请求首个非法条目或字段 | 2 |
| `E_PORT` | 后端端口占用 | 3 |
| `E_HANDSHAKE` | 启动就绪行或 TCP 握手超时/不匹配 | 4 |
| `E_RECEIPT` | 回执缺失、字段非法或批内序号不匹配 | 5 |
| `E_PREVIEW` | 快速视图与权威重算不一致 | 6 |
| `E_REPLAY` | 恢复重放与账本不一致 | 7 |
| `E_SCHEDULER` | 调度器无法满足配额 | 8 |
| `E_LEDGER` | 账本事务/内容摘要异常 | 9 |
| `E_UI` | Tkinter/Tk 显示环境不可用（补充错误码） | 1 |

错误 JSON 同时给出可用的 `entry`、`field`、`request_id`、`backend_id`、`port`、`batch_id` 定位信息。

## 本次实际验证

执行环境：Linux/WSL2，Python `3.11.0rc1`，NumPy `1.26.4`；Windows 专属启动参数与 `SO_EXCLUSIVEADDRUSE` 分支已实现，但当前容器没有 Windows 实机，Windows 分支仍需在目标机执行同一测试命令复核。

```text
python -m unittest discover -s tests -v
Ran 39 tests in 31.144s
OK
```

已验证：真实回环握手/推理、同端口重启、端口占用 `E_PORT`、握手缺失 `E_HANDSHAKE`、回执缺字段 `E_RECEIPT`、稳定分桶、参数跨进程逐位一致、真实线程池并发上限与单后端配额、快速视图/权威视图规范字节一致、`E_PREVIEW` 字段定位、`E_REPLAY` 字段定位、revision 整体替换与陈旧隔离，以及运行进程在至少一个批次提交后被 `SIGKILL`、重启标记 `UNKNOWN`、重跑缺失批次并成功重放。

运行目录中的样例端口可能处于系统 `TIME_WAIT` 或被占用；测试全部使用动态空闲端口，因此不受固定样例端口影响。

更详细的逐项实跑记录见 `docs/VERIFICATION.md`。
