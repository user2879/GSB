# 本地推理网关路由增量更新与依赖失效诊断台

这是一个仅使用本地回环网络的确定性诊断工具。它可以：

- 强校验推理后端、路由规则、规则依赖与 NDJSON 请求流；
- 用稳定哈希完成稳定版/灰度版分流，并真实调用 `127.0.0.1` 模拟推理后端；
- 规则或后端参数变化时，沿依赖图计算完整失效闭包；
- 只重算失效规则，保留未失效规则的回执 JSON 字节和回执版本号；
- 将增量结果与从零全量重建逐规则、逐批、逐字段比较；
- 用 SQLite 事务账本支持强制终止后的恢复、重放和独立验证；
- 注入 `E_SPEC`、`E_REF`、`E_SNAPSHOT` 三类故障；
- 提供 CLI 和一个可选的 Windows tkinter 面板。

整个系统不使用 Docker、外部网络或远程服务。模拟后端只绑定 `127.0.0.1`。

## 环境

- Python 3.11+
- 标准库 `sqlite3`
- `numpy`
- 可选：标准库 `tkinter`，仅 GUI 需要

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

本工作区实际验证环境：Linux/WSL 下的 Python 3.11.0rc1、NumPy 1.26.4、SQLite 3.37.2。该环境缺少 Tk 动态库，所以这里没有运行 GUI；Windows 官方 Python 安装包默认包含 Tk，可直接运行 GUI。CLI、SQLite、回环调用和全部测试均已实际执行。

## 目录

```text
gateway_diag/                 核心实现与 CLI
  loaders.py                  JSON/NDJSON 加载和 E_SPEC/E_REF 校验
  graph.py                    依赖图、环检测、稳定拓扑与失效传播
  routing.py                  稳定哈希和派生灰度目标解析
  backend_server.py           确定性 NumPy 回环后端
  backend_client.py           JSONL 客户端
  engine.py                   稳定组批、回执生成
  ledger.py                   SQLite 事务账本
  snapshot.py                 会话快照与一致性检查
  workflow.py                 首次执行、增量更新、崩溃恢复与重放
  compare.py                  增量 vs 全量比较
  verify.py                   只读账本的独立验证器
  inject/cli/gui              CLI、故障注入和可选面板
mock_backend.py               可直接运行的回环脚本入口
samples/base.json             基线声明
samples/update_rule.json      规则权重变更
samples/update_backend.json   后端种子/隐藏层维度变更
samples/update_batch.json     批大小变更
samples/requests.ndjson       48 条确定性请求
tests/test_diagnostics.py     校验、端到端、崩溃、重放、异常码测试
```

## 声明格式

`samples/base.json` 是一个同时包含后端和规则的组合文件：

```json
{
  "schema_version": 1,
  "batch_limit": 4,
  "backends": [
    {
      "id": "blue",
      "host": "127.0.0.1",
      "port": 29470,
      "model_dim": 6,
      "hidden_dim": 5,
      "seed": 101,
      "quantization": "fp32"
    }
  ],
  "rules": [
    {
      "id": "chat",
      "request_prefix": "/chat",
      "stable_backend": "blue",
      "canary_backend": "green",
      "canary_weight_bp": 5000,
      "batch_size": 2,
      "version": 3,
      "depends_on": []
    },
    {
      "id": "chat_cascade",
      "request_prefix": "/chat/cascade",
      "stable_backend": "blue",
      "canary_backend": {
        "from_rule": "chat",
        "select": "conclusion"
      },
      "canary_weight_bp": 6500,
      "batch_size": 2,
      "version": 2
    }
  ]
}
```

字段约束：

- 后端 `id` 全局唯一，`host` 必须是字面量 `127.0.0.1`，`port` 在 `1..65535` 且全部后端间唯一。
- `model_dim`、`hidden_dim` 为正整数。
- `seed` 为非负整数。
- `quantization` 仅支持 `fp32`、`fp16`、`int8`。
- 规则 `id` 唯一，`canary_weight_bp` 在 `0..10000`。
- `batch_size` 为不大于顶层 `batch_limit` 的正整数。
- `version` 为非负整数。
- 后端引用、派生规则引用必须存在。
- 依赖图必须无环。
- `depends_on` 可以显式补充依赖；`canary_backend.from_rule` 本身也会形成依赖边。
- 派生选择器：
  - `conclusion`：取被引用规则对同一客户端键得到的目标后端；
  - `stable_target`：取被引用规则的稳定版后端；
  - `canary_target`：取被引用规则的灰度版后端。
- 被引用的规则必须已声明 `batch_size`。

也可用 `--backends backends.json --rules rules.json` 分开加载。组合文件必须有顶层 `batch_limit`（也接受 `max_batch_size`）。

NDJSON 每行一条：

```json
{"request_id":"req-0001","arrival_seq":1,"client_key":"client-007","prefix":"/chat","vector":[0.1,0.2,0.3,0.4,0.5,0.6]}
```

强校验包括：

- `request_id`、`arrival_seq` 唯一；
- `vector` 长度等于该请求实际去向后端的 `model_dim`；
- 所有数值有限；
- 每行合法，否则整体拒绝，不部分加载；
- 相同完整前缀时按最长前缀选择规则，再按规则 ID 稳定打破平局。

首条非法声明或请求会以 `E_SPEC` 整体拒绝，并记录 `entry` 和 `field`。若引用目标缺失，统一以 `E_REF` 报告全部引用者；若有环，以 `E_CYCLE` 报告从重复起点闭合的环序列。

## 稳定分流

规则 `r`、客户端键 `k` 的桶值：

```text
bucket = first64be(
  SHA-256("gateway-route-v1" NUL r NUL k)
) mod 10001
```

桶范围为 `0..10000`：

- 权重 `0` 永远稳定版；
- 权重 `10000` 永远灰度版；
- 其他权重下，`bucket < canary_weight_bp` 去灰度版。

同一客户端键、同一规则和同一权重始终得到同一位图。`/chat/cascade` 等更长前缀优先于 `/chat`。

## 回环推理后端

单独启动：

```bash
python mock_backend.py --config samples/base.json --allow-dynamic
```

或让 CLI 临时启动并在退出时关闭：

```bash
python -m gateway_diag run --config samples/base.json --requests samples/requests.ndjson --workdir .work/demo --spawn-backends
```

协议为 TCP 上每行一个 JSON：

```json
{"op":"infer","protocol":"infer.v1","backend_id":"blue","params":{"model_dim":6,"hidden_dim":5,"seed":101,"quantization":"fp32"},"requests":[{"request_id":"req-0001","vector":[0.1,0.2,0.3,0.4,0.5,0.6]}]}
```

后端以 NumPy、声明种子和维度生成确定性参数，执行前向推理，返回批次 `output_digest`。`fp16`/`int8` 会对参数做确定性量化；输出按 12 位科学计数法序列化后摘要。后端仅监听声明中的回环端口。

`--spawn-backends` 会在启动前实际尝试绑定所有端口。若端口被其他进程占用，会以 `E_BACKEND` 报告。需要改端口时直接修改 JSON 的 `port` 后立即重新加载。脚本带父进程看门狗，父进程被强制终止后会自动释放端口。

## 常用命令

首次运行或加载新声明：

```bash
python -m gateway_diag --pretty run \
  --config samples/base.json \
  --requests samples/requests.ndjson \
  --workdir .work/demo \
  --spawn-backends
```

同一工作目录加载更新声明（只重算失效规则）：

```bash
python -m gateway_diag --pretty run \
  --config samples/update_rule.json \
  --requests samples/requests.ndjson \
  --workdir .work/demo \
  --spawn-backends
```

增量与从零全量重建校验：

```bash
python -m gateway_diag --pretty validate \
  --before samples/base.json \
  --after samples/update_rule.json \
  --requests samples/requests.ndjson \
  --workdir .work/validate \
  --spawn-backends
```

独立验证账本（只读 SQLite，不修改账本）：

```bash
python -m gateway_diag --pretty verify \
  --config samples/update_rule.json \
  --requests samples/requests.ndjson \
  --workdir .work/demo \
  --spawn-backends
```

查看快照：

```bash
python -m gateway_diag snapshot \
  --config samples/update_rule.json \
  --requests samples/requests.ndjson \
  --workdir .work/demo
```

可选 GUI：

```bash
python -m gateway_diag gui \
  --config samples/base.json \
  --requests samples/requests.ndjson \
  --workdir .work/gui
```

GUI 和 CLI 调用同一套实现。

## 失效传播

更新时比较旧目标声明与新声明，直接失效来源包括：

- 规则字段变化：权重、批大小、稳定/灰度目标、派生引用、前缀、版本等；
- 新增规则；
- 后端新增/删除或参数变化；
- 直接引用变化后端的规则。

然后沿“依赖规则 -> 依赖者”闭包传播。输出按稳定拓扑序排列，每一步包含：

- `step`；
- `rule_id`；
- `code: E_STALE`；
- 首个传播依据 `basis`；
- 所有直接依据和上游失效规则 `all_bases`（写入账本）。

`E_STALE` 只存在于传播轨迹和运行报告中，不写入回执。未失效规则的 `receipt_json`、`receipt_digest` 和 `receipt_version` 保持逐字节不变。

失效规则在事务中整体删除旧回执、批次提交记录和旧规则状态，再按稳定顺序整体重写；不会把新旧批回执做增量合并。报告中的 `replacement.rules` 稳定列出替换集合。

## 比较判定

`validate` 在三个独立目录中执行：

1. `baseline`：旧声明的全量构建；
2. `incremental`：复制 baseline 后加载新声明，只重算失效规则；
3. `full`：新声明从空目录全量重建。

判定顺序显式为：

```text
按 rule_id 排序，再按 batch_index 升序
```

逐批比较：

- 去向；
- 目标后端；
- 后端参数摘要；
- 请求 ID 集合；
- 输出摘要；
- 回执版本与回执摘要。

未失效规则还会与 baseline 的原始 `receipt_json` 做字节级比较。首个不一致以 `E_DIVERGE` 报告 `rule_id`、`batch_index` 和 `field`。

由声明更新引起、且被明确归因的后端参数摘要变化会进入 `allowed_differences`，不会被误判为增量/全量不一致；请求集合、去向和批号仍必须一致。比较只依赖排序后的主键，不依赖实际执行顺序。

## SQLite 账本与快照

工作目录默认文件：

```text
ledger.sqlite3
snapshot.json
events.jsonl
```

账本使用整数 `format_version=1` 和整体 `content_digest`。主要表：

- `runs`：运行目标、请求摘要、图摘要和状态；
- `rule_state`：规则状态、回执集摘要、完成集合和批游标；
- `receipts`：每个规则批次一份回执；
- `invalidations`：每代失效传播轨迹；
- `batch_commits`：每批事务提交记录、轨迹摘要、后端参数摘要和游标。

每个批次在同一事务内写回执、轨迹、后端参数摘要、游标及更新后的账本内容摘要，并启用 `synchronous=FULL`。因此崩溃后只信任完整提交的记录。

`snapshot.json` 保存：

- 规则版本集合；
- 后端参数版本集合；
- 依赖图摘要；
- 全量回执摘要；
- 已完成规则；
- 声明、请求和账本内容摘要。

重新打开时会比较快照、账本和当前声明。正常声明变化会按失效集合整体替换对应规则状态；快照与账本内容摘要不一致以 `E_SNAPSHOT` 拒绝。若账本已完成且全部规则均为 `complete`，但进程恰好停在最终事务与快照原子替换之间，下一次 `run` 会从账本重建快照并在报告中标记 `snapshot_repaired=true`。

## 强制终止与恢复

演示：

```bash
# 一个终端启动后端
python mock_backend.py --config samples/base.json --allow-dynamic
# 另一个终端让主进程提交 5 批后直接退出，不打印结果、不写快照
python -m gateway_diag run --config samples/base.json \
  --requests samples/requests.ndjson --workdir .work/crash \
  --crash-after-batches 5
# 退出码为 137
```

重启同一命令且不加 `--crash-after-batches`：

- 已完成规则直接复用；
- 处于 `in_progress` 且未完成的规则先标记为 `unknown`；
- 已提交的部分批次在重跑时逐字段比较；
- 已提交的失效轨迹与重新推导结果比较；
- 任一差异以 `E_REPLAY` 定位到规则、批号和字段；
- 全部重算完成后才写最终快照和完成报告。

恢复报告包含：

```json
{
  "recovered": true,
  "recovery": {
    "marked_unknown_rules": ["chat"],
    "committed_partial_batches": {"chat": [0, 1]},
    "requires_rerun": ["chat"]
  }
}
```

强制终止时 CLI 不声称生成退出报告。

## 故障注入与业务码

生成损坏声明：

```bash
python -m gateway_diag inject --kind declaration \
  --config samples/base.json --output .work/broken.json --workdir .work/inject
python -m gateway_diag run --config .work/broken.json \
  --requests samples/requests.ndjson --workdir .work/e-spec
# E_SPEC
```

生成悬空引用：

```bash
python -m gateway_diag inject --kind reference \
  --config samples/base.json --output .work/dangling.json --workdir .work/inject
python -m gateway_diag run --config .work/dangling.json \
  --requests samples/requests.ndjson --workdir .work/e-ref
# E_REF，details.referrers 列出全部引用者
```

篡改账本内容摘要：

```bash
python -m gateway_diag inject --kind ledger-digest --workdir .work/demo
python -m gateway_diag run --config samples/update_rule.json \
  --requests samples/requests.ndjson --workdir .work/demo
# E_SNAPSHOT
```

也可直接在 `run` 上加：

```bash
--inject spec
--inject ref
--inject ledger
```

注入只创建输入副本或修改账本摘要参数，不改变实现代码。

业务码：

| 码 | 含义 |
|---|---|
| `E_SPEC` | 声明或请求规格非法，整体拒绝 |
| `E_REF` | 引用目标缺失，列出引用者 |
| `E_CYCLE` | 依赖成环，列出环序列 |
| `E_STALE` | 失效传播标记，仅出现在轨迹/报告 |
| `E_DIVERGE` | 增量与全量结果首个字段不一致 |
| `E_SNAPSHOT` | 账本/快照内容摘要或格式不一致 |
| `E_REPLAY` | 恢复重跑与已提交批次/轨迹首个不一致 |
| `E_VERIFY` | 独立验证器发现首个违规规则/字段 |
| `E_BACKEND` | 回环端口、连接或后端协议失败 |
| `E_GUI` | 当前 Python 缺少 tkinter，仅 GUI 入口 |

CLI 业务错误退出码为 `2`；成功为 `0`；未捕获的内部 I/O 错误为 `3`。成功和失败结果都以 JSON 输出。`events.jsonl` 每行包含 `mono_ns`、UTC 时间、级别、事件、业务码、规则标识和字段。

## 独立验证器

`verify` 以只读 URI 打开 SQLite，并：

- 校验整数格式版本和账本内容摘要；
- 校验快照与账本一致；
- 比对声明摘要、请求摘要和依赖图摘要；
- 自行按稳定哈希分流、组批并真实调用后端；
- 自行生成期望回执并验证每份持久化回执自摘要；
- 比较批号、去向、后端参数摘要、请求集合、输出摘要、回执版本；
- 比较规则回执集摘要、规则状态分布和失效轨迹；
- 首个违规以 `E_VERIFY` 报告 `rule_id`、`field`，通常还带 `batch_index`。

## 实际验证结果

以下结果来自本工作区实际执行。

测试：

```text
python -m unittest discover -s tests -v
Ran 11 tests in ~8.8s
OK
```

覆盖：声明顺序无关图摘要、环、悬空引用、向量长度、稳定哈希、样本增量/全量、后端参数变更、前缀新增导致的请求重归属、强制终止恢复、重放篡改、快照修复、三类异常码、独立验证器发现篡改。

48 条样本、仅规则 `chat` 权重变化：

```text
affected rules: chat, chat_cascade
unaffected rules: assist, embed, embed_audit
replacement rules: chat, chat_cascade
unaffected receipt bytes verified: 13
batches compared: 25
incremental vs full: identical
```

后端 `green` 的 seed/hidden_dim 变化：

```text
affected rules: assist, chat, chat_cascade
unaffected rules: embed, embed_audit
unaffected receipt bytes verified: 10
incremental vs full: identical
allowed difference: backend_parameter_change / green
```

独立验证样本工作目录：

```text
result: verified
rules: 5
batches: 25
trace steps: 2
```

强杀测试：

```text
crash exit: 137
stdout bytes: 0
snapshot written: false
before recovery: assist=complete, chat=in_progress, others pending
after recovery: all complete
committed partial batches replayed: chat[0,1]
independent verify: verified
```

引入一条不一致的已提交部分回执后恢复：

```text
code: E_REPLAY
rule_id: chat
batch_index: 0
field: output_digest
```

异常注入实测：

```text
broken JSON      -> E_SPEC
missing backend  -> E_REF, referrers=["assist"]
tampered digest  -> E_SNAPSHOT
tampered receipt -> E_VERIFY, rule_id=chat, batch_index=0, field=output_digest
```

## 已知边界

- 请求流一旦进入会话，摘要必须保持一致；更换请求流会以 `E_SPEC` 拒绝，避免把旧回执混入新请求集合。
- 未完成的更新目标必须先用同一目标声明恢复，之后才能加载第三代声明。
- 规则允许不同后端维度，但每条请求必须匹配其实际稳定/灰度去向的维度；同一客户端的去向变化可能使原先合法的向量在新声明下触发 `E_SPEC`。
- `--spawn-backends` 默认允许请求携带声明的计算参数；独立脚本不使用 `--allow-dynamic` 时会强制参数与自身声明一致。
