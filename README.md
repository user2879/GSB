# 本地推理回执账本

这是一个不依赖 Docker、外部服务或网络数据的本地推理回执系统。它将工作进程回传的 r1/r2/r3 三种历史格式归一为同一份确定性账本，沿「工作进程参数版本 → 批次 → 请求归属」传播失效，并提供快速/权威双视图、阶段化快照和独立故障注入。

## 快速开始（Windows 10/11）

需要 Python 3.11+ 与 NumPy 2.x。仅需这一个第三方运行时依赖。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import numpy as np; assert np.__version__.startswith('2.'); print(np.__version__)"

python gsb.py validate --project sample
python gsb.py ledger --project sample --state-dir artifacts/runtime/state --output artifacts/ledger.json
python gsb.py compare --project sample --state-dir artifacts/runtime/view-state --budget-ms 25
```

工作区自带 `gsb.py` 启动器，因此无需先执行 `pip install -e .`。如果已有可导入的 NumPy 2.x，只需让 `src` 可导入；启动器会自动加入 `src`，并在存在本工作区隔离目录 `.deps` 时优先使用它。也可安装为控制台命令：

```powershell
python -m pip install -e .
gsb-ledger validate --project sample
```

## 工作进程

工作进程是独立子进程，只监听配置中允许范围内的 `127.0.0.1` 端口。输入由固定整数种子、参数版本和请求标识通过 SHA-256 派生，前向使用 NumPy `int64` 矩阵乘加，输出摘要和参数摘要均为 SHA-256。

```powershell
python scripts/start_workers.py sample
python scripts/smoke_worker.py sample w-alpha b-smoke-001
python scripts/stop_workers.py
```

启动器将 PID、端口和日志位置写入 `artifacts/runtime/workers.json`，日志位于 `artifacts/worker-logs/`。烟雾客户端发送一行 JSON 请求并接收 r3 回执行。协议示例：

```json
{"batch_id":"b-smoke-001","request_ids":["r-001","r-004"],"receipt_version":"r3","duration_ms":6,"covered_refs":{}}
```

## 常用命令

```powershell
# 配置、请求流与合并规则校验
python gsb.py validate --project sample

# 构建权威账本；--render-version r1 会先降级、记录旁路、再恢复后入账
python gsb.py ledger --project sample --state-dir artifacts/runtime/state --render-version r1

# 快速视图；0 ms 预算会返回 E_RESPONSE_BUDGET_EXCEEDED 和已完成前缀
python gsb.py query --project sample --state-dir artifacts/runtime/state --mode fast --budget-ms 25

# 快速与权威共同完成范围逐字段比较
python gsb.py compare --project sample --state-dir artifacts/runtime/view-state --budget-ms 25

# 独立注入配置；四类故障及其编号、业务码写入诊断报告
python gsb.py diagnose --project sample --fault-config faults/isolated_faults.json --output artifacts/fault-report.json

# 七步命令序列；最后一个命令后与空白全量重建逐字段比较
python gsb.py scenario --source-project sample --work-dir artifacts/runtime/scenario-work --commands sample/commands/incremental_sequence.json --out-dir artifacts/scenario

# 回执版本转换与旁路文件
python gsb.py convert --input sample/receipts/w-alpha.r3.ndjson --source-version r3 --target-version r1 --sidecar artifacts/runtime/sidecar.json --output artifacts/runtime/down.r1.ndjson
python gsb.py convert --input artifacts/runtime/down.r1.ndjson --source-version r1 --target-version r3 --sidecar artifacts/runtime/sidecar.json --output artifacts/runtime/up.r3.ndjson --worker-id w-alpha

# 离线确定性回执生成
python gsb.py generate --project sample --worker-id w-alpha --version r3 --request-id r-001 --request-id r-004 --output receipts-new.ndjson

# 快照续跑/配置变化识别
python gsb.py snapshot-status --project sample --state-dir .state
```

## 数据契约

`sample/config.json` 的定义要点：

- `workers[]`：`worker_id` 唯一；`param_version` 为当前活动版本；`known_param_versions` 保存允许出现在历史回执中的版本；`port` 必须位于 `port_range`；`model_dim`、`hidden_dim`、`quant_scale` 为正整数。
- `batch_size`：正整数且不超过请求数。
- `merge_rule.entries[]`：引用的工作进程和参数版本必须在配置内已知；可同时选择同一工作进程的当前和历史版本，以便失效批次进入账本。
- `receipt_sources[]`：NDJSON 文件、格式 `r1/r2/r3`，以及在字段缺失时用来解析归属的 `default_worker_id` 和 `default_param_version`。文件内字段始终保留显式缺失标记，来源默认值不伪造字段。
- `fixed_point_bits`、`throughput_ops_per_ms`：整数定点资源策略。
- `fault_config`：可为空；诊断命令也可用 `--fault-config` 覆盖。

请求流每行一个对象，必须包含 `seq`、`request_id`、`arrival_ms`；`seq` 严格递增，`arrival_ms` 非负，`request_id` 唯一。

回执行：

- r1：`batch_id`、`request_id`、`output_digest`。
- r2：在 r1 上增加 `param_version`、`duration_ms`、`worker_id`。
- r3：在 r2 上增加 `rank`、`covered_refs`、`param_digest`。

摘要宽度固定为 `sha256:` 加 64 位小写十六进制；`batch_id` 最宽 40，`request_id` 最宽 64，`worker_id` 最宽 32。格式错误、宽度不符、缺失必需字段或非有限数值均记录 `E_RECEIPT_MALFORMED`，坏行不会阻断其他回执。

## 输出与错误码

权威账本按 `request_id` 字典序排列，每个请求只有一条权威条目，同时保留完整投递链、覆盖边、裁决依据、批次资源、失效/复用边界和失败清单。

主要业务码：

| 业务码 | 含义 |
| --- | --- |
| `E_CONFIG_INVALID` | 配置或批大小非法 |
| `E_REQUEST_INVALID` | 请求序号、到达时间或标识非法 |
| `E_RECEIPT_MALFORMED` | 回执行格式、宽度、缺失字段或非有限值非法 |
| `E_UNKNOWN_PARAM_VERSION` | 回执引用完全未知的参数版本 |
| `E_DUPLICATE_RANK` | 同一批次/工作进程内 r3 名次重复 |
| `E_RECEIPT_MISSING` | 请求没有可被归并规则选中的权威回执 |
| `E_COVERAGE_REF_INVALID` / `E_COVERAGE_CYCLE` | 覆盖引用无效或成环 |
| `E_LEDGER_DIVERGED` | 快速与权威共同范围存在字段差异，附字段路径 |
| `E_RESPONSE_BUDGET_EXCEEDED` | 交互预算耗尽，返回最接近的已完成前缀 |
| `E_SNAPSHOT_CORRUPT` | `CURRENT` 或引用快照损坏，后续可冷重建 |
| `E_CONVERSION_LOSS` | 转换无法恢复的字段未被旁路保留 |

## 目录

```text
src/gsb_ledger/        配置、解析、转换、归属、资源、账本、快照、视图、工作进程
sample/                手工配置、请求流、r1/r2/r3 回执、故障配置、命令序列
scripts/               子进程启停、烟雾调用、一键验收
tests/                 22 项自动化测试
docs/DESIGN.md         算法与一致性设计
docs/VALIDATION.md     实际验证环境、命令和结果
artifacts/verification/一键验收生成的账本、故障诊断和汇总
```

## 验证

```powershell
python -m pytest -q
python scripts/run_acceptance.py
```

一键验收会覆盖三版本归一、覆盖裁决、参数版本失效边界、七步增量/全量逐字段一致、四类故障命中、预算耗尽、转换旁路、定点位宽变化和真实回环子进程调用。当前环境实测结果见 `docs/VALIDATION.md` 与 `artifacts/verification/summary.json`。

系统不使用 Docker、容器或网络服务，所有样本和结果均由本地确定性计算产生。
