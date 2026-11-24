# MM Ledger：多模态批量推理回执归一账本

这是一个可独立运行的 Python 3.11 领域组件与完整示例，用于处理多模态生成服务的本地批量推理结果。它把磁盘上共存的 r1/r2/r3 三种回执归一为同一份确定性账本，并沿“工作进程参数版本 → 批次 → 请求归属”判定失效边界、复用未受影响缓存、执行预算/配额调度，并生成质量与故障诊断报告。

示例中的推理工作进程是真实的 `127.0.0.1` 回环 HTTP 子进程。输入向量由固定种子和请求序号确定性生成，定点路径使用 NumPy `int64`，float64 参考路径使用同一 NumPy 版本；不访问真实模型或线上服务。

## 已交付内容

- `mmledger/`：配置校验、三版本回执解析、双向转换、归属裁决、定点资源账本、预算调度、阶段快照、故障注入、回环工作进程与 CLI。
- `samples/`：确定性生成的配置、请求流、归并规则、r1/r2/r3 回执、跨批覆盖、批内名次冲突、转换旁路和四类故障输入。
- `tests/`：30 个单元/集成测试，覆盖真实回环子进程。
- `snapshots/`：不可变阶段快照和原子 `CURRENT.json` 指针。
- `reports/`：归一账本、诊断报告、故障演练报告、验收证据。
- `scripts/verify.py`：一键复现验收检查并写出 `reports/verification.json` 与 `.md`。
- `scripts/demo.cmd`、`scripts/verify.ps1`：Windows 命令提示符/PowerShell 入口。

## 环境准备（Windows）

```bat
cd /d C:\path\to\workspace
python --version
python -m pip install -r requirements.txt
```

`requirements.txt` 固定为 `numpy==1.26.4`。Mac/Linux/WSL 使用相同 Python 命令即可。

确认回环端口可用并清理旧快照/报告：

```bat
if exist snapshots rmdir /s /q snapshots
if exist reports rmdir /s /q reports
python -m mmledger demo --root . --jobs 4
```

PowerShell 等价命令：

```powershell
Remove-Item -Recurse -Force snapshots,reports -ErrorAction SilentlyContinue
python -m pip install -r requirements.txt
python -m mmledger demo --root . --jobs 4
python scripts/verify.py
```

示例配置默认从 `18900` 起申请两个连续空闲回环端口；生成器若发现占用会自动向后查找。工作进程只绑定 `127.0.0.1`。

## 快速命令

```bat
:: 生成样本，启动 w-alpha/w-beta 子进程并执行一次全量构建
python -m mmledger generate --root . --jobs 4

:: 从空白快照全量运行
python -m mmledger run --root . --jobs 4 --full-rebuild

:: 续跑；相同输入应全部命中缓存，账本逐字段不变
python -m mmledger run --root . --jobs 4

:: 预算与单工作进程配额
python -m mmledger run --root . --full-rebuild --budget 9
python -m mmledger run --budget 4
python -m mmledger run --full-rebuild --quota 5

:: 四类故障注入演练，不提交主快照
python -m mmledger fault-drill --root . --jobs 4

:: 单条回执转换
python -m mmledger convert --root . ^
  --input samples/receipts/r3_batch_003.json ^
  --to r1 ^
  --output samples/conversion/manual_r1.json ^
  --sidecar-out samples/conversion/manual_sidecar.json ^
  --journal samples/conversion/manual_journal.json
```

所有 CLI 输出均为确定性 JSON；异常以业务码和失败清单返回。配置致命错误时退出码为 `2`，故障演练未全部命中时退出码为 `1`。

## 示例数据

`samples/config.json` 定义两个工作进程：

- `w-alpha`：参数版本 `3`，`127.0.0.1:18900`（实际端口由生成器选择）。
- `w-beta`：参数版本 `4`，相邻回环端口。

请求流被拆为 5 个批次。回执覆盖：

- `r1_batch_001.json`：仅批次、请求、输出摘要。
- `r2_batch_002.json`：增加参数版本、整数耗时毫秒、工作进程标识。
- `r3_batch_003.json`：增加批内稳定名次、覆盖引用、参数摘要快照。
- `r3_batch_004_overwrite.json`：`req-009` 的跨批重复投递，通过覆盖引用取代 `batch-003` 中的投递。
- `r3_batch_005_rank_conflict.json`：同一批同一请求的不同工作进程名次冲突，双方均写入裁决依据。

归一后得到 12 条权威请求（13 次投递中消除 1 次重复）。跨批重复请求的权威回执由覆盖引用和整数耗时裁决；批内名次冲突按工作进程标识升序裁决。

## 输出

主要文件：

- `reports/ledger.json`：最终可复核账本，包含归属裁决、资源指标、依赖失效边界、规范化的失效/复用计划、调度轨迹、失败清单、质量摘要。
- `reports/diagnostic_report.json`：详细诊断报告，另含物理执行审计 `actual_invalidation` 和注入命中。
- `reports/diagnostic_report.md`：易读摘要。
- `reports/fault_drill/`：独立故障演练输入得到的账本与报告。
- `reports/verification.json` / `.md`：实际验收结果。
- `snapshots/generations/gXXXXXXXX/`：不可变阶段产物；`snapshots/CURRENT.json` 原子指向当前代。

账本排序键固定：请求归属 `(batch_id, request_id, worker_id, receipt_uid)`，失败 `(source, ordinal, code, item, injection_id)`，资源批次 `(batch_id, worker_id)`。线程池任务键由批次、工作进程和阶段名组成，结果收集后重新排序，因此 `--jobs 1` 与 `--jobs 8` 输出一致。

## 三版本内部表示

每条回执始终暴露同一字段集合：

```text
batch_id, request_id, output_digest,
parameter_version, duration_ms, worker_id,
stable_rank, overwritten_ref, parameter_digest
```

缺失值使用显式 JSON 标记 `{"$missing": true}`，而不是 `0`、空字符串或 `null`。内部对象同时保留 `format_version`、原始 `source_version`、来源文件/序号、缺失字段列表和完整性标签 `complete|partial`。

更完整的字段布局、旁路与错误码见 `docs/SCHEMAS.md`；算法与恢复设计见 `docs/DESIGN.md`。

## 定点资源口径

对模型维度 `D=input_dim`、`H=hidden_dim`、量化位宽 `Q`：

```text
base_parameters       = D*H + 2*H
macs_per_request      = 2*base_parameters
quantized_macs        = ceil(request_count * macs_per_request * Q / 8)
utilization_ppm       = min(1_000_000,
                            request_count*1_000_000 / (batch_size*duration_ms))
```

批次耗时取该批权威回执中已观测整数耗时的最大值；没有任何已观测值时保留缺失标记，利用率也保持缺失。质量摘要分别输出 `image_text` 与 `image_only` 的请求数、已观测/缺失耗时数、总和、均值、最小/最大和固定桶分布。

## 失效与缓存语义

每个批次缓存包含：工作进程参数版本、请求/回执语义指纹、归一结果、归属裁决指纹和资源结果。参数版本变化只失效引用该工作进程的批次；回执替换失效对应批次；模型维度/量化参数/批大小变化失效资源计算；归并规则变化触发全局重裁，但未变化的批内归一结果可继续复用。

`ledger.invalidation_reuse_plan` 是由**最终输入**派生的规范化失效/复用清单，因而全量与增量逐字段一致；`report.actual_invalidation` 是本次运行的真实计算/缓存命中审计。两者分离，既保证结果可比较，也证明未受影响条目未重新计算。

## 快照与故障隔离

流水线按 `parse → ownership → resources → ledger` 写出阶段产物。所有阶段先进入 `snapshots/.staging-*`，写完后整目录改名到 `generations/gXXXXXXXX`，再以临时文件 + `os.replace` 原子替换 `CURRENT.json`。清单记录全局版本、配置哈希、回执 SHA-256 指纹、已完成阶段/工作进程、已消耗预算、待办和实际失效审计。

故障演练从独立配置加载四种输入：

| 注入编号 | 故障 | 业务码 |
| --- | --- | --- |
| `INJ-TRUNCATED` | 截断 JSON | `E_TRUNCATED_RECEIPT` |
| `INJ-UNKNOWN-PARAM` | 未知参数版本 | `E_UNKNOWN_PARAM_VERSION` |
| `INJ-NONFINITE-DURATION` | `NaN` 耗时 | `E_DURATION_NONFINITE` |
| `INJ-DUPLICATE-BATCH` | 重复投递身份 | `E_DUPLICATE_BATCH_ID` |

单条坏回执进入失败清单，其他回执继续归一。

## 实际验证结果

本次在 Python `3.11.0rc1`、NumPy `1.26.4` 下实际执行：

```text
python -m unittest discover -s tests -v
30 tests, OK

python scripts/verify.py
status = passed
ledger_digest = 9574beb364c2d5aa2f4297a7a0b64c552461f0c15814bfc82c96f0d01b64da4f
```

已通过的验收项包括：串行/并发账本一致、增量/全量账本一致、三版本归一、转换往返完整还原、四类注入全部命中、参数版本仅失效 `w-alpha` 的 `batch-001/003/005` 而复用 `w-beta` 的 `batch-002/004`、预算待办与下一轮优先、工作进程配额中止、真实回环子进程、快照清单完整、相同输入续跑零归一/零资源重算。

完整证据见 `reports/verification.json`、`reports/verification.md` 和 `reports/ledger.json`。

## 测试

```bat
python -m unittest discover -s tests -v
python scripts/verify.py
```

测试只使用临时工作区和本地回环端口，不访问外部目标，不依赖 Docker。
