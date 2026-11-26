# GSB 列存跳读诊断管线 — 技术说明

本交付在本地 WSL/Linux 上运行，不使用 Docker、外部账号、网络站点或外部数据集。回环通信固定为 `127.0.0.1`，端口由操作系统动态分配。

## 目录

- `gsb/spec.py`：JSON 声明加载、别名归一、强校验与规格摘要。
- `gsb/codec.py`：固定种子行值生成、`raw/dict/rle` 编码、`none/zlib-1..9` 压缩、GSB1 块文件读写。
- `gsb/planner.py`：SQL 三值谓词求值、索引跳读规则、稳定谓词顺序与计划摘要。
- `gsb/engine.py`：真实行匹配、漏行判定、误保留判定、投影与聚合下推。
- `gsb/attribution.py`：索引失真、压缩/编码不符、谓词与空值顺序三源归因。
- `gsb/scheduler.py` / `gsb/worker_server.py`：全局块组预算、单列租约、子进程 TCP 执行器和真实套接字超时。
- `gsb/invalidation.py`：块组/列/索引/谓词/查询/结果引用图与失效传播。
- `gsb/ledger.py` / `gsb/snapshot.py`：SQLite 权威账本、事务摘要、崩溃恢复、快照版本识别和整体替换。
- `gsb/pipeline.py` / `gsb/cli.py`：端到端编排与命令行。
- `gsb/ui.py`：可选 Tkinter 报告查看器。
- `samples/diagnostic_spec.json`：确定性四组样本；`samples/diagnostic_spec_v2.json`：BG-A02 种子与压缩档位变更。
- `tests/test_spec_codec_planner.py`、`tests/test_pipeline_runtime.py`：与验收项对应的 20 个 unittest 用例。
- `scripts/run_acceptance.py`：一键本地验收并写出证据。

根目录的 `spec_codec_planner.py` 与 `pipeline_runtime.py` 是兼容门面，分别转发到 `gsb` 包内对应模块。

## 确定性规则

- 规格、计划、归因、缓存和账本摘要均使用 `sort_keys=True`、紧凑分隔符的 UTF-8 canonical JSON，再计算 SHA-256。
- 块组按 ID 排序，列按列名排序，查询按 ID 排序；谓词按 `(column, op, canonical(value))` 排序。
- 行生成使用稳定的 counter-hash：
  `sha256("gsb-generator/v1|row|seed=<seed>|row=<i>")`，对 `modulus` 取模并加 `offset`。
- `null_every > 0` 时，`(row_index + 1) % null_every == 0` 的行固定为 `NULL`。
- 字典编码仅包含实际出现的非空整数，并按数值升序排列；`0xffffffff` 表示 NULL 索引。
- RLE 按完整值与 NULL 标志划分最大连续段。
- GSB1 块摘要是整个 `.gsb` 字节文件的 SHA-256；逻辑摘要只绑定块组 ID、行数和解码列，因此压缩档位变化改变字节摘要但不改变逻辑摘要。

## 常用命令

```bash
python3 --version
python3 -m pip install -r requirements.txt
python3 scripts/generate_sample.py

python3 -m unittest tests.test_spec_codec_planner tests.test_pipeline_runtime -v
python3 scripts/run_acceptance.py

python3 -m gsb.cli doctor
python3 -m gsb.cli run \
  --spec samples/diagnostic_spec.json \
  --output-dir artifacts/local-demo \
  --session-id local-demo
python3 -m gsb.cli recover \
  --spec samples/diagnostic_spec.json \
  --output-dir artifacts/local-demo \
  --session-id local-demo
python3 -m gsb.cli ui --report artifacts/local-demo/report.json
```

## 崩溃恢复模型

1. 会话开始时在 `batch_intents` 中为全部计划块组写入 `running` 意图。
2. 调度器在租约内通过独立子进程的唯一 `127.0.0.1` 端口解码块组。
3. 每个块组批次在同一个 SQLite 事务中写入：
   - `batches` 结局、内容/逻辑摘要、解码行数；
   - `payload_json`、`occupancy_json`、`lease_json`、`cursor_json`；
   - 整数 `row_version` 与绑定整行的 `tx_digest`；
   - 同时把对应意图更新为 `completed`。
4. 强杀后，已提交批次可重放；仍为 `running` 的意图在恢复时标记为 `unknown` 并要求重跑。
5. 续跑只解码未知/失效块组；查询结果在全部批次齐备后写入 `query_results`。
6. 重放会校验事务摘要、结果摘要，并用当前规格确定性重建块字节，定位首个不一致的块组与字段。

## 实际验证结果

最近一次一键验收记录在 `artifacts/verification/acceptance.json` 与 `artifacts/acceptance-local.json`；完整 unittest 输出在 `artifacts/verification/test-results.txt`，环境体检在 `artifacts/verification/doctor.json`。

已观察到：

- `Ran 20 tests ... OK`。
- 确定性样本共 4 个块组、44 行；全量核实解码 4/4 个块组。
- 注入的 BG-A02 索引区间失真产生 1 个 `E_FALSE_SKIP`，漏行索引为 `[1, 4, 5, 9]`、漏行数 4，主导归因为 `index_distortion`。
- 调度峰值为全局在途 3、单列在途 2；4 个回执来自 4 个不同回环端口。
- 强杀退出码为 `-9`；恢复得到 `unknown_batches=[BG-A02,BG-A03,BG-A04]`、已提交 BG-A01；续跑仅解码这 3 组，最终 4 个批次和 4 个查询全部重放一致，且查询摘要与基线逐项相同。
- BG-A02 的种子和压缩档位变更后，仅重解 BG-A02；失效查询为 `q_amount_high`、`q_null_amount`、`q_region_3`，`q_id_values` 的 canonical payload 字节保持不变。
- 快照版本/规格摘要/内容摘要不符时整体替换，不做字段合并。

`artifacts/demo*` 是工作区中保留的既有参考产物；本交付新生成的稳定演示位于 `artifacts/local-demo/`，一键验收的独立运行目录记录在 `artifacts/verification/acceptance.json` 的 `run_dir`。
