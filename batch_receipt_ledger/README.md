# Batch Receipt Ledger

离线、确定性的本地推理批归属回执账本。项目不依赖 GPU、CUDA、外部模型或网络服务；计算仅使用 CPU、Python 整数定点和 NumPy。

## 快速运行

```bash
cd batch_receipt_ledger
python3 -m pip install -r requirements.txt

# 16 项 unittest/pytest 验证（包含真实 127.0.0.1 Worker 子进程与 Tk 桩接线）
python3 -m pytest tests -q

# 生成样本、执行命令流、Worker 回环调用、视图比对和快照提交
PYTHONPATH=. python3 -m batch_ledger.cli demo \
  --sample-dir samples/acceptance \
  --out artifacts/acceptance

# 打开 Tkinter 界面（需要本机 Tcl/Tk 运行库；Windows 官方 Python 已包含）
PYTHONPATH=. python3 -m batch_ledger.cli ui \
  --report artifacts/acceptance/final_report.json

# 根据最终输入指纹判断快照 resume/rebuild
PYTHONPATH=. python3 -m batch_ledger.cli recover \
  --snapshot-dir artifacts/acceptance/snapshot \
  --input-dir artifacts/acceptance/final_inputs
```

也可以直接执行 `./run_acceptance.sh`。

## 目录与职责

- `batch_ledger/model.py`：统一回执模型、`$missing` 显式缺失标记、失败结构。
- `batch_ledger/receipts.py`：r1/r2/r3 解析器、通用入口和业务版本校验。
- `batch_ledger/conversion.py`：三版本转换、降级旁路、恢复和往返审计。
- `batch_ledger/inputs.py`：配置、Worker、请求到达序号、批切分和失败排序。
- `batch_ledger/adjudication.py`：批次/Worker 聚合、重复投递裁决、落选引用和比率。
- `batch_ledger/resources.py`：整数定点 MAC、参数量化、利用率和失效传播。
- `batch_ledger/engine.py`：确定性账本引擎、稳定排序、增量缓存和全量报告。
- `batch_ledger/concurrency.py`：线程池批次任务与 `stage + batch_id` 稳定排序键。
- `batch_ledger/snapshot.py`：阶段产物临时目录、原子清单、指纹和恢复判定。
- `batch_ledger/lifecycle.py`：SIGINT/SIGTERM 批边界优雅停止。
- `batch_ledger/view.py`：快速视图、权威视图、共同范围字段比对和 `E_VIEW_DIVERGENCE`。
- `batch_ledger/worker.py`：`127.0.0.1` 回环 Worker 子进程、空闲端口选择和 JSON 行协议。
- `batch_ledger/samples.py`：固定种子 `20261125` 的样本、三版本回执和命令流。
- `batch_ledger/ui.py`：Tkinter 快速结论/权威重算界面。
- `tests/test_ledger.py`：验收测试。
- `artifacts/acceptance/run_report.json`：最近一次实际运行报告。
- `docs/RUN_REPORT.md`：人工可读验证记录。

## 样本账本

固定种子生成 32 条正常到达记录和 1 条到达毫秒倒退记录；批大小为 4 时形成 `B-0001` 到 `B-0008`，按 `W1/W2/W3` 循环分配。`B-0008` 包含 4 个跨批重复投递。另有：

- `receipts/r1.jsonl`：`B-0001`、`B-0002`，共 8 条。
- `receipts/r2.jsonl`：`B-0003` 到 `B-0005`，共 12 条。
- `receipts/r3.jsonl`：`B-0006` 到 `B-0008`，共 12 条。
- `receipts/invalid_worker.jsonl`：未知 `W999`，输出 `E_WORKER_INVALID`。
- `receipts/bad_version.jsonl`：版本码 `r9`，输出 `E_RECEIPT_VERSION`。
- `requests_invalid.jsonl`：到达毫秒倒退，输出 `E_ARRIVAL_ORDER`。
- `workers.json` 中 `W-BAD-PORT` 端口越界，输出 `E_WORKER_INVALID`。

命令流依次执行回执替换、批大小 4→3、`W2` 参数版本 21→22、新增 `B-9001`、归并规则切换为 `version_then_batch_v1`。

## 裁决规则

默认规则 `coverage_then_version_v1`：

1. 若候选 r3 回执的 `supersedes_ref` 能匹配另一候选的稳定 `receipt_ref`，该回执成为覆盖方。
2. 多个覆盖方之间按 `elapsed_ms` 升序、参数版本降序、批次标识字典序和稳定引用排序。
3. 没有可解析覆盖引用时，按 `param_version` 降序、`batch_id` 字典序升序，最后以稳定引用兜底。
4. 每个请求输出 `basis`、完整 `ranked_refs` 和 `rejected_refs`。

归并规则可通过命令切换；所有顺序均在内存中排序，不依赖线程调度或文件枚举顺序。

## 版本转换与旁路

r1 只有核心字段；升级到 r2/r3 时，缺失字段写为内部 `$missing`，并生成 `missing_fields` 与 `completeness_label`。r3→r1 时，六个扩展字段全部写入 `SidecarStore`，其中题目点名的 `rank_in_batch`、`supersedes_ref`、`param_digest` 会记录在降级条目的 `degraded_r3_fields`。恢复时从旁路还原，并与直接载入的 r3 条目逐字段比较。

最终验收报告中的 `conversion_audit.all_fields_match=true`，36 条回执全部一一对应。

## 失效边界

依赖链固定为：

```text
worker_param_version -> batch/resource -> decision/request attribution
```

`W2` 从 21 切到 22 后，只有回执显式引用 `W2@21` 的 `B-0005`、`B-0008` 被标记失效；权威归属落在这两个批次上的 5 个请求进入失效清单。`W1/W3` 的 5 个批次资源键以及无版本字段的 `B-0002` 命中缓存，共 6 个缓存命中；加入 `W2@22` 新批次后，可复用条目为 7。

## 一致性与并发

`LedgerEngine` 保存内容相关的资源缓存。每条命令后都用相同最终状态从空白构建报告并比较完整字典；测试还使用固定随机种子生成 6 组、每组 10 条混合命令。`build_report(parallel=True)` 与 `parallel=False)` 逐字段相等，因为每个线程任务都带 `stage=resource|batch=...` 稳定键，合并前按键排序。

报告包含 `report_hash`，但一致性比较同时比较除哈希外的完整报告和哈希值。

## 快照与退出

`SnapshotStore.commit` 将 `parse.json`、`attribution.json`、`resources.json`、`ledger.json` 先写入同目录 `.staging-*` 临时目录，校验 fsync 后原子改名为正式快照，再原子替换 `CURRENT` 指针。清单记录：

- `global_version`
- `config_hash`
- `receipt_fingerprints`
- 每个阶段产品的 `version`、文件和 SHA-256

启动恢复时，全局版本、配置哈希或回执指纹不一致即判定 rebuild；全部匹配且阶段哈希有效才 resume。`.staging-*` 残留不会被当前指针引用，模拟强制终止后仍可从最后一次整体替换的快照恢复。

`GracefulStop` 在批次边界检查 SIGINT/SIGTERM；收到信号后不再开始新批次，但会完成当前批次并执行提交回调。命令行可用 `recover` 按配置哈希和回执文件指纹输出恢复判定。

## 交互视图

`view.interactive_view` 提供：

- 已完成请求的归属摘要；
- `completed/total` 进度；
- 当前范围内失效条目数；
- 权威固定请求顺序；
- 共同范围内逐字段比较；
- 注入差异时返回 `E_VIEW_DIVERGENCE`；
- 超预算或发现差异时优先返回上一次已完成摘要。

Tkinter 按钮分别触发快速路径和权威重算。当前无图形沙箱缺少 `libtk8.6.so`，因此只完成编译与无界面逻辑验证；Windows 官方 Python 安装 Tcl/Tk 后可直接运行 UI。
