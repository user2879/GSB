# 实际验证报告

验证时间：2026-11-25（Asia/Shanghai）。环境信息见 `docs/environment.txt`。

## 环境

- Python：3.11.0rc1
- NumPy：1.26.4
- 平台：当前 Linux 沙箱；全部网络调用仅访问 `127.0.0.1`
- GPU/CUDA/外部模型：不需要，未使用
- Tkinter：当前无图形基础环境报 `ImportError: libtk8.6.so`；`batch_ledger/ui.py` 已通过编译检查，视图逻辑由测试覆盖。Windows 安装官方 Python 时勾选 Tcl/Tk 后可启动界面。

## 执行命令

```bash
cd batch_receipt_ledger
python3 -m pytest tests -v
PYTHONPATH=. python3 -m batch_ledger.cli demo \
  --sample-dir samples/acceptance \
  --out artifacts/acceptance
```

完整测试输出保存在 `docs/pytest-output.txt`，CLI 输出保存在 `docs/demo-output.txt`。

## 测试结果

```text
16 passed
```

覆盖内容：

- 四类业务码与加载不中断；
- 工作区 JSON/JSONL 文件生成和加载；
- r1/r2/r3 通用入口与版本入口一致；
- 旁路降级、恢复和逐字段往返；
- 覆盖引用、耗时排序、无覆盖时版本/批次排序；
- 每条命令后的增量报告等于空白全量报告；
- 6 组固定随机命令序列；
- 并发与串行报告逐字段一致；
- 参数版本失效边界和缓存命中；
- 快照提交、恢复判定和残留临时目录隔离；
- 快速/权威视图一致、`E_VIEW_DIVERGENCE` 和预算摘要；
- 优雅停止批边界语义；
- 两个真实 `127.0.0.1` Worker 子进程和确定性摘要。
- 使用 Tk 桩模块验证 `ui.py` 能构造窗口、渲染权威行并绑定主循环。

## 验收样本结果

`artifacts/acceptance/final_report.json` 的报告哈希：

```text
sha256:6f290014bf382cb6c2c3f6354189bb81336bc72157a0fa4a56689d23721b19b0
```

最终摘要：

- 有效 Worker：3
- 有效请求：36
- 计划批次：12（命令将批大小改为 3 后重切）
- 有效回执：36
- 归属决定：32
- 重复请求：4
- 批次统计：9
- 资源条目：9
- 失败：4（`E_ARRIVAL_ORDER=1`、`E_RECEIPT_VERSION=1`、`E_WORKER_INVALID=2`）

单独载入批大小零配置时，失败清单额外包含：

```text
E_BATCH_SIZE_INVALID: batch_size must be an integer greater than zero
```

## 版本转换审计

最终报告的 `conversion_audit`：

- `entry_count=36`
- `all_fields_match=true`
- `mismatch_count=0`
- `completeness_matches=true`
- `r3_to_r1_downgrade_count=16`
- `r2_to_r1_downgrade_count=12`

这表示所有 r2/r3 条目降级到 r1、写入旁路并恢复后，均与直接载入对应版本逐字段一致。

## 裁决与批次比率

4 个重复请求均完成权威裁决。例如：

- `B-0002`：4 个请求中 1 个跨批重复，重复率 250‰，消解率 1000‰。
- `B-0003`：Q0009 被跨批引用；按最终 `version_then_batch_v1` 规则，`W3@31` 胜过 `W2@21`。
- `B-0008`：4 个请求全部是跨批重复，重复率 1000‰，消解率 1000‰；其中 Q0005 默认覆盖规则下由显式 `supersedes_ref` 决定。

每条决定均包含 `basis`、`ranked_refs` 和 `rejected_refs`。

## 失效边界

`W2 21 -> 22` 后：

- 缓存命中：6；未命中：2。
- 未命中/失效版本组：`B-0005:W2`、`B-0008:W2`。
- 命中包括 `W1` 的 `B-0001/B-0004/B-0007`、`W3` 的 `B-0003/B-0006`，以及无参数版本字段的 r1 批次 `B-0002:W2`。
- 权威归属失效请求：`Q0005`、`Q0017`、`Q0018`、`Q0019`、`Q0020`。
- 新增 `B-9001@W2:22` 后，总复用条目为 7，新批次未被误判为旧版本失效。

## 一致性

`artifacts/acceptance/run_report.json` 中：

```text
incremental_vs_blank_full_equal = true
parallel_vs_serial_equal = true
```

随机命令序列测试还覆盖回执替换、新增批次、参数版本切换、批大小调整、归并规则修改和无效批大小命令。

## Worker 回环验证

演示启动两个子进程：

- `W1-demo` -> `127.0.0.1:29181`
- `W2-demo` -> `127.0.0.1:29182`

两端口均在声明范围 `20000-39999`，JSON 行协议返回 `ok=true`。Q0001/Q0005 的摘要与进程内 `deterministic_forward(seed=20261125)` 完全一致。详见 `artifacts/acceptance/worker_demo.json`。

## 视图与快照

- 部分快速视图与权威视图在共同的 16 个请求范围内一致。
- 注入错误依据时返回 `E_VIEW_DIVERGENCE`，并保留上一次完成摘要。
- 快照恢复判定为 `resume / manifest_and_fingerprints_match`。
- 命令行复核：`batch-ledger recover --snapshot-dir artifacts/acceptance/snapshot --input-dir artifacts/acceptance/final_inputs`。
- 修改配置哈希、全局版本或回执指纹会判定 rebuild。
- 快照清单见 `artifacts/acceptance/snapshot_manifest.json`，当前指针和阶段产物位于 `artifacts/acceptance/snapshot/`。

## 复现说明

若需要清理旧产物，可使用文件管理器或 `gio trash`，避免不可恢复的 `rm -rf`。随后重新执行上述命令即可生成完全相同的账本字段；实际耗时、端口和报告生成时间不作为账本字段。
