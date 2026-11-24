# 回执归并与资源诊断应用 — 技术说明

## 1. 模块划分

| 模块 | 职责 |
| --- | --- |
| `receipt_ledger/loader.py` | 工作器、请求流、归并规则、命令流加载与校验 |
| `receipt_ledger/receipts.py` | r1/r2/r3 解析、显式缺失标记、行号/版本畸形识别、文件指纹 |
| `receipt_ledger/convert.py` | 三版本双向转换、降级旁路元数据、完整性标签、稳定回执引用 |
| `receipt_ledger/attribution.py` | 批次/工作器聚合、覆盖引用消解、固定顺序归属裁决 |
| `receipt_ledger/resources.py` | 整数定点乘加量/利用率、参数版本失效链传播（numpy 整数向量化） |
| `receipt_ledger/snapshot.py` | 四阶段临时目录写入、`os.replace` 整体替换、清单恢复判定 |
| `receipt_ledger/commands.py` | 五类编辑命令 + 转换命令的确定性纯函数应用 |
| `receipt_ledger/injection.py` | 独立注入配置、命中编号贯穿失败清单与报告 |
| `receipt_ledger/engine.py` | 阶段编排、确定性线程池、快速/权威双路径、报告汇总 |
| `receipt_ledger/__main__.py` | `python -m receipt_ledger` 命令行入口 |

## 2. 三版本统一表示

统一内部模型为 `Receipt`（`receipt_ledger/models.py`），始终保留
`source_version` 与 `missing` 元组：

- r1 天然缺失的六个字段全部登记在 `missing`；
- 显式占位 `__MISSING__` 与天然缺失等价登记，与“缺失位非空”畸形互斥；
- 多余键、版本字段与文件名不符（字段错位）、类型错误产出
  `E_RECEIPT_MALFORMED`，记录物理行号与版本；
- 截断/无法解析的 JSON 行产出 `E_RECEIPT_TRUNCATED`；
- 非有限耗时产出 `E_NON_FINITE_DURATION`。

## 3. 降级旁路与往返一致性

- r1 → r3：缺失字段以 `__MISSING__` 显式补齐，完整性标签为
  `complete` 或 `incomplete:<字段列表>`。
- r3 → r1：被剥离字段写入 `sidecar`（含原始版本、原始引用、原始完整性
  标签与全部字段值），账本 `conversion_log` 记录降级条目。
- 再升级时由旁路逐字段还原，`receipt_ref` 与完整性标签保持不变，
  因此参与归属裁决的身份与原回执完全一致（见
  `tests/test_receipts_convert.py`）。

## 4. 归属裁决（固定顺序）

1. 按 `(batch_id, worker_id)` 聚合；
2. r3 `supersedes` 覆盖引用先消解，悬空/跨请求引用进入失败清单；
3. 存活候选排序键：参数版本降序 → 整数耗时升序 → 批次字典序升序 →
   批内名次升序 → 回执引用升序；
4. 裁决依据（覆盖链 + 排序规则 + 胜出引用 + 候选数）写入每条账本记录；
5. 某请求候选全部不可用时输出 `E_ATTRIBUTION_UNRESOLVED`，其余请求保留。

## 5. 定点资源指标与失效链

定点比例 `FIXED_SCALE = 1_000_000`，全程仅整数运算：

- 乘加量 `MACs = Σtoken × model_dim × hidden_dim × 2 × 量化系数`；
- 利用率 = 实际令牌乘加量 / 批次满载容量（定点除法，向零截断）；
- 吞吐 = MACs / 批内总耗时毫秒。

失效沿固定链传播并记录完整路径：

```
worker:<id> → param_version:<变化> → batch:<id> → request:<id> → attribution
```

未知参数版本与版本过期分别标注；未受影响条目在资源阶段直接复用，
复用/失效决策写入 `reuse_log`。

## 6. 快照与恢复

四阶段 `parse / attribution / resources / summary` 各自：

1. 计算规范化输入哈希；
2. 哈希命中清单则直接复用持久化产物，不重算；
3. 未命中则写入 `tmp/` 临时文件并 `os.replace` 整体替换；
4. 清单记录全局版本号、配置哈希、回执文件指纹（size + sha256）、
   命令游标与各阶段 `input_hash/payload_hash/version/reused`。

启动恢复只比对配置哈希与文件指纹，不做逐文件覆盖。

## 7. 双路径一致性

- 权威路径按固定阶段顺序跑完全部条目；
- 快速路径在交互预算内返回已完成阶段的归属摘要、失效条目数与进度；
- 超预算或结论差异时回退最近一次已完成结论并标注 `E_VIEW_STALE`；
- 共同完成范围内，快速路径与权威路径摘要逐字段一致（测试使用假时钟
  确定性验证，真实运行使用 `time.monotonic`，墙钟不进入持久化产物）。

## 8. 并发与确定性

- 所有并发只通过 `map_reduce_ordered`：线程池执行、按输入顺序回填；
- 失败清单按 `(source, line, code, injection_id, message)` 规范化排序；
- 报告使用规范化 JSON（键序固定、分隔符固定），因此并发度 1 与 8 的
  `report.json` 与 `snapshot_manifest.json` 可逐字节一致；
- 逐步应用命令流的最终账本与相同最终输入一次性全量运行逐字段一致。

## 9. 异常注入隔离

注入配置仅通过 `--inject` 显式启用，样本目录与故障目录物理分离：

| 编号 | 类型 | 期望业务码/效果 |
| --- | --- | --- |
| INJ-001 | 非有限耗时 | `E_NON_FINITE_DURATION` |
| INJ-002 | 未知参数版本 | 资源失效链标注未知版本 |
| INJ-003 | 截断回执行 | `E_RECEIPT_TRUNCATED` |
| INJ-004 | 重复请求标识 | `E_DUPLICATE_REQUEST_ID` |

命中编号写入失败清单 `injection_id` 与报告 `injection.hit_ids`。

## 10. 运行方式

```bash
python -m pip install -r requirements.txt
python tools/generate_samples.py
python -m receipt_ledger \
  --config samples/workers.json \
  --samples samples \
  --commands samples/commands.json \
  --output output/normal
python -m receipt_ledger \
  --config samples_faulty/workers.json \
  --samples samples_faulty \
  --commands samples_faulty/commands.json \
  --inject samples_faulty/injection.json \
  --output output/faulty
python -m unittest discover -s tests -v
```
