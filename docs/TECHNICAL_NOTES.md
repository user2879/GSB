# 缓存命中诊断技术说明

## 1. 范围与确定性边界

`fcdiag` 只使用 Python 标准库完成运行时计算；样本生成器使用独立 LCG 整数随机数，不依赖全局 NumPy RNG。样本、流水线、配置、三版回执、命令流和注入清单均由固定种子生成，可通过 `python scripts/generate_fixtures.py` 重建。

流水线所有数值先转换为 `10 ** frac_bits` 定点整数，统计累加严格按样本 ID 顺序执行，除法采用最近整数舍入、平方根除使用整数算法。量化档位 `gear` 对中间整数做最近倍数吸附，并记录 `step / before / after / residual`。

## 2. 三版回执归一

内部模型 `ReceiptRecord` 同时保留：

- `source_version`：业务来源版本，转换后仍可追溯；
- `wire_version`：当前文本线格式版本，决定定宽字段与 `present` 标记；
- `present`：字段是否存在；缺失字段保持 `None`，不编造默认值；
- `digest`、`duplicate_refs` 等仅在来源支持时存在。

v1 升 v2 时保留千分比整数 `hit_permille`，旁路写入精确浮点 `hit_permille / 1000` 与精度标签；v2 降 v1 时旁路保存截断残差、精度标签、参数版本和耗时。v3 降 v2 时摘要与重复计算引用进入 sidecar；无 sidecar 再升 v3 时写入 `IRRECOVERABLE###`，明确标记不可还原。

## 3. 内容摘要、缓存键与失效边界

每个制品的摘要由四类信息派生：阶段名、作用域、定点 payload、上游摘要集合、参数指纹。参数指纹只纳入当前阶段相关参数与定点配置，避免不相关参数编辑污染摘要。缓存键为 `{artifact_id, digest}` 的稳定 SHA-256 短哈希。

判定顺序为：

1. 无回执：重算；
2. 上游已重算：当前制品及其下游重算；
3. v3：摘要与参数版本同时匹配才复用；
4. v1/v2：摘要缺失，按配置策略处理：
   - `strict`：不得假定可复用；
   - `param_version`：仅在样本指纹未变化、参数版本一致且上游可复用时保守复用，并记录退化依据。

## 4. 预算与并发确定性

任务排序键由阶段位次、全局/样本作用域、样本 ID 稳定位次组成：`stage_index:scope_kind:sample_rank:stage:scope`。规划器单线程决定接受、待办或阶段中止，线程池只执行已接受任务，因此并发度不改变制品顺序、账本顺序或预算扣减序列。

- 批次点数不足：任务进入待办队列并记 `E_ARTIFACT_BUDGET_PENDING`；
- 阶段配额超出：中止该阶段并记 `E_STAGE_BUDGET_EXCEEDED`；
- 下一批先处理上一批待办，再处理新任务；
- 账本只记录最终接受的 `REUSE/RECOMPUTE`，预算事件进入失败/事件清单，避免重复记账。

## 5. 快照与恢复

每个阶段先写 `snapshots/tmp/...`，再通过 `os.replace` 整体替换到 `snapshots/v######/stages/<stage>`。根清单记录全局版本、配置哈希、样本指纹、回执版本指纹、已完成阶段和预算消耗。恢复时先比较版本号与配置哈希，再比较样本/回执指纹；版本变化不会逐文件覆盖，而是进入新的版本目录。

## 6. 验证命令

```bash
python scripts/generate_fixtures.py
python -m pytest -q
python -m fcdiag.cli run --output output/run-report.json
python -m fcdiag.cli incremental-check --output output/incremental-report.json
python -m fcdiag.cli injections --output output/injection-report.json
```

测试使用 pytest 内置 `tmp_path`，不依赖历史快照目录；验收结果以本地命令输出为准。
