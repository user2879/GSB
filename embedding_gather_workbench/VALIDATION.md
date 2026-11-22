# 验证记录（本地确定性样本）

环境：Python 3.11.0rc1、NumPy 1.26.4、PyYAML 6.0.2；单线程与线程池（并发=3）两种模式。
样本种子固定 `20261122`，无外部网络、无容器。

## 复现命令

```bash
pip install -r requirements.txt
python -m egw generate --out-dir samples
python -m unittest discover -s tests -v
python -m egw verify --config samples/config.yaml --rows samples/rows.jsonl \
  --requests samples/requests.jsonl --edits samples/edits.jsonl --work-dir out
```

## 实际结果

### 1. 加载校验（要求 1）
- 注入样本全部命中预期错误码：
  `E_ROW_INVALID=6`（JSON 截断、重复词 ID、槽位冲突、负坐标、空序列、重复请求）、
  `E_DIM_MISMATCH=1`、`E_NONFINITE_VECTOR=1`、`E_AGG_MODE_UNKNOWN=1`。
- 零预算配置 `config_zero_budget.yaml` 被拒绝，返回 `E_CONFIG_INVALID`，进程退出码 1。
- 合法行 32 条、合法请求 8 条正常加载，失败清单按稳定键排序。

### 2. 行量化（要求 2）
- 步长 `2^-frac_bits`；相同档位/位宽重复运行定点整数与残差逐元素一致（无随机源）。
- 档位 16/8 → 12/4、位宽 8 → 6 均真实改变整数表示；大值在不同档位触发不同截断。

### 3. Gather 与聚合（要求 3）
- SUM/MEAN 共用同一 `gathered_int`；MEAN 定点域除法记录余数，满足
  `total = quotient*count + remainder`，四种舍入规则均有单元测试。
- 缺失 ID 返回 `E_ID_MISSING` 并记录请求与位置后继续（样本 `rq_missing_01`）。

### 4. 指纹与失效（要求 4）
- 行/块/布局三级指纹随档位切换而变化；单行替换只失效引用该 ID 的请求，
  块替换/追加按块失效，档位切换全量失效，聚合切换只失效单请求。
- 增量账本显示未失效请求进入 `reused` 且不重算，清单按请求 ID 排序。

### 5. 五环节误差与归因（要求 5）
- 报告逐请求输出 `quantize/gather/accumulate/divide/round` 的最大绝对/相对误差；
  容差收紧到 1e-3 时所有请求首次超差环节正确标注为 `quantize`。
- 双档位（16/8 vs 12/4）对比样本 `rq_cross_01` 最大差异 0.1133、判定翻转，
  归因均为 `quantization_truncation`；MEAN 请求可观测 `accumulate_rounding`。
- 高贡献位置按量化残差 L1 降序、同值按请求 ID/位置升序决断。

### 6. 块布局顺序可观测性（要求 6）
- 探针构造 8 行（4 正 4 负）同一 ID 序列两种布局：
  - 全局顺序累加两布局完全一致：`fixedpoint_max_abs_diff = 0`
  - 分块部分和归并差异：`block_merge_max_abs_diff = 52425`（饱和非结合性）
  - float64 参考路径两布局完全一致：`float64_reference_max_abs_diff = 0.0`
- 报告含每请求块遍历顺序、每块部分和及与最终和差值，按贡献绝对值降序。

### 7. 增量与全量等价（要求 7）
- 编辑序列含单行替换、整块替换、追加新块、档位切换、位宽调整、聚合切换共 9 步。
- 单线程与并发两种模式下，增量运行与空白全量重建的
  `results / failures / reuse_ledger / stage_versions` 逐字段相等
  （`incremental_full_equivalent = true`，版本序列 1..10 一致）。

### 8. 预算调度、快照与报告（要求 8）
- 预算 12 点按请求 ID 稳定顺序消耗，超预算请求标记 `E_POINT_BUDGET_PENDING=5` 并保留待办。
- 并发=3 不超过配置；单线程与并发最终请求/行/失败顺序完全一致
  （`single_vs_concurrent_equal = true`）。
- 快照经临时目录 `os.replace` 整体提交，含 `manifest.json`、`quant_cache.json`、
  `request_cache.json`；配置哈希变化时拒绝续跑，哈希一致时命中复用。
