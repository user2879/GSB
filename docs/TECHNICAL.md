# 技术说明：架构、缓存键、失效语义与验证记录

## 1. 总体数据流

```
samples.jsonl + config.yaml (+injections.json)
        │  read_jsonl / load_config / load_injections
        ▼
按序列排序 → 行级注入 → build_windows（等间隔/非有限校验 + 确定性插补 + 窗口内容指纹）
        ▼
Scheduler（线程池，全局预算 + 单序列配额，稳定排序键）
        ▼
process_window：window→train(每模型)→predict(每模型)→metric 分层内容寻址缓存
        ▼
WindowResult（定点预测/指标/拟合参数/残差/插补位置）
        ▼
build_ir → v1 定宽文本 或 v2 JSON（+降级旁路 sidecar） ＋ 诊断报告 + 快照清单
```

## 2. 确定性保证

- 样本生成用 `np.random.SeedSequence([seed, hash(series_id)])` 为每条序列派生独立
  `Generator`，不读取/修改全局随机状态；观测 = 截距 + 线性趋势 + 正弦周期 + 高斯噪声。
- 模型拟合只用确定性运算：naive 取末值、moving_average 取窗口均值、autoregressive 用
  `np.linalg.lstsq` 最小二乘，多步预测递归；同输入多次运行逐元素相等（测试覆盖）。
- 指纹统一走 `json.dumps(sort_keys=True, separators=(",",":"))` 的 SHA-256 短哈希，
  与字典插入顺序、进程、线程调度无关。
- 指标在 Python 任意精度 int 上计算，避免浮点累加顺序差异与整数溢出。

## 3. 定点量化与指标

- 标度 `scale = 2 ** (frac_bits + gear)`，`quantize(x)=round(x*scale)`。切档位 `gear` 或
  位宽 `frac_bits` 都会改变 scale，从而真实改变内部定点整数与指标（有专门测试断言）。
- 集成预测：各模型定点预测按权重加权后 `round` 回定点整数，再与定点真值比较。
- MAE/RMSE 在整数域计算，按窗口序号升序、窗内步序升序累加；MAPE 对真值
  `|actual_q| <= 0`（近零）的点跳过百分比、改由绝对误差判定，并置 `near_zero_used`；
  若全部近零则 MAPE 记哨兵 `-1`。

## 4. 分层缓存键与失效语义

| 阶段 | 键派生输入 |
| --- | --- |
| window | 窗口内容指纹（时间戳/原始值/缺失位置）+ 配置哈希 + 交换版本 |
| train | window 键 + 模型名/阶数/超参指纹 + 交换版本 |
| predict | train 键 + 预测步长 + 交换版本 |
| metric | 全部 predict 键集合 + 量化参数指纹 + 交换版本 |

- 缓存为内容寻址：输入不变则键不变 → 命中直接复用，不触发重算；任一上游变化使下游键自然失配。
- 实测编辑效果（见 `demo_out/demo_report.json`）：
  - 替换单序列：仅该序列窗口及其下游重算（reused=144, recomputed=48）。
  - 新增序列：旧窗口全复用，仅新序列重算（reused=192, recomputed=48）。
  - 删除序列：保留窗口全复用（reused=160, recomputed=0）。
  - 改窗口/模型/量化或切交换版本：相关层键全变，对应全量重算。
- 账本 `Ledger.canonical()` 按稳定排序键去重排序，记录每键最终 `reused/recomputed` 状态与
  指标引用的模型键集合；增量与全量的账本键集合一致（仅状态因命中而不同）。

## 5. 交换版本互操作与旁路

- v1 定宽行：`HD/MW/PR/MT` 四类，字段宽度集中在 `constants.V1_LAYOUT`，解析严格校验总长度，
  不符即 `E_FIELD_LAYOUT`；权重为千分比整数、模型用整数码、缺失策略无法表达。
- v2 JSON：浮点权重 + `precision_label` + 命名模型 + 命名缺失策略 + 配置/版本指纹。
- 降版 v2→v1：`permille=round(weight*1000)`，残差 `weight - permille/1000` 连同精度标签与
  缺失策略写入 sidecar；sidecar 携带 `config_fingerprint` 与 `version_fingerprint`。
- 升版 v1→v2：无旁路时精确换算并打 `permille-exact`；有旁路且双指纹匹配时还原原始浮点权重、
  精度标签与命名缺失策略（测试断言还原 `mean_fill` 与 0.30047，残差 0.00047）。

## 6. 调度、预算、快照与稳定顺序

- 并发：`ThreadPoolExecutor(max_workers=min(cfg.max_workers, 任务数))`，绝不超过配置并发。
- 预算：观测点按窗口训练长度计费；线程安全扣减全局预算与单序列配额，超单序列配额记
  `E_SERIES_BUDGET_EXHAUSTED`（全局耗尽记 `E_GLOBAL_BUDGET_EXHAUSTED`），中止该序列并保留
  待办游标，其它序列继续。
- 顺序：任务带 `series_id|window_index|stage` 稳定排序键，结果与失败清单最终按键串行归并，
  与线程调度顺序/并发度/预算耗尽时机无关（并发 vs 串行测试逐字段相等）。
- 快照：缓存与清单先写 `*.tmp` 再 `os.replace` 整体替换；清单记录全局版本号、配置哈希、
  交换版本指纹与已完成窗口游标；`can_resume` 以配置哈希 + 交换版本判定是否续跑。

## 7. 故障注入与业务码

| 注入类型 | 业务码 | 命中层级 |
| --- | --- | --- |
| irregular_timestamp | E_TIMESTAMP_IRREGULAR | 行/序列 |
| non_finite_value | E_NON_FINITE | 行/序列 |
| window_too_long | E_WINDOW_TOO_LONG | 窗口 |
| quant_gear_out_of_range | E_QUANT_GEAR_INVALID | 窗口 |

注入仅作用于内存副本，不改源样本；注入编号写入失败清单与报告 `injection_hits`，并映射到上表
业务码。解析类还有 `E_FIELD_LAYOUT/E_DUPLICATE_KEY/E_LAG_NEGATIVE/E_MODEL_ENUM/E_MISSING_ENUM`，
预算类 `E_SERIES_BUDGET_EXHAUSTED/E_GLOBAL_BUDGET_EXHAUSTED`，旁路错配 `E_SIDECAR_MISMATCH`。

## 8. 实测验证记录（本机）

- 单元/集成测试：`python -m unittest discover -s tests` → **Ran 14 tests ... OK**。
- 9 步编辑序列增量 vs 空白全量逐字段一致性（全部 `ok=True`，`all_consistent=True`）：

| 步骤 | 版本 | 窗口 | 失败 | 复用 | 重算 |
| --- | --- | --- | --- | --- | --- |
| baseline | v1 | 24 | 0 | 0 | 192 |
| switch-version | v2 | 24 | 0 | 0 | 192 |
| replace-series | v2 | 24 | 0 | 144 | 48 |
| add-series | v2 | 30 | 0 | 192 | 48 |
| window-params | v2 | 25 | 0 | 0 | 200 |
| model-params | v2 | 25 | 0 | 0 | 200 |
| quant-gear | v2 | 25 | 0 | 0 | 200 |
| delete-series | v2 | 20 | 0 | 160 | 0 |
| switch-back | v1 | 20 | 0 | 0 | 160 |

- 并发(4)/串行(1) 一致性：无差异；预算专项：正确产出 `E_SERIES_BUDGET_EXHAUSTED` 且游标正确。
- 跨版本往返：v2→v1 残差记录正确（构造权重 0.30047 → 千分比 300，残差 0.00047），再升版据
  sidecar 无损还原权重与命名缺失策略。
