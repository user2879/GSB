# 多波段辐射定标与跨传感器回归对齐组件 — 技术说明

## 1. 目标与范围

面向遥感影像预处理的确定性组件，解决两个核心业务问题：

- **增量重标定**：传感器定标系数（增益、偏移、量化位宽、饱和上限）与跨传感器回归模型随批次反复调整，每批瓦片需重新标定为物理量纲并回归到统一参考传感器尺度。
- **精确依赖失效**：某传感器系数变更后，仅其定标瓦片、引用该传感器的回归模型及其下游对齐结果失效；未失效产物按内容指纹直接复用，绝不重算。

仅依赖 Python 3.11 标准库与 NumPy，无网络、无外部账号、无容器。YAML 使用内置的受控子集解析器 `rs_calib/yaml_lite.py`。

## 2. 目录结构

```
rs_calib/
  errors.py        业务码与受控阶段异常
  fingerprint.py   SHA-256 内容指纹与缓存键派生
  fixedpoint.py    定点量化（bankers rounding）与位宽越界检测
  yaml_lite.py     确定性 YAML 子集解析
  models.py        Tile / CalibRecord / RegressionModel / AlignmentRecord
  loader.py        瓦片包/定标表/回归配置加载与逐项校验
  calibration.py   逐像素辐射定标
  regression.py    确定性岭回归求解与逐瓦片对齐
  executor.py      线程池执行器（稳定排序键回收）
  edits.py         增量编辑引擎（追加/回填/系数或配置变更）
  injection.py     异常注入与业务码命中映射
  store.py         阶段化快照、原子整体替换、stale 判定
  pipeline.py      预算调度、指纹失效/复用、回归编排、失败剪枝
  runner.py        增量/全量端到端运行与诊断报告
  samplegen.py     固定种子样本生成器
  cli.py           命令行入口 generate/run/compare
tests/test_pipeline.py  11 项单元/端到端测试
scripts/run_demo.sh     一键演示
sample_data/            固定种子生成的本地样本
```

## 3. 处理阶段与确定性保证

### 3.1 加载校验（load）
逐行解析 JSONL，所有失败以业务码进入失败清单且不中断整体加载，校验项包括：

| 校验 | 业务码 |
| --- | --- |
| 瓦片 ID 重复 | `E_TILE_DUPLICATE` |
| 传感器不在定标表 | `E_TILE_SENSOR_UNKNOWN` |
| 波段未声明 | `E_TILE_BAND_UNDECLARED` |
| 像素为负 / 非整数 / 超位宽 | `E_PIXEL_NEGATIVE` / `E_PIXEL_NOT_INTEGER` / `E_PIXEL_OVERFLOW` |
| 时间戳未严格递增 | `E_TIMESTAMP_NOT_ASCENDING` |
| 增益非正或非有限 | `E_GAIN_INVALID` |
| 偏移非有限 | `E_OFFSET_INVALID` |
| 回归白名单为空 / 引用不存在 | `E_REG_SOURCE_EMPTY` / `E_REG_SOURCE_UNKNOWN` |
| 回填时间戳早于瓦片游标 | `E_BACKFILL_TIMESTAMP_ORDER` |

### 3.2 逐像素定标（calib）
固定累加顺序：**瓦片 ID 升序 → 波段号升序 → 像素行优先**。

1. 饱和像素按 `saturation_limit` 计数处理并统计饱和像素数；
2. 物理量 `physical = count * gain + offset`；
3. 按 `2**frac_bits` 量化档位做四舍六入五成双，输出有符号定点整数；
4. 越界抛 `E_FIXEDPOINT_OVERFLOW`；预算不足抛 `E_CALIB_BUDGET_PENDING` 并入待办。

### 3.3 跨传感器回归（regress）
- 按 `scene_id + band` 配对源传感器与参考传感器同名场景像素；
- 特征矩阵由配置阶数构造（样本为 `[1, x, x²]`）；
- 正规方程 `(XᵀX + λP)w = Xᵀy + λPw₀`，常数阶不惩罚；`np.linalg.cond` 超 `near_singular_rcond` 时标记 `near_singular=true` 并以提升后的正则兜底；
- 对齐结果回定点，并对参考瓦片输出残差均值、最大绝对残差与 RMSE。

## 4. 内容指纹与精确失效

三级缓存键均由上游内容指纹经 SHA-256 派生（`fingerprint.py`）：

- 定标键 = 指纹(瓦片内容 + 传感器系数指纹 + 定点参数)；
- 模型键 = 指纹(源/参考传感器系数指纹 + 回归参数指纹 + 样本场景集合)；
- 对齐键 = 指纹(定标键 + 模型键 + 定点参数)。

模型记录 `referenced_sensors`，对齐记录引用 `model_id/model_key`。因此：

- 传感器 `S2` 系数变更 → 仅 S2 瓦片定标键变化；引用 S2 的模型键变化并 `RECOMPUTED`；其下游对齐键变化；
- S3 与参考 S1 的未失效产物键保持不变，直接 `REUSED`，测试 `test_07_precise_invalidation` 显式断言 S3 模型零重算。

失败清单采用内容寻址去重与阶段末剪枝：待办解除、重算成功或瓦片消失后，对应旧失败自动移除，保证增量与全量失败清单逐字段一致。

## 5. 预算调度与待办

- 每批授予 `pixel_budget` 像素预算，调度器按稳定顺序消耗；
- 单瓦片像素超过剩余预算时整体入待办，记 `E_CALIB_BUDGET_PENDING`，下一批优先消费待办；
- 永久失败（如位宽越界）不回待办，避免无进展死循环；
- 账本字段含 `pixels / cumulative_pixels / budget_grant_remaining / status / sort_key`，报告内 `reconciled = 总处理像素 ≤ 总授予预算`。

## 6. 增量 vs 全量等价性

- **增量**：依次应用编辑计划，每批结束运行一次，期间按缓存键复用；
- **全量**：从空白状态先施加全部编辑，再按最终瓦片集合与最终资源一次性运行（预算轮次仅控制切分，不影响结果）。

两者的定点物理量、对齐值、残差摘要、指纹集合、失效复用清单与失败清单顺序通过 `cli compare` 逐字段比对。由于全部求解为确定性矩阵运算、顺序由排序键固定、缓存为内容寻址，批量切分不改变最终内容。

## 7. 快照与并发一致性

- 加载/定标/回归三阶段产物先写入临时目录，再 `os.replace` 整体替换；
- `manifest.json` 记录快照版本、全局版本、配置哈希、定标表哈希、已完成瓦片游标与各产物版本；
- 启动时 `--resume` 校验版本号与配置哈希，不一致输出 `E_SNAPSHOT_STALE` 并自动重建；非 resume 启动默认从空白重建，避免账本重复累加；
- 线程池任务携带 `stable_key(tile_id, band)`，回收时按键排序，最终顺序与单线程严格一致（`test_08_concurrency_determinism`）。

## 8. 异常注入

`injection_config.json` 五项注入，全部在诊断报告中回写注入编号、期望业务码与命中明细：

| 注入 | 业务码 |
| --- | --- |
| 计数超量化位宽 | `E_PIXEL_OVERFLOW` |
| 未知波段号 | `E_TILE_BAND_UNDECLARED` |
| 回填时间戳早于游标 | `E_BACKFILL_TIMESTAMP_ORDER` |
| 回归源传感器不存在 | `E_REG_SOURCE_UNKNOWN` |
| 定点位宽越界（仅目标瓦片窄位宽） | `E_FIXEDPOINT_OVERFLOW` |

单瓦片异常记录瓦片 ID、波段、阶段名与业务码后继续其余瓦片。

## 9. 实际验证结果

```
python3 -m unittest tests.test_pipeline
Ran 11 tests ... OK
```

覆盖：加载校验、定点值手工复算、回归对齐与残差、增量/全量一致（含注入）、预算待办与账本核对、精确失效、并发确定性、快照 stale 重建、注入诊断、YAML 解析。

一键演示 `bash scripts/run_demo.sh` 输出 `output/demo/*/diagnostic_report_*.json`，对比结果 `equal: true`。

## 10. 命令行用法

```bash
python3 -m rs_calib.cli generate --out sample_data --seed 20261122
python3 -m rs_calib.cli run --sample-dir sample_data --output-dir output/inc --mode incremental
python3 -m rs_calib.cli run --sample-dir sample_data --output-dir output/full --mode full
python3 -m rs_calib.cli compare output/inc/diagnostic_report_incremental.json \
                                output/full/diagnostic_report_full.json
python3 -m rs_calib.cli run --sample-dir sample_data --output-dir output/inj \
    --mode incremental --injection sample_data/injection_config.json
```
