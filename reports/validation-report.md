# 验收验证报告

## 环境与复现

- 环境：Linux 工作区，Python `3.11.0rc1`
- NumPy：`1.26.4`
- pytest：`8.2.2`
- 固定样本种子：`20261124`
- 样本：`samples/weights.jsonl`、`samples/calibration.jsonl`、`samples/calibration_alt.jsonl`、`samples/config.yaml`、`samples/commands.jsonl`
- 完整机器可读报告：`reports/diagnostic-report.json`
- 可视化看板：`reports/dashboard.html`

复现命令：

```bash
python -m pip install numpy==1.26.4 pytest==8.2.2
python -m quant_workbench generate-samples --out samples
python -m quant_workbench run \
  --config samples/config.yaml \
  --weights samples/weights.jsonl \
  --calibration samples/calibration.jsonl \
  --commands samples/commands.jsonl \
  --snapshot .workbench_snapshot \
  --report reports/diagnostic-report.json \
  --html reports/dashboard.html
pytest -q
```

## 自动化测试

执行结果：`28 passed in 2.28s`（最终报告生成时以实际本地耗时为准）。

覆盖内容：

- JSONL/YAML 加载、重复层、版本单调、正形状、非有限权重、非法配置。
- 未知层、替换形状不符、替换非有限权重的命令序号和行号诊断。
- 对称/非对称逐组缩放与零点、有效位数裁剪、饱和方向、截断前整数。
- 目标位宽、方案、小数位宽对内部整数、固定点整数和反量化值的真实影响。
- 六类命令后的增量稳定结果与清空缓存全量重建逐字段一致。
- 位宽/分组/权重变化仅失效目标层及下游；温度/校准集变化保留重量化缓存。
- 1 线程与 3 线程的逐层产物、部署清单、预算扣减计划和失败顺序一致。
- 延迟预算内的陈旧摘要、结果代次、进度、`E_RESULT_STALE` 和 `E_POINT_BUDGET_PENDING` 语义。
- 快照整体替换、恢复续跑、版本/配置哈希冲突以及冲突后快照文件不变。
- CLI 端到端生成 JSON 报告和 HTML 看板。

## 样本运行结果

最终状态：5/5 层完成，部署包 5 个，权重点总数 158，校准样本 9 个。稳定结果与清空全部阶段缓存后的全量重建逐字段一致。

| 层 | 最大绝对误差 | MSE | 校正后最大绝对误差 | 校正后 MSE | 样本数 |
|---|---:|---:|---:|---:|---:|
| layer.0 | 0.0398827424 | 0.0002422213 | 0.0384379388 | 0.0001824159 | 9 |
| layer.1 | 0.2430346528 | 0.0048601690 | 0.2428233740 | 0.0040063413 | 9 |
| layer.2 | 0.7056512559 | 0.0614240620 | 0.6988698404 | 0.0508014933 | 9 |
| layer.3 | 0.0673769820 | 0.0004481311 | 0.0550169751 | 0.0002816498 | 9 |
| layer.4 | 0.1127403817 | 0.0027671417 | 0.0964591412 | 0.0017148642 | 9 |

聚合指标：最大绝对误差 `0.7056512558781853`，MSE `0.014558216344056431`。

## 交互延迟与陈旧结果

配置：交互预算 `20 ms`，线程池 `3`，线程上限 `8`，单批点数预算 `64`，快照间隔 `2` 条命令。

| 命令 | 类型 | 状态 | 代码 | 指标代次 | 上次完成层 | 当前完成层 | 实际耗时 ms | 预算内 |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | set_bits | stale | E_RESULT_STALE, E_POINT_BUDGET_PENDING | 0 | 0 | 1 | 5.995 | 是 |
| 2 | set_group_size | stale | E_RESULT_STALE, E_POINT_BUDGET_PENDING | 0 | 0 | 2 | 4.934 | 是 |
| 3 | set_scheme | complete | — | 3 | 5 | 5 | 12.343 | 是 |
| 4 | replace_weight | complete | — | 4 | 5 | 5 | 6.990 | 是 |
| 5 | set_temperature | stale | E_RESULT_STALE, E_POINT_BUDGET_PENDING | 4 | 5 | 1 | 4.079 | 是 |
| 6 | replace_calibration | stale | E_RESULT_STALE, E_POINT_BUDGET_PENDING | 4 | 5 | 1 | 5.343 | 是 |

4 条陈旧响应均未标记 `is_latest=true`，且 `result_generation` 明确指向旧代次。最终命令流排空后，摘要为最新完整代次。墙钟值仅代表本机本次运行，不作为跨平台绝对性能承诺。

## 调度与缓存

固定调度计划：

- 批次 0：`layer.0`，48 点，余量 16。
- 批次 1：`layer.1`，48 点，余量 16。
- 批次 2：`layer.2`（30 点）、`layer.3`（20 点）、`layer.4`（12 点），末余量 2。
- 批次内任务按稳定层名键排序；并发只改变提交/完成时序，不改变最终顺序或扣减计划。

本次运行缓存统计：实际重量化计算 170 点，重量化复用 206 点；失效/复用账本 30 条。六条命令执行完成后，增量结果与全量重建一致。

## 快照与版本判定

快照包含 `ingest`、`calibration`、`quantization`、`evaluation`、`deployment` 五个阶段。提交在临时目录完成后整体替换；运行态同时保存有效逐层配置、替换权重和替换校准集。

本次快照清单：

- 全局版本：1，修订号：4，代次：6，已执行命令：6。
- 校准集哈希：`ab15a8bb37e85e8f52b21c93e190cbee10bce43db6bf3dea3a7f4ce91e2d1ac9`。
- 完成层：`layer.0` 至 `layer.4`。
- 五阶段产物版本均已写入 `manifest.json`。

验证结果：

- 从快照重启后的稳定结果与全量重建逐字段一致，恢复过程未重算任何层。
- 将快照版本改为不匹配值后，启动返回 `E_SNAPSHOT_VERSION_MISMATCH`。
- 冲突前后整个快照目录摘要不变，未发生逐文件覆盖。
- 配置哈希不一致同样触发版本冲突分支。

## 结论

本地样本满足逐层压缩结果可解释、预算内摘要不谎报最新代次、任意命令序列增量结果与全量重建逐字段一致、并发调度与串行稳定顺序一致，以及快照可恢复且版本冲突不破坏旧快照的验收目标。
