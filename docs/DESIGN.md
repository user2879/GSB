# 逐层激活定点化诊断管线 — 技术说明

## 1. 目标与问题边界

管线回答：同一批多层激活在定点化后进入 Softmax 与 LayerNorm 两个数值敏感算子时，
误差在哪一层被放大、在哪个环节越界。四个追踪环节固定为：

1. `row_stats`：行内统计量（均值、方差）
2. `softmax_norm`：Softmax 归一化
3. `ln_scale_bias`：LayerNorm 缩放与偏置
4. `cross_layer`：跨层累计

以 float64 无量化路径为真值；真值元素为零时相对误差不可定义，自动切换为
绝对误差判定，并在产物里以 `zero_truth_branch=true` 标注该分支。

## 2. 目录结构

```
fxpdiag/
  errors.py        业务错误码、PipelineError、有序失败清单
  models.py        数据模型、v1 定宽字段规格、方案/位宽/舍入码表
  fixedpoint.py    定点量化器、定点乘除、整数 rsqrt、定点指数近似
  operators.py     定点 Softmax/LayerNorm 与 float64 真值路径
  weights.py       v1 定宽文本行、v2 JSON 的读写与字段校验
  conversion.py    v1<->v2 换算、降级旁路 sidecar、往返损耗
  convert_service.py  双版本互转编排
  validation.py    张量/配置/命令/层间无环校验
  fingerprints.py  内容指纹、参数指纹、阶段缓存键
  ledger.py        行级缓存条目、复用账本与决策
  scheduler.py     元素预算 + 单层配额的稳定调度
  snapshots.py     临时目录写入 + 整体替换 + 续跑判定
  error_track.py   四环节误差、首次越界、翻转归因
  pipeline.py      管线状态、行计算、增量失效、跨层累计、并发执行
  edits.py         编辑命令流
  loader.py        磁盘加载与故障注入
  samplegen.py     固定种子样本生成
scripts/
  generate_samples.py  仅生成样本
  run_diagnostic.py    端到端诊断并落报告
tests/               unittest 测试套件（30 个用例）
```

## 3. 定点数值口径

- 定点数 `q` 表示真值 `q * 2**(-frac_bits)`，`frac_bits` 为定点小数位宽（1..15）。
- 位宽枚举仅允许 `8/16/32`；量化档位 `coarse/standard/fine` 对应 0/1/2 个 guard 位，
  改变饱和上下界，因此切档会真实改变整数中间值，而不仅是显示。
- 每次量化、定点乘除、右移都记录 `RoundingEvent`（`EXACT/UP/DOWN/SAT_UP/SAT_DOWN`、
  量化前浮点、量化后整数、量化步长）。
- Softmax：减行最大值（数值稳定）→ 范围归约 + 四阶泰勒定点指数
  （元数据 `EXP_RANGE_REDUCTION_TAYLOR4`）→ 列序号升序累加（`COL_ASC`）→ 定点除法归一。
- LayerNorm：升序累加求均值/方差 → 整数 rsqrt（定点比例 `r^2*q = 2^(3f)`，
  位长初值 + 固定 32 次 rsqrt 牛顿迭代，中间乘积为任意精度整数避免窄位宽溢出）
  → 归一化 → 缩放 → 可选偏置。
- 零方差退化行：rsqrt 回退为 1.0，标记 `degenerate=true` 与
  `ZERO_VAR_RSQRT_FALLBACK_ONE`，不抛异常。

## 4. 双版本交换格式

- v1：64 列定宽文本行，字段 `layer(24) row(6) scale_permille(12) bias_code(12) bit_width(10)`；
  千分比与偏置码范围、位宽枚举逐行校验，非法行进失败清单且不静默跳过。
- v2：JSON，逐行浮点缩放、命名方案（如 `Q8.8_NORM` / `Q8.8_NORM_NO_BIAS`）、命名舍入模式。
- v1 升 v2：`scale = permille/1000`、偏置码按方案小数位宽精确换算、位宽按码表映射命名方案，
  生成精度标签 `permille-exact-bw*`。
- v2 降 v1：浮点四舍五入还原千分比，截断残差、精度标签、被丢弃的命名舍入模式写入
  `*.sidecar.json`；再次升回优先从旁路还原原始浮点与舍入模式，逐行/逐层记录往返损耗。

## 5. 指纹、缓存与增量

- 阶段缓存键由上游内容指纹、参数指纹、量化档位（`QuantSpec.key()`）与交换版本号派生，
  覆盖 Softmax 输出、LayerNorm 统计量、LayerNorm 输出、版本转换产物四类。
- 每行记录引用的上游行集合；编辑单行激活或单层参数时，仅目标行与其下游传递闭包失效，
  其余行走 `fingerprint-hit` 直接复用，账本给出 REUSE/RECOMPUTE 与依据。
- 预算分批时采用两遍判定：初判 → 重算并传播上游阶段键 → 复判，
  保证同批上游变化也能被下游识别为 `upstream-key-changed`。

## 6. 调度、快照与异常隔离

- 线程池大小取 `max_workers` 且不超过重算行数；已实测并发峰值不超过配置值。
- 调度按拓扑层序（同层名字典序）+ 行序号稳定消耗预算，超预算行保留待办并打
  `E_OP_BUDGET_PENDING`；无进展时强制放行首行，保证有限步内完成。
- 快照先写临时目录再 `os.replace` 整体替换；启动按 `config_version + config_hash`
  判定能否续跑，版本号或配置哈希不匹配则拒绝续跑。
- 故障注入与正常流程隔离：非有限缩放、零方差、位宽越界、层名重复分别落到
  `E_NON_FINITE`、运行时退化标记、`E_BITWIDTH_ENUM`、`E_DUPLICATE_LAYER`，
  注入项编号写入报告。

## 7. 翻转归因

两档位（coarse vs fine）下 Softmax argmax 不一致时，按固定优先级归因：

1. `VARIANCE_DEGENERATE`：方差退化
2. `QUANT_ROUNDING`：量化舍入导致
3. `CROSS_LAYER_ACCUM`：跨层累计导致

## 8. Windows 环境准备（任务指定）

```powershell
# 安装 Python 3.11 官方安装包时勾选 Add python.exe to PATH
python --version
pip install numpy==1.26.4
```

当前验证环境为 Linux + Python 3.11 + numpy 1.26.4，仅使用标准库
（json/hashlib/concurrent.futures/tempfile）与 numpy，路径全部相对工作区。

## 9. 验证方式

```bash
python scripts/generate_samples.py           # 生成 samples/
python -m unittest discover -s tests -v      # 30 个单元测试
python scripts/run_diagnostic.py             # 端到端诊断报告 reports/diagnostic_report.json
```
