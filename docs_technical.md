# 定点卷积与饱和截断算子组件 — 技术说明

## 1. 目标与范围

面向部署前的整数激活/整数权重，提供可集成的纯 CPU、无外部账号/网络/GPU 依赖的
定点卷积算子组件，覆盖：定点乘加、累加缓冲溢出处置、饱和截断、多模式反量化归约、
float64 参考路径逐级误差归属、乘加点预算调度、内容指纹位置级缓存失效、
批次快照整体替换/恢复、异常注入隔离与 JSON 诊断报告。

运行环境：Python 3.10+（实测 3.11），NumPy 仅用于固定种子样本生成，核心算子零 NumPy 依赖。

## 2. 目录结构

```
fxpconv/
  errors.py        业务错误码与稳定有序失败清单
  fingerprint.py   SHA-256 确定性内容指纹（不受 PYTHONHASHSEED 影响）
  models.py        算子描述数据模型与加载校验（枚举/维度/预算/位宽组合）
  tensorpack.py    整数张量包 JSONL 模型与形状/符号/数据区间校验
  calibration.py   校准样本集加载校验
  kernel.py        整数定点卷积内核、溢出首处记录、窗口指纹、受影响位置几何判定
  quantize.py      量化位宽饱和截断 + 三种舍入的反量化归约
  reference.py     float64 无截断参考路径
  attribution.py   三环节误差比对与首次超差唯一归属
  cache.py         三阶段内容指纹缓存键与位置级精确失效
  scheduler.py     单批 MAC 预算切分、稳定排序键、串行/线程池同序执行
  snapshot.py      临时目录暂存 + 整体替换提交 + 陈旧快照判定
  injections.py    四类异常注入加载、隔离引擎与业务码映射
  pipeline.py      主编排
  report.py        JSON 诊断报告
samples/           固定种子(20261121)生成的确定性样本
scripts/           样本生成、端到端演示、负样本校验
tests/             21 个 unittest 用例
snapshots/ reports/ 运行产物（进程可读写）
```

## 3. 关键设计

### 3.1 固定累加顺序与确定性
累加顺序固定为 `输出通道 → 输入通道 → 核行 → 核列` 嵌套升序；使用 Python 大整数
先精确累加，再对照描述声明的缓冲位宽逐坐标检查区间，因此可记录溢出发生的第一处
累加坐标 `(ic, kh, kw)`、方向与溢出前部分和。填充区按清零规则贡献 0，缓冲起始显式清零。

### 3.2 饱和策略真实改值
- `saturate`：越界裁剪到缓冲/量化区间边界；
- `wrap`：二补数回绕（仅允许 `acc_bits>=16 且 qbits>=16`，非法组合在加载期拒绝）。
恰等于边界的值归入区间并标注 `boundary_sample`；饱和后的边界值同样标注。

### 3.3 反量化与三种舍入
`value = trunc_int * act_scale * weight_scale[oc]`，量化到 `frac_bits` 定点网格，
单位阶跃 `step=2^-frac_bits`：
- `trunc_zero` 向零舍入；`round_nearest_even` 就近取偶；`floor` 向负无穷。
最后裁剪到输出区间；元数据记录舍入模式、小数位宽、阶跃与每组缩放因子来源。

### 3.4 位置级精确失效
每个位置三阶段（accum/trunc/deq）分别派生缓存键，键内只纳入真正影响取值的策略指纹：
未溢出位置不带饱和策略标签、舍入结果稳定的位置不带舍入模式标签。替换单个激活单元时，
先用窗口指纹实测受影响位置，并与几何判定 `positions_referencing_activation_cell`
对账；未失效位置直接复用、零重算。

### 3.5 逐级误差归属（同一位置唯一环节）
三环节：`accum`（缓冲值 vs 精确整数和）、`trunc`（截断整数 vs 精确整数和）、
`deq`（反量化值 vs float64 真值）。按描述容差判定超差，取第一个超差环节作为
`first_fail_stage`，根因映射为 `accum_overflow_saturation` / `boundary_truncation` /
`dequant_rounding`，同一位置不会同时归属两个环节。

### 3.6 预算调度与并发一致
调度先独立完成预算切分（与执行模式无关），任务稳定排序键为
`(批次, 位置, 阶段序号)`。单位置成本超过当前批次剩余预算记
`E_BATCH_BUDGET_EXHAUSTED` 并保留待算；超过整批预算直接记录并跳过避免无限重排。
线程池最大并发严格取配置，结果按稳定键重排，因此串行/并发的整数结果、截断结果、
反量化值、误差归属、预算扣减序列、失败清单顺序完全一致。

### 3.7 快照整体替换与续跑
每批先写独立临时目录，再原子改名发布不可变版本目录，并用 `os.replace` 换入
`manifest.json`（版本号、配置哈希、张量包指纹、已完成批次游标、已消耗预算）。
启动时配置哈希或张量指纹不一致即 `E_SNAPSHOT_STALE` 并从头重算。

### 3.8 注入隔离
| 注入类型 | 业务码 | 命中阶段 |
| --- | --- | --- |
| 非有限缩放因子 | E_INJECT_NONFINITE_SCALE | dequant |
| 量化位宽越界 | E_INJECT_QBITS_OUT_OF_RANGE | truncate |
| 整数数据超出形状声明 | E_INJECT_DATA_OUT_OF_SHAPE | load（全局审计） |
| 饱和策略与位宽组合非法 | E_INJECT_ILLEGAL_STRATEGY | validate（全局审计） |

单位置异常只隔离该位置，其余位置继续计算；注入编号、业务码与消息写入报告。

## 4. 运行方式

```bash
python3 -m scripts.generate_samples        # 固定种子生成样本
python3 -m scripts.validate_negative_cases # 加载层负样本业务码
python3 -m scripts.run_demo                # 端到端演示并落报告
python3 -m unittest tests.test_fxpconv -v  # 21 个单测
```

## 5. 实测验证结果（本地 CPU，种子 20261121）

- 输出布局：200 个输出位置，单位置 18 个乘加点，单批预算 60 → 67 个批次。
- 16 位累加缓冲真实溢出 23 个位置，首处溢出坐标/方向/溢出前部分和均有记录；
  量化边界样本 200 个；saturate 与 wrap 截断整数取值真实不同。
- 三种舍入模式在 106 个位置产生不同反量化值；frac_bits 4/6/8 切换在 50 个位置
  产生不同取值，单位阶跃分别为 1/16、1/64、1/256。
- 缓存：无变化第二轮 200 个位置全部复用、0 重算；替换单个激活单元仅 36 个
  几何/指纹一致的受影响位置重算，其余 164 个复用；切换舍入模式部分位置失效。
- 串行与线程池（配置并发 4、实测峰值 3）：196 个正常位置产物、67 条预算扣减序列、
  72 条失败清单顺序逐元素一致。
- 注入：6 个注入项全部命中并映射 4 类业务码；4 个单位置故障被隔离，196 个位置照常完成。
- 快照：相同输入续跑 `resumed=True`；配置哈希变化命中 `E_SNAPSHOT_STALE` 并重算。
- 误差归属：accum 超差 23（与缓冲溢出数一致）、trunc/deq 超差 196，报告含三环节
- 直方图、最大差值位置与环节、预算消耗、快照版本、失败码统计。

报告路径：`reports/diagnostic_report.json`。

## 6. 业务码清单

`E_DESC_INVALID`、`E_SHAPE_MISMATCH`、`E_BUDGET_INVALID`、
`E_BATCH_BUDGET_EXHAUSTED`、`E_SNAPSHOT_STALE`、
`E_INJECT_NONFINITE_SCALE`、`E_INJECT_QBITS_OUT_OF_RANGE`、
`E_INJECT_DATA_OUT_OF_SHAPE`、`E_INJECT_ILLEGAL_STRATEGY`。
