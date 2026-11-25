# fixedseason

面向季节性销量序列的确定性定点最小二乘拟合组件与可复现实验台。项目只依赖
Python 3.11 与 NumPy；样本、配置与命令流均由固定整数种子在本地生成。

## 快速开始

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src          # Windows PowerShell: $env:PYTHONPATH="src"
python -m fixedseason generate --directory examples --force
python -m fixedseason run \
  --series examples/series.jsonl \
  --config examples/config.yaml \
  --workspace artifacts/demo \
  --fresh
```

增量命令链并自动与最终状态的全量重建比较：

```bash
python -m fixedseason run \
  --series examples/series.jsonl \
  --config examples/config.yaml \
  --commands examples/commands.jsonl \
  --workspace artifacts/incremental \
  --fresh \
  --check-equivalence
```

中断/恢复：

```bash
# 第一次在首个并发批次检查点后停下
python -m fixedseason run --series examples/series.jsonl --config examples/config.yaml \
  --workspace artifacts/resume --fresh --stop-after 2
# 同版本、同配置、同数据再次启动会续跑
python -m fixedseason run --series examples/series.jsonl --config examples/config.yaml \
  --workspace artifacts/resume
```

位宽翻号诊断：

```bash
python -m fixedseason width-check \
  --series examples/series.jsonl \
  --config examples/config.yaml \
  --workspace artifacts/widths \
  --bits 32 48
```

完整验收：

```bash
python -m fixedseason verify --workspace artifacts/verification
python -m unittest discover -s tests -t . -v
```

## 模型与定点语义

每个序列在零基时间序号 `t = 0..n-1` 上拟合：

```text
y(t) = c_intercept
     + c_slope * ((t - (n-1)/2) / period)
     + sum_k [ c_k_cos * cos(2*pi*k*t/period)
             + c_k_sin * sin(2*pi*k*t/period) ]
```

- 整数基矩阵为 `Q * B`，其中 `Q = 2^quantization_shift`；目标值量化为
  `round(y*Q)`。正规方程约为 `Q^2 B^T B c = Q^2 B^T y`，因此输出的原始系数
  `c` 与 float64 参考系数处于同一物理单位，但全部为 Python 整数。
- 三角函数使用高精度 `Decimal` 级数和计算，不依赖平台 `libm`；量化与插补统一
  使用“最近值、中点远离零”的规则。
- 正规方程按 `时间序号升序 -> 列序升序 -> 内层列序升序` 累加。若原始累加器
  超过配置位宽，所有矩阵元素和右端项使用同一个二进制右移缩放，并记录
  `common_downshift`。
- 整数消元按“绝对值最大主元、并列取最小行号”选主元，逐行记录乘子、中间余量
  和舍入方向。主元低于阈值时固定从最高谐波阶开始降阶并重解；若纯趋势仍失败，
  再退回仅截距，并记录完整失败轨迹和降阶依据。
- `harmonic_order: 0` 会显式标记 `configured_harmonic_zero_pure_trend`，基向量只含
  截距与趋势列。

## 输入格式

`series.jsonl` 每行一个序列。时间戳必须严格递增且等间隔；观测值必须为有限数，
或缺失标记 `null`、`"NA"`、`"NaN"`、`"missing"`、`""`：

```json
{"series_id":"sku-alpha","timestamps":[1000,1001,1002],"values":[180.2,null,184.5]}
```

`config.yaml` 使用内置最小 YAML 子集解析（映射、列表、标量、注释、内联列表）。
核心字段：

```yaml
version: 1
model:
  period: 12
  harmonic_order: 3
  regularization_strength: 0.05
fixed_point:
  bits: 48                    # 16/24/32/40/48/56/62
  quantization_shift: 16      # 6/8/10/12/14/16/18/20
  pivot_threshold: 1000
tolerance:
  relative: 0.0001
  absolute: 0.0001
budget:
  scale_profile: standard
  global_fit_points: 4096
  per_series_quota: 768
  batch_observation_points: 1024
runtime:
  thread_pool_workers: 2      # 必须 <= os.cpu_count()
diagnostics:
  comparison_bit_widths: [32, 48]
```

`commands.jsonl` 每行一个命令，支持：

- `set_harmonic_order`
- `set_regularization_strength`
- `set_fixed_point`
- `set_scale_profile`
- `set_budget`
- `set_thread_pool_workers`
- `replace_observation`
- `append_series`
- `delete_series`
- `run`

非法命令只产生 `E_COMMAND_INVALID`，后续命令继续执行。加载阶段的
`E_SERIES_INVALID`、`E_CONFIG_INVALID`、`E_COMMAND_INVALID` 都会进入报告失败清单。

## 插补、参考路径与误差归因

- 缺失值只使用该序列整个窗口内已经出现的观测。内部空洞使用相邻已观测点线性
  插补；首尾空洞使用最近边缘观测。最近距离并列时固定取较小索引（左侧）。
  规则、来源索引、小数位数和中点舍入策略写入插补产物元数据。
- 真值路径为 **float64、仅使用未缺失观测、不插补** 的最小二乘解。
- 逐步误差按四个阶段输出逐系数 `stage_delta`、相对真值的累计误差、绝对误差、
  判定类型和是否超限：
  1. `interpolation`：插补后 float64 解相对无插补真值；
  2. `normal_equation_accumulation`：定点基/目标量化及正规方程相对插补路径；
  3. `integer_elimination`：整数消元、主元选择和降阶相对定点正规方程；
  4. `coefficient_back_substitution`：整数回代舍入相对整数消元浮点回代。
- 真值系数严格等于零时使用绝对误差和 `absolute_zero_truth` 分支，并列出相关
  系数；非零系数使用相对误差 `relative_nonzero_truth`。
- 报告记录 `first_exceeded_stage` 与首次超限的系数名。

## 缓存、预算和调度

四个持久化内容寻址阶段为 `interpolation`、`basis`、`normal_equations`、
`coefficients`。每个阶段的键由序列内容指纹、计算相关配置指纹和位宽档位派生。
预算、线程数、报告版本不会改变计算键；因此只修改预算不会驱逐系数产物。
每份产物带内容校验哈希，损坏缓存按 miss 处理。

每个序列记录引用的已观测/缺失索引集合。替换单序列只令该序列失效；谐波阶数、
正则强度、位宽或量化档位属于计算配置变化，会令全部序列的计算键变化。

调度器在启动任何线程前，严格按 `series_id` 排序完成预算预分配：

- `fit_points = 序列观测槽位数`；
- 单序列需求超过 `per_series_quota`：`E_FIT_BUDGET_EXCEEDED`，不扣全局预算；
- 超过单批上限或全局剩余不足：进入稳定的待办队列，标记
  `E_POINT_BUDGET_PENDING`；
- 可执行序列按批打包，批内观测点总和不超过 `batch_observation_points`；
- 预算预留和批次划分不受线程完成时序影响。最终报告始终按序列标识排序。

## 快照与恢复

`loading`、`interpolation`、`basis`、`normal_equations`、`elimination`、
`back_substitution` 六个阶段及其待办队列写入临时目录，然后用整目录替换
`snapshot/current`。清单包含：

- 全局版本号；
- 完整配置哈希与计算配置哈希；
- 逐序列数据指纹和库指纹；
- 已完成序列游标/清单；
- 位宽档位、量化档位、预算计划、待办队列、执行失败。

启动时若版本、配置哈希、数据指纹和位宽档位都一致，则从未完成批次续跑；否则
当前快照移入 `snapshot/archive/` 作废。内容缓存独立于运行快照，仍可安全复用。

## 报告与等价口径

`run` 输出 `final_report.json`，主要字段：

- `series[].coefficients`、`trend_slope`、基列阶数/相位；
- `series[].error_attribution`；
- `series[].elimination`（主元、余量、舍入方向、降阶轨迹）；
- `budget.ledger`、`pending`、`failures`；
- `cache_ledger`（本次 hit/miss）与 `reuse_ledger`（稳定的阶段键和引用观测）。

增量最终状态与“相同最终序列集合/配置的空白全量重建”比较时：

- 系数、误差归因、失败/待办顺序、阶段缓存键和复用身份必须逐字段相同；
- `cache_ledger` 的 hit/miss 动作和预算扣减可以因历史复用而不同，这是执行历史，
  不属于模型等价身份。`--check-equivalence` 会输出
  `equivalence-report.json` 并按此口径判定。

## 实际验证结果

本工作区执行结果：

```text
python -m unittest discover -s tests -t . -v
Ran 24 tests ... OK

python -m fixedseason verify --workspace artifacts/verification
PASS  failure_isolation
PASS  config_validation
PASS  command_validation
PASS  serial_concurrent_equivalence
PASS  incremental_vs_full_rebuild
PASS  budget_constraints
PASS  interrupt_resume
PASS  snapshot_contract
PASS  width_sign_diagnostics
PASS  fixed_point_cache_replay
overall_passed=True
```

验收证据文件：

- `artifacts/verification-report.json`：完整检查摘要与差异列表；
- `artifacts/sample-fit-report.json`：完整并发样例的逐序列系数、基列、正规方程
  规模、消元轨迹、误差归因、预算扣减与缓存账本；
- `artifacts/incremental-final-report.json` 与
  `artifacts/fresh-full-report.json`：同一最终状态的增量报告和空白全量重建报告；
- `artifacts/width-sign-report.json`：演示库 32/48 位宽对比；
- `artifacts/width-sign-flip-report.json`：真实检测到近零趋势序列在 16/48 位下
  斜率符号不一致（负/零），并归因到 `fixed_point_rounding`。

## 目录

```text
src/fixedseason/   实现包
tests/             unittest 验收测试
examples/          固定种子样本、配置、命令流
artifacts/         运行报告与验收证据
```

## 约束与说明

- 配置位宽为有符号整数域；正规方程原始累加器允许扩展精度，但提交消元前统一
  缩放至配置位宽范围内，缩放量会写入报告。
- 单批预算不能拆分单个序列；若单序列观测数超过批上限，该序列进入待办。
- `thread_pool_workers` 使用逻辑核数校验；实际并发上限受 Python 线程调度影响，
  但输出顺序、预算账本和产物不依赖完成时序。
- 快照目录替换在 Windows 上采用“旧目录备份 -> 新目录替换 -> 删除备份”的
  跨平台流程；同一工作区不应由多个进程同时写快照。
