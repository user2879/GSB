# GEMM/BMM 数值精度与误差传播可观测组件

这是一个可集成、可复现的本地 GEMM/BMM 实验组件。主路径使用 float32 分块乘加，参考路径使用 float64 串行 K 升序累加；每次运行都会生成逐元素误差归因、分块/循环序对比、分层敏感性和异常注入证据。

当前实现不使用 Docker、容器或外部数据源。默认样本全部由固定种子在本地生成。

## 快速开始

环境要求：Python 3.10+、NumPy、pytest。

```bash
python3.10 -m pip install --user -r requirements.txt
python3.10 main.py --config configs/default_experiment.json
python3.10 -m pytest -q
```

也可以直接运行示例入口：

```bash
python3.10 examples/run_experiment.py --config configs/default_experiment.json
```

默认输出：

- `artifacts/precision_report.json`：完整机器可读报告。
- `artifacts/precision_report.md`：人工审计表与结论摘要。

如需把确定性样本物化到本地，可增加：

```bash
python3.10 main.py --config configs/default_experiment.json --samples-dir artifacts/samples
```

## 项目结构

- `src/gemm_precision/kernel.py`：float32 分块 GEMM/BMM、边界块裁剪、线程池任务与稳定排序键。
- `src/gemm_precision/reduction.py`：K 块局部结果归并拓扑。
- `src/gemm_precision/reference.py`：float64 串行参考路径。
- `src/gemm_precision/diagnostics.py`：逐元素重放、局部舍入、K 带归因与 top 非零项。
- `src/gemm_precision/analysis.py`：绝对/相对误差、超差判定、分层表和组合摘要。
- `src/gemm_precision/samples.py`：四类确定性合成样本及特殊值位置追踪。
- `src/gemm_precision/injection.py`：四类异常注入与失败隔离。
- `src/gemm_precision/runner.py`：完整实验编排、串行/并发逐字段一致性验证。
- `src/gemm_precision/report.py`：JSON/Markdown 原子写盘。
- `configs/default_experiment.json`：默认实验与独立注入配置。
- `tests/`：内核、参考、归因、分层、异常、并发、复现性测试。

## 计算与累加语义

数学目标为：

```text
GEMM: C[m,n] = alpha * sum_k(A[m,k] * B[k,n]) + beta * C0[m,n]
BMM:  对每个 batch b 独立执行同一 GEMM 语义
```

主路径为 float32：

1. 每个输出块按 `(BM, BN, BK)` 裁剪边界块。
2. 每个 K 块使用 float32 局部累加缓冲。
3. 各 K 块局部结果按循环序对应的归并拓扑组合。
4. 最后在 float32 中执行 `alpha` 缩放和 `beta*C0` 加法。

参考路径为 float64、逐输出元素、K 升序串行乘积累加。主路径与参考路径始终使用同一份 float32 输入数组。

相对误差定义为：

```text
abs_error = abs(main - reference)
relative_error = abs_error / max(abs(reference), 1e-30)
```

超差判定使用标准形式：

```text
abs_error > atol + rtol * abs(reference)
```

`atol` 与 `rtol` 必须有限且非负，不能同时为零。

## 循环序与 K 块归并拓扑

为避免“仅仅交换工作项遍历顺序、但每个输出元素的 K 累加完全相同”导致循环序无法产生数值差异，组件把循环序显式绑定为三种可审计的局部归并拓扑：

| 循环序 | 块内遍历 | K 块归并 |
|---|---|---|
| `ijk` | `i -> j -> k` | 前向顺序合并 |
| `ikj` | `i -> k -> j` | 反向顺序合并 |
| `jik` | `j -> i -> k` | 平衡二叉树合并 |

三种方式数学等价，但 float32 舍入链不同。每个输出元素的 `accumulation_order_id` 会记录循环序、K、BK、块数、块内 K 升序链和归并拓扑。

## API 示例

```python
import numpy as np
from gemm_precision import BlockShape, gemm, reference_gemm

rng = np.random.default_rng(7)
A = rng.normal(size=(16, 129)).astype(np.float32)
B = rng.normal(size=(129, 12)).astype(np.float32)
C = rng.normal(size=(16, 12)).astype(np.float32)

main = gemm(
    A, B, C,
    alpha=1.0,
    beta=0.25,
    block_shape=BlockShape(4, 6, 8),
    loop_order="jik",
    max_workers=4,
)
reference = reference_gemm(A, B, C, alpha=1.0, beta=0.25)

print(main.c.shape, reference.c.shape)
print(main.element_metadata[0])
```

BMM 输入形状为 `A[B,M,K]`、`B[B,K,N]`、`C[B,M,N]`。批大小不一致会得到 `E_SHAPE_MISMATCH`，错误记录包含三张量形状、首个无效批索引和阶段名。

## 误差归因与分层

对每个超差元素，报告包含：

- 主路径值、参考值、绝对误差、相对误差、容差阈值。
- 累加顺序标识、链长、K 块数、rounding step 数。
- 每个 K 带的绝对局部舍入贡献。
- 贡献最大的非零乘积项：A/B 下标、K 下标、float32/float64 乘积、局部舍入贡献。
- 排序规则：按 `abs(local_rounding_contribution)` 降序，同分按累加位置升序。
- 四级传播摘要：K 块局部累加、K 块归并、alpha/beta 收尾、相对参考总误差；保留最差块/归并事件和完整级联哈希。

分层维度：

- `M`：连续行带。
- `N`：连续列带。
- `K`：按元素主导的局部舍入 K 带归属。
- `BATCH`：批次索引带。

每层统计 `max_relative_error`、超差比例、误差归因份额和主导结论。超过 `layer_exceed_limit` 的层会进入 `layer_exceedance_summary`，并记录对应的 block/loop 组合。

## 确定性样本与异常注入

默认四类样本：

1. `small_mn`：小规模、非整除块边界。
2. `large_k`：长 K 链，放大 K 块归并误差。
3. `large_batch`：6 批 BMM，`beta != 0`。
4. `special_values`：种子选择的精确零和正/负 float32 次正规数。

特殊值位置、值、float32 位模式和选择序号都写入报告。样本数组摘要、seed、生成策略和特殊值列表可用于追溯。

默认独立注入项：

| 注入编号 | 类型 | 期望业务码 |
|---|---|---|
| `INJ-NONFINITE-001` | 非有限元素 | `E_NONFINITE_INPUT` |
| `INJ-SHAPE-002` | BMM 批大小不一致 | `E_SHAPE_MISMATCH` |
| `INJ-BLOCK-003` | 零分块参数 | `E_BLOCK_INVALID` |
| `INJ-TOLERANCE-004` | 零容差 | `E_TOLERANCE_INVALID` |

注入结果包含 `hit`、实际业务码、阶段、形状和批索引。正常组合若在执行或误差统计阶段抛异常，会记录失败并继续后续组合。

## 并发与可复现性

每个输出块是一个线程池任务，任务键为 `(batch_index, block_row, block_col)`。结果按提交键顺序收集与写入，不按线程完成顺序写入。

完整实验对每个组合执行两次：

- 配置并发度 `max_workers`。
- 单线程 `serial_workers=1`。

逐字段比较 C 位模式、输出摘要、逐元素累加标识、任务顺序、超差归因、分层表、超差层和审计摘要。任一不一致都会产生 `E_CONCURRENCY_MISMATCH`。

报告不写时间戳；JSON 使用固定键排序、`allow_nan=false`、临时文件 `fsync` 后 `os.replace`。同一配置重跑应得到字节级相同 JSON（测试已覆盖）。

## 实际验证记录

验证环境：Linux/WSL 本地环境、Python 3.10.12、NumPy 2.2.6、pytest 9.2.0；未使用 Docker 或其他容器。

```text
python3.10 -m pytest -q
19 passed in 2.90s
```

默认完整运行结果：

```text
samples:              4
combinations:         24
injections hit:       4/4
failures:             4        # 四个预期注入失败
concurrency equal:    True     # 24/24 组合逐字段一致
layer exceedances:    15
report digest:        96b8e19d6aadaf958f70e8d955b8ad5603f3f5e9e9b2478fd6457bd7a0f6b21b
JSON sha256:          eb9cf6c60e735cc946e251000bfe24796efd7353c029c14acde98b4532c69095
```

同一配置连续运行两次后，上述 JSON SHA-256 完全一致；CLI 输出也逐字节一致。

本次运行中，三种循环序在四个样本的每个固定分块下均得到不同的输出/审计摘要；例如 `large_k` 在 `BM7/BN5/BK4` 下的最大绝对误差分别为 `ijk=8.703e-06`、`ikj=8.335e-06`、`jik=2.822e-06`。分块参数也从另一个维度改变了 K 块划分与归并深度。所有组合都与 float64 串行参考进行了逐元素比较。

完整结果见 `artifacts/precision_report.json` 与 `artifacts/precision_report.md`。

## 约束与说明

- 内核为面向可观测性的纯 Python/NumPy 标量实现，不是高性能 BLAS 替代品。
- 每个输出块暂存各 K 块局部矩阵，内存复杂度与该块的 `K/BK` 块数相关；大规模生产使用时应按硬件预算限制块尺寸。
- 报告中的 `failures` 包含有意注入的失败；`injections[*].hit=true` 表示异常按预期命中。
- 相对误差对零参考使用 `1e-30` 分母下限，报告字段明确记录该策略。
