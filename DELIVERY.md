# 定点 Ridge / Lasso 确定性实验台

本目录首版自建了样本、配置、编辑命令、确定性样本生成器、定点求解内核、误差归因、内容指纹账本、稳定预算调度器和原子快照。核心计算只依赖 Python 标准库整数/Decimal；NumPy 仅用于 float64 无量化对照路径。

## 快速复现

当前环境的 `python3 -m venv` 因系统缺少 `ensurepip` / `python3.11-venv` 且 `sudo` 需要密码，未能直接创建；已用环境中的 `virtualenv .venv` 得到等价虚拟环境。标准 WSL Ubuntu 准备命令为：

```bash
sudo apt update
sudo apt install -y python3.11-venv python3-tk
python3 -m venv .venv
. .venv/bin/activate
pip install numpy pytest
```

本仓库已用 `virtualenv .venv && .venv/bin/pip install numpy pytest` 完成验证。tkinter 不是 CLI/计算内核依赖；当前沙箱中系统 tkinter 还缺少 `libXft.so.2`，安装 `python3-tk` 及相关 X11 运行库后可恢复。

```bash
# 生成确定性本地样本（已生成在 data/samples.jsonl）
.venv/bin/python scripts/generate_samples.py --count 18 --seed 396

# 校验样本、配置和编辑流
.venv/bin/python -m fixed_lab.cli validate \
  --samples data/samples.jsonl \
  --config configs/lab.json \
  --edits edits/demo_edits.jsonl

# 一键验收：增量/全量、串行/线程、精确失效、快照
.venv/bin/python scripts/verify_acceptance.py

# 全量测试
.venv/bin/python -m pytest -q
```

## 真实 CLI 演示产物

```bash
# 初始版本
.venv/bin/python -m fixed_lab.cli run \
  --samples data/samples.jsonl --config configs/lab.json \
  --out-dir runs/01_initial --workers 4

# 应用 edits/demo_edits.jsonl 的增量版本，复用 ledger/reuse_ledger.json 与 ledger/cache
.venv/bin/python -m fixed_lab.cli run \
  --samples data/samples.jsonl --config configs/lab.json \
  --edits edits/demo_edits.jsonl \
  --out-dir runs/02_incremental_edits --workers 4

# 从空白缓存全量重建同一最终状态
.venv/bin/python -m fixed_lab.cli run \
  --samples data/samples.jsonl --config configs/lab.json \
  --edits edits/demo_edits.jsonl \
  --out-dir runs/03_full_rebuild \
  --cache-dir ledger/full_rebuild_cache \
  --ledger ledger/full_rebuild_ledger.json \
  --workers 4
```

`runs/02_incremental_edits/report.json` 与 `runs/03_full_rebuild/report.json` 已验证逐字段一致。主演示末尾切换了定点位宽/量化档位，因此所有定点派生键按设计整体失效；若要看大量缓存命中，使用 `edits/reuse_demo_edits.jsonl`，对应产物在 `runs/04_reuse_initial`、`runs/05_reuse_after_edit` 和 `ledger/reuse_demo_ledger.json`。

## 代码地图

- `fixed_lab/config.py`：配置 JSON Decimal 解析、正则档位/区间/间距、惩罚枚举、位宽档位、预算和求解参数校验。
- `fixed_lab/dataset.py`：JSONL 逐行校验、唯一 ID、有限数、维度校验、样本表、编辑命令流应用。
- `fixed_lab/fixedpoint.py`：2 的/10 的十进制定点口径、半入整、带余数的整数除法。
- `fixed_lab/features.py`：特征定点化、量化步长/整数/残差台账、加权定点协方差和精确整数平方比较。
- `fixed_lab/accumulation.py`：按目标列、源列、样本 ID 的稳定顺序累加 `X'WX` 与 `X'Wy`，逐项内容指纹缓存。
- `fixed_lab/solvers.py`：Ridge Bareiss 整数消元、主元记录和降 alpha 回退；Lasso 升序坐标下降、整数软阈值、目标值和收敛边界记录。
- `fixed_lab/pathing.py`：系数向量、非零集合、目标函数、残差统计、零值半步规则、符号翻转/进入/退出事件。
- `fixed_lab/attribution.py`：float64 原始参考、仅量化、定点累加/float 求解、整数求解、路径汇总的逐级差值和事件归因。
- `fixed_lab/ledger.py`、`fixed_lab/cache.py`：内容指纹键、样本/列引用、失效原因、复用事件和原子 JSON 对象存储。
- `fixed_lab/scheduler.py`：稳定排序键、单批 MAC 预算、单列配额、`E_BUDGET_DEFERRED` / `E_COLUMN_QUOTA_DEFERRED` 待办、串行/线程池执行。
- `fixed_lab/snapshot.py`：临时目录写完整快照，再按报告版本哈希整体提交并原子更新 `current.json`。
- `scripts/verify_acceptance.py`：机器可复核验收脚本，结果写入 `artifacts/verification_result.json`。

## 定点与决断口径

设定点小数位宽为 `d`，量化档位为 `q`：

- 细粒度 scale：`S = 10^d`。
- 量化网格整数：`G = 10^q`，有效步长：`G / S = 10^(q-d)`。
- 特征直接按有效步长做 `ROUND_HALF_UP`（平局远离零），因此量化残差不超过有效步长的一半。
- 正规方程累加尺度为 `AS = 10^(3d-m)`，其中 `m=accum_rounding_margin`；每个 `w*x_i*x_j` 细尺度项先按 `10^m` 做带余数半入整，再按固定顺序相加。
- Ridge 求解 `(Acc + round(alpha*AS)I) B = S*bAcc`，物理系数为 `B/S`。
- Lasso 坐标更新为 `B_j = round_div(soft_threshold(S*b_j - sum_k G_jk B_k, round(alpha*AS*S)), G_jj)`。

边界规则固定为：

- 相关性使用协方差交叉整数的平方比较，精确等于阈值属于匹配，并标记 `boundary_exact_threshold`；略低于阈值但距离不超过量化半步传播带时，按 `include_below_threshold` 策略纳入并标记边界。
- 收敛阈值精确相等，或未收敛但 `max_delta - tol <= grid/2` 时，按配置 `converge` 策略判定收敛并记录边界轮次。
- `2*|B_j| <= grid` 视为零值半步带；恰好 `2*|B_j| == grid` 为半步平局，默认策略 `zero` 将其归入零集合并打 `boundary` 标记。

## 缓存与精确失效

缓存键由规范化 JSON 的 SHA-256 派生：

- 特征单元格：样本 ID、列名、精确 Decimal 值。
- 列产物：列内容指纹、位宽和量化档位。
- MAC 项：两个特征单元格指纹、权重单元格、目标单元格（RHS）、定点/累加参数。
- 系数档位：正规方程指纹、样本指纹集合、列指纹集合、alpha、求解器指纹和位宽。
- 路径汇总：有序 alpha 键集合、各档系数键、零值策略。

因此替换一个样本时，只有该样本的 27 个 `p=6` MAC 项失效，其余 `17*27=459` 项命中；只改一列中的一个样本值时，只有该样本涉及该列的 `p+1=7` 个矩阵/RHS 项失效。系数档位引用完整样本集合和列集合，所以输入变化后档位结论会按其依赖键失效；未变化的底层 MAC、相关性或路径对象可直接复用。

`report.json` 中的 `lineage_checklist` 只含稳定的 `artifact_id/stage/cache_key/refs/depends_on`，增量与空白重建逐字段相同。`ledger/reuse_ledger.json` 的条目也按同一组键稳定排序；其中 `status=reused/rebuilt` 是运行态证据：空白缓存必然全部 `rebuilt`，增量运行才会出现 `reused`。

## 实际验证结果

验证时间：2026-11-27（Asia/Shanghai）；实际解释器为 Python 3.11.0rc1，`.venv` 中 NumPy 2.4.6、pytest 可用。

最终结果以 `artifacts/verification_result.json` 为准。最近一次验证包含：

- `.venv/bin/python -m pytest -q`：14 个测试全部通过（进度点 14 个，退出码 0；摘要已存 `artifacts/pytest_output.txt`）。
- `incremental_equals_blank_rebuild = true`。
- `serial_equals_thread_pool = true`。
- 单样本替换：MAC `rebuilt=27`、`reused=459`。
- 单列单点修改：MAC `rebuilt=7`，其余同一样本的非相关列对仍复用。
- 预算计划显式产生 `E_BUDGET_DEFERRED` / `E_COLUMN_QUOTA_DEFERRED` 待办；并发只在线程中执行纯计算，回主线程后按稳定键排序合并。

## 复核入口

- 主报告：`runs/02_incremental_edits/report.json`
- 完整求解/累加 trace：`runs/02_incremental_edits/trace.json`
- 增量账本：`ledger/reuse_ledger.json`
- 复用演示账本：`ledger/reuse_demo_ledger.json`
- 验收机器结果：`artifacts/verification_result.json`
- 快照目录：`snapshots/<report-version-hash>/` 与 `snapshots/current.json`
