# Feature Crossing & Reduction Diagnostics Workbench

面向确定性表格样本的特征交叉、编码、标准化与降维精度诊断工作台。项目在本地运行，不依赖 Docker、外部数据集或网络服务。

## 功能

- 从 YAML 配置与列描述生成固定种子的表格数据，并完成数值中位数填充、类别众数填充与类别整数编码。
- 按阶数升序、列名序列字典序枚举类别交叉组合；使用 SHA-256 确定性编码，超过基数预算时支持 `hash_bucket` 截断或 `skip`。
- 实现 PCA、截断随机投影、方差筛选三条路径；PCA 按最大绝对分量符号归一，随机投影矩阵由配置指纹派生种子。
- 在 `float32` 与 `float64` 两档位重复交叉编码、标准化、投影和最终指标计算，输出逐样本四阶段差值、首次越界环节、主误差环节与精度敏感样本。
- 对保留维度、交叉特征和判定阈值执行近并列边界诊断；判定翻转且超出容差时记录 `E_BOUNDARY_SENSITIVE`，并区分 `encoding_rounding` 与 `projection_accumulation`。
- 使用内容指纹缓存交叉、标准化、各降维路径和指标产物；参数变化只失效对应路径及其下游。
- 使用不可变阶段目录和原子清单提交快照；启动时校验全局版本、配置哈希和精度指纹，不一致时拒绝旧产物并记录 `E_SNAPSHOT_STALE`。
- 线程池并发评估阶数 × 路径组合；输出按阶数、路径和样本 ID 派生稳定键排序，保证串行/并行逐字段一致。

## 环境准备

本仓库固定使用 Python 3.11、`numpy==2.2.6`、`PyYAML==6.0.2` 和 `pytest==8.3.5`。

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

当前 Linux/WSL 环境（系统缺少 `ensurepip` 时可使用官方引导脚本）：

```bash
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

## 使用

```bash
.venv/bin/python -m feature_diag validate --config sample/experiment.yaml
.venv/bin/python -m feature_diag run \
  --config sample/experiment.yaml \
  --out artifacts/reports/diagnostic-report \
  --cache-dir artifacts/runtime-final-cache \
  --snapshot-dir artifacts/runtime-final-snapshot \
  --max-workers 4
```

输出：

- `report.json`：完整机器可读报告，含逐样本四阶段误差、边界诊断、缓存账本和快照信息。
- `report.md`：可读摘要、敏感样本、方案误差表、边界与缓存明细。
- `cli-output.json`：本次 CLI 摘要（由重定向命令生成）。

非法配置样例：

```bash
.venv/bin/python -m feature_diag validate --config sample/experiment_invalid.yaml
```

快照恢复测试：

```bash
# 只提交 crossing 阶段后退出，CLI 返回 75
.venv/bin/python -m feature_diag run --config sample/experiment.yaml \
  --out artifacts/reports/partial \
  --cache-dir artifacts/runtime-recovery-cache \
  --snapshot-dir artifacts/runtime-recovery-snapshot \
  --stop-after-stage crossing

# 再次启动，只从完整替换的 crossing 产物恢复
.venv/bin/python -m feature_diag run --config sample/experiment.yaml \
  --out artifacts/reports/recovered \
  --cache-dir artifacts/runtime-recovery-cache \
  --snapshot-dir artifacts/runtime-recovery-snapshot
```

## 配置结构

主配置见 `sample/experiment.yaml`，列描述见 `sample/data_spec.yaml`。核心字段：

- `version`、`engine.schema_version`：快照全局版本组成部分。
- `seed`、`data_spec`：确定性数据生成种子与本地列描述。
- `tolerances`：四阶段误差容差，阶段名固定为 `cross_encoding`、`standardization`、`projection`、`final_metric`。
- `boundary.neighborhood`：近并列邻域；`boundary.tie_tolerance`：翻转敏感阈值。
- `scenarios[]`：采样规模档、交叉阶数、类别列白名单、基数预算、溢出策略、三条降维参数、下游判定和并发上限。
- `cache_dir`、`snapshot_dir`：内容缓存与阶段快照目录；相对路径按配置所在目录解析。

## 业务码

- `E_CONFIG_INVALID`：逐方案配置错误；错误方案跳过，其余方案继续。
- `E_PRECISION_SENSITIVE`：某一输入样本的最终判定在两档位间不同。
- `E_BOUNDARY_SENSITIVE`：近并列项发生判定翻转且档位差值超过配置容差。
- `E_SNAPSHOT_STALE`：已有快照的全局版本、配置哈希或精度指纹不一致，旧产物被拒绝。

## 测试

```bash
.venv/bin/pytest -q
```

测试覆盖：六类配置错误隔离、交叉排序与预算降级、数据确定性、PCA 符号、随机投影复现、四阶段误差归因、编码/投影两类边界翻转、缓存定向失效、阶段恢复、陈旧快照拒绝，以及并发/串行逐字段一致。

## 实际验收结果

最终样例结果见 `artifacts/reports/diagnostic-report/`，完整恢复结果见 `artifacts/reports/diagnostic-report-resumed/`。

- `pytest`：16 项通过。
- 主样例：27 个阶数 × 路径任务，42 个精度敏感样本，42 条 `E_BOUNDARY_SENSITIVE`，357 条边界候选。
- 首次缓存账本：144 个产物未命中；完整恢复运行：144 个产物通过完整快照复用。
- 恢复报告分析指纹与首次运行一致：`caffa36269dca162553672022f9ca5ffb6704d7d2c6e4b3952007d0a8019b9bf`。
- 串行/并行逐字段一致性、陈旧快照拒绝结果见 `artifacts/reports/verification-summary.json`。

更详细的实现说明见 `docs/ARCHITECTURE.md`，验证记录见 `VERIFICATION.md`。
