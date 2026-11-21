# 验收运行报告

## 1. 环境

- 操作系统：Linux（容器/沙箱内验证），另提供 Windows PowerShell 引导 `run_windows.ps1`
- Python：3.11.0rc1（虚拟环境 `.venv`，由 `virtualenv` 创建）
- 依赖：`numpy==1.26.4`、`PyYAML==6.0.2`、`pytest==8.3.3`（见 `requirements.txt`）
- 无 Docker、无外部服务、无网络数据集，全部数据由固定种子 `20240917` 本地生成

## 2. 复现步骤

```bash
# 1. 创建虚拟环境并安装依赖
/home/zsl/.feedback/python-base/bin/virtualenv --clear -p python3.11 .venv
.venv/bin/python -m pip install -r drift_monitor/requirements.txt

# 2. 回归测试
cd drift_monitor
PYTHONPATH=src ../.venv/bin/python -m pytest tests -q

# 3. 确定性样本生成
PYTHONPATH=src ../.venv/bin/python scripts/generate_sample_data.py \
  --config configs/default_monitor.yaml --tier small --output output/sample

# 4. 端到端验收场景（增量 vs 全量、并发、快照、翻转）
PYTHONPATH=src ../.venv/bin/python scripts/validate_scenarios.py
```

## 3. 实际验证结果

### 3.1 回归测试
- 结果：`33 passed`
- 覆盖：配置校验与错误码、确定性生成与缺失标记、分窗短尾规则、五种度量、
  近似/精确分位可观测差异、策略选择路径切换、定点量化与判定翻转、
  增量/全量逐字段一致、并发确定性、失败清单顺序、快照恢复与陈旧判定、
  原子替换无残留、JSON/SVG 报告完整性。

### 3.2 自适应策略路径（三档规模）

| 规模档 | 样本量 | 特征乘法器 | 选中策略 | 近似分位 | 窗数 |
| --- | --- | --- | --- | --- | --- |
| small | 480 | 1 | `tiny_exact` | False | 4 |
| medium | 2400 | 2 | `standard_mixed` | False | 4 |
| large | 12000 | 4 | `large_approx` | True | 4 |

`large` 档真实切换到近似分位路径，测试
`test_large_tier_observably_switches_quantile_path` 断言近似与精确原始得分不同。

### 3.3 增量与全量一致性（13 条编辑命令序列）
编辑序列包含：分窗数量 4→6、阈值修改、量化档位 coarse/fine 切换、单列度量替换
（`age`→`mean_shift`）、规模档 small→medium、并发度→8。

- `business_fields_equal = true`：逐列逐窗得分、判定、翻转清单、决策记录、
  数据指纹、预算扣减序列、失败清单完全一致。
- `required_keys_equal = true`：内容指纹派生的缓存键集合与运行历史无关。
- 增量末轮：复用 70 个产物、重算 0；空白工作区全量：重算 70 个产物。
- 单列度量替换时仅该列 4 个窗的 metric 键失效，其余 20 个列窗物理复用。

### 3.4 并发一致性
并发度 1/4/8 三次运行的业务字段与预算扣减序列完全相等
（`concurrency_deterministic = true`），失败清单按稳定排序键排列。

### 3.5 定点判定与翻转
- 量化档位真实改变整数得分，例如 `0.123456789` 在 coarse(2 位) 为 `12`，
  在 fine(6 位) 为 `123457`。
- 每条相邻档位记录含两档整数分、保留值、整数差、容差、首次超差环节、
  高两位宽参考分。
- 边界单元测试 `test_judgement_flip_at_quantization_boundary` 构造真实判定翻转：
  粗档显著、细档不显著，`first_breach_stage = judgement`。
- 默认数据下多数差异在 `quantize` 环节即超容差，业务判定翻转需要得分贴近阈值的
  边界条件，已由定点单元测试确定性覆盖。

### 3.6 快照与恢复
- 健康续跑：`resume_status = resumed`。
- 修改阈值后续跑：`resume_status = E_SNAPSHOT_STALE`。
- 版本门控：相同输入重复提交不递增全局版本；配置变更后版本递增。
- 原子替换：`snapshot/tmp` 无 `snap_*` 残留，`current/manifest.json` 与
  各阶段 `artifact.json` 完整。

### 3.7 报告产物
- JSON：`output/<run>/reports/latest_report.json`，含逐列逐窗得分、判定、
  翻转清单、失效/复用统计、预算序列、失败清单、快照元数据。
- 可视化：`output/<run>/reports/heatmap.svg` 零依赖 SVG 热力图。
- 样本：`output/sample/{reference,current}_small.json` 与 `summary.json`，
  缺失值同时以 `null` 与 `__missing` 布尔标记保留。

## 4. 已知边界
- 判定翻转是阈值附近的窄带现象；真实数据未触发时由定点单元测试保证机制正确。
- 高基数列在 `large_approx/huge_stream` 策略下自动改用卡方统计，决策记录中
  `metric_overrides` 明确列出被覆盖的列。
