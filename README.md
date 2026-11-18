# 表格二分类概率校准与增量复用服务

本工作区提供一个完全本地的确定性后处理服务：按 YAML 描述生成表格数据，训练逻辑回归和决策树，用四种方案校准原始概率，计算 ECE/MCE 与阈值网格指标，并以内容指纹驱动四阶段精确复用。

## 快速开始

要求 Python 3.10+、`numpy`、`PyYAML`，不依赖网络服务、容器或 GPU。Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m probcal run --config configs\calibration.yaml --state .probcal_state
```

Linux/WSL：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m probcal run --config configs/calibration.yaml --state .probcal_state
```

工作区还提供 `scripts/setup.ps1`、`scripts/validate.ps1` 与命令示例 `examples/commands.md`。完整验收可执行：

```powershell
.\scripts\validate.ps1
```

## 输入与产物

- `configs/calibration.yaml`：模型候选、四种校准方案、评测口径、并发数和快照间隔。
- `data/data_spec.yaml`：固定种子、96 条样本、数值/类别/缺失/冗余列和不平衡目标列描述。
- `data/sample_overrides.yaml`：单样本替换输入；通过命令编辑而不是手工修改。
- `.probcal_state/cache/<stage>/<cache-key>/artifact.json`：四阶段内容寻址产物。
- `.probcal_state/manifest.json`：全局版本、配置哈希、数据指纹和各阶段产物版本。
- `.probcal_state/snapshots/vNNNNNNNNN/`：不可变整体快照，包含清单与 JSON 报告。
- `.probcal_state/latest_report.json`：最近一次运行报告。

运行报告包含逐模型逐方案校准概率、ECE/MCE、阈值网格、逆序样本对、复用/重算阶段、失效输入、快照版本和失败清单。

## 四阶段与缓存边界

| 阶段 | 产物 | 键依赖 |
| --- | --- | --- |
| `data` | 确定性生成、训练折填充/编码、划分 | 配置/数据描述哈希、样本替换、特征类型、种子、划分 |
| `model` | 原始概率、损失轨迹、模型参数 | 数据键、模型候选、定点概率设置 |
| `calibration` | 校准概率、拟合参数、逆序记录 | 模型键、启用的方案、分箱数、正则强度 |
| `metrics` | ECE/MCE、阈值网格、边界样本 | 校准键、数据验证标签、评测分箱/阈值/定点口径 |

阶段仅在当前缓存键等于上一已提交快照中的键时复用。切换校准方案只重算校准和指标；修改启用方案的分箱/正则只重算校准和指标；修改评测分箱或阈值步长只重算指标；替换样本使四阶段全部重算。已关闭方案不进入有效参数指纹，因此修改它不会触发无意义重算。跳过阶段不会调用其计算函数；审计报告记录每阶段实际读取来源。

缓存键不包含仅影响调度的 `concurrency.max_workers` 和 `snapshot.commit_interval`，因此并发运行与串行运行得到相同阶段键与相同逐字段结果。

## 受控编辑

快照提交后，直接修改 YAML 再执行 `run` 会返回 `E_CONFIG_MISMATCH`。需要改动时使用 `edit`：

```powershell
$PY = ".\.venv\Scripts\python.exe"
& $PY -m probcal edit --config configs\calibration.yaml --state .probcal_state --edit examples\edits\01_switch_and_bins.yaml
& $PY -m probcal edit --config configs\calibration.yaml --state .probcal_state --edit examples\edits\03_threshold.yaml
& $PY -m probcal edit --config configs\calibration.yaml --state .probcal_state --edit examples\edits\04_replace_sample.yaml
& $PY -m probcal verify --state .probcal_state
```

支持 `switch_calibration`、`set_bins`、`set_regularization`、`set_threshold_step`、`replace_sample`。一个编辑文件可包含多个操作；编辑失败不会写入配置。`--expect-version N` 可用于校验续跑版本，不一致返回 `E_CONFIG_MISMATCH`。

## 确定性规则

- 数据生成按列描述顺序消费固定 `PCG64` 随机流；划分使用另一个固定种子并对两类分层。
- 数值缺失用训练折中位数，类别缺失用训练折众数；类别编码只用训练折并保留未知类别槽。
- 逻辑回归按训练样本 ID 升序、固定批量顺序和固定学习率执行；决策树按特征索引、阈值升序计算增益，只有严格更优才替换，平分保留首个候选。
- 校准器只接收训练折概率和训练折标签。模型预测函数只接收验证折特征，不读取验证折标签。
- 校准概率与指标用整数定点。`decimal_places` 决定定点位宽与指标刻度，`quantization_levels` 决定概率量化档位。
- 校准误差分箱采用左闭右开，`1.0` 归最后一箱；阈值预测采用 `p >= threshold`。恰在分箱/阈值边界的样本分别记录在 `boundary_sample_ids`。
- 候选、指标、阈值、逆序和失败清单均按稳定键排序；线程完成顺序不影响输出。

## 业务码与关闭方案

未知特征列、特征类型与数据描述不符、目标列缺失、分箱数为零、阈值步长非正、并发数非正等均汇总为 `E_CONFIG_INVALID`，逐项写入失败清单，进程返回非零而不是崩溃或静默修正。外部配置哈希或指定快照版本不一致返回 `E_CONFIG_MISMATCH`。非法编辑操作返回 `E_COMMAND_INVALID`。

方案 `enabled: false` 时不会进入校准任务列表，校准产物中不会出现该方案的空占位或拟合参数。

## 快照与退出

每个阶段先在快照目录同分区的临时目录写完整产物，再用 `os.replace` 原子提交。清单按 `snapshot.commit_interval` 阶段间隔提交，运行结束生成完整不可变快照。收到 `SIGINT`、`SIGTERM` 或 Windows `SIGBREAK` 时只设置停止标志，当前阶段完成后提交快照，报告标记 `interrupted`，进程以整数 `0` 退出。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v tests
.\.venv\Scripts\python.exe scripts\validate.py
```

测试覆盖六类配置错误、定点/边界规则、关闭方案、验证折标签隔离、每次编辑后的增量/全量逐字段一致、并发等价、清单完整性和中断快照。实际验收记录见 `VALIDATION.md` 与 `artifacts/validation/acceptance_summary.json`。
