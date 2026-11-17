# RoPE 长上下文外推诊断工作台

面向长上下文 LLM 位置编码外推问题的本地诊断工具。工作台对 RoPE、线性内插（Position Interpolation）与 NTK 感知缩放生成统一的位置旋转对，计算训练长度内/外注意力打分，追踪“位置向量 → 点积打分 → softmax 权重”的误差传播，扫描基频与缩放因子边界，并在真实异常注入下输出确定性 JSON 报告。

项目只依赖 Python 3.10+ 与 NumPy，不使用 GPU、容器、外部服务或网络账号。

## 快速开始

### Windows

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python tools\generate_samples.py
python -m rope_diag ^
  --config samples\position_config.json ^
  --samples samples\attention_samples.jsonl ^
  --injections samples\injections.json ^
  --state-dir .workbench_state ^
  --report artifacts\diagnostic_report.json
```

### Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/generate_samples.py
python -m rope_diag \
  --config samples/position_config.json \
  --samples samples/attention_samples.jsonl \
  --injections samples/injections.json \
  --state-dir .workbench_state \
  --report artifacts/diagnostic_report.json
```

若系统缺少 `venv/ensurepip`，需先安装对应的 `python3-venv`。本仓库实际验证环境为 Python 3.11.0rc1、NumPy 1.26.4。

## 常用命令

```bash
# 运行全部自动化测试
python -m unittest discover -s tests -v

# 中断/续跑验证：完成 encoding 快照后以退出码 0 退出
python -m rope_diag ... --stop-after-stage encoding
python -m rope_diag ... --resume

# 关闭缓存读取/写入，得到不含执行态缓存差异的诊断运行
python -m rope_diag ... --no-cache
```

主要选项：

- `--config`：位置编码配置 JSON。
- `--samples`：合成注意力样本 JSONL。
- `--injections`：异常注入清单 JSON；允许空列表。
- `--state-dir`：阶段缓存、原子快照和运行清单目录，默认 `.workbench_state`。
- `--report`：最终 JSON 诊断报告路径，默认 `artifacts/diagnostic_report.json`。
- `--resume`：要求已有未完成快照与当前版本、配置、注入清单、样本指纹完全一致。
- `--no-cache`：禁用缓存读写，仅用于字节级复现实验；仍保留运行清单。
- `--stop-after-stage`：验收/调试钩子；完成指定阶段和快照后返回 0。

## 样本与格式

仓库内已提供固定种子生成的样本：

- `samples/position_config.json`：三种方案、训练长度 32、外推目标长度 96、维度 16、并发数 4。
- `samples/attention_samples.jsonl`：9 条样本，覆盖训练内、相对距离等于训练长度的边界、混合外推与远距离外推。
- `samples/injections.json`：四个注入项，分别触发 `E_NONFINITE_INPUT`、`E_POSITION_OUT_OF_RANGE`、`E_BASE_ZERO`、`E_SCALE_INVALID`。
- `tools/generate_samples.py`：可按固定种子重新生成全部样本；特征使用 `numpy.random.default_rng` 与 SHA-256 派生种子。

详细字段定义见 [docs/input-format.md](docs/input-format.md)，算法与缓存语义见 [docs/design.md](docs/design.md)。

## 核心语义

- **相对距离**：`abs(query_position - key_position)`。`distance <= train_context_length` 归入 `in_train`；严格大于时归入 `out_of_train`；等号同时标记 `boundary_marked=true`。
- **RoPE**：标准频率 `theta_j = base^(-2j/d)`，角度为 `position * theta_j`。
- **线性内插**：使用 `position / scaling_factor` 计算角度，基频不变。
- **NTK 感知缩放**：使用 `base_eff = base * scaling_factor^(d/(d-2))`；`d=2` 时采用成对极限 `base * scaling_factor` 并记录公式版本。
- **参考扩展**：逐级误差以同基频、同维度的标准 RoPE 在真实位置上的解析扩展为参考。
- **误差阶段**：依次为 `position_vector`、`dot_score`、`normalized_attention_weight`；首个超差阶段即最终阶段判定，后续阶段偏差始终保留供审计。
- **边界判据**：`dimension_truncation` 表示某一旋转维度角度低于当前精度档位的分辨率；`scaling_overflow` 表示有效基频、频率、角度或旋转向量非有限。

## 报告内容

`artifacts/diagnostic_report.json` 主要包含：

- `stratified_statistics`：逐方案、逐相对距离的训练层/外推层样本数、打分均值、总体方差、最小值、最大值、P25/P50/P75/P90/P95/P99 与边界样本。
- `error_traces`：每个外推键的三阶段偏差、容差、是否超差、首个超差阶段与参考策略。
- `sensitivity.rankings`：逐方案、逐参数的敏感性排序；同分按参数值升序。
- `sensitivity.boundary_sensitive_samples`：触发维度截断或溢出的方案、样本、参数值、判据与触发维度。
- `injection_hits` / `injection_hits_by_sample`：注入项编号、业务码与逐样本汇总。
- `reuse_manifest`：`reused`、`recomputed`、`invalidated`、`bypassed` 四类逐阶段审计记录。
- `failures`：业务失败清单。
- `diagnostic_digest`：排除执行态 `reuse_manifest` 后，对所有诊断字段计算的 SHA-256；相同输入在冷/热缓存下保持一致。

业务码：`E_CONFIG_INVALID`、`E_CONFIG_MISMATCH`、`E_NONFINITE_INPUT`、`E_POSITION_OUT_OF_RANGE`、`E_BASE_ZERO`、`E_SCALE_INVALID`。

## 缓存、并发与恢复

- **阶段缓存键**：由方案参数指纹、样本内容指纹、精度档位和阶段依赖指纹派生。编码键不包含训练长度；打分键包含逐样本层分类指纹；追踪键包含打分键与容差。因此修改训练长度时，编码全部复用，只有分类真正变化的样本重算打分与追踪。
- **并发确定性**：线程池只并行独立阶段任务；每个任务使用 `(方案规范序号, 样本 ID, 阶段序号)` 稳定排序键，结果在提交报告前重新稳定排序。
- **原子快照**：阶段文件、缓存索引、运行清单和最终报告均先写同目录临时文件、`fsync`，再以 `os.replace` 原子替换。
- **续跑规则**：存在未完成快照时，版本/配置/注入/样本指纹任一不一致即输出 `E_CONFIG_MISMATCH` 并拒绝续跑。已完成运行允许以新配置开启新一代运行，同时复用内容键未变化的阶段结果。
- **退出信号**：`SIGINT`/`SIGTERM` 只设置停止标志；当前阶段提交快照后返回整数退出码 0。强制终止后，只可能丢失尚未原子替换的临时文件。

`reuse_manifest` 是执行态审计，冷/热缓存下必然不同，这是增量验证所需信息；诊断内容的一致性由 `diagnostic_digest` 保证。若需要整份报告在不同运行间字节级一致，请使用 `--no-cache` 或全新的 `--state-dir`。

## 验证

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖 12 个场景：严格配置校验、三方案元素级确定性、边界分层、三阶段首个超差判定、敏感性边界与同分决断、四类注入、CLI 失败清单、冷/热缓存、训练长度/基频/单样本内容变更的精确失效范围、串行-并发一致性、中断续跑和 `E_CONFIG_MISMATCH` 拒绝。实际结果见 [docs/validation.md](docs/validation.md)。

额外验证中，两次 `--no-cache` 报告字节级一致；真实 `SIGTERM` 实验在完成 `encoding` 快照后以退出码 0 中断，可由 `--resume` 继续。

## 目录

```text
rope_diag/       编码、注意力、追踪、敏感性、缓存、快照、引擎与 CLI
samples/         固定种子配置、JSONL 样本与注入清单
tools/           样本生成器
tests/           unittest 验收测试
docs/            输入格式、设计与验证说明
artifacts/       最终诊断报告与验证摘要
```
