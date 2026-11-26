# 多方案位置编码外推误差复核

本地确定性的定点位置编码、旋转注意力打分、逐级 float64 真值归因、内容指纹缓存、预算调度和原子快照实现。仅依赖 Python 3.10+ 与 NumPy；不使用 Docker、GPU、外网或真实账号。

## 快速运行

```bash
python3 --version
python3 -m pip install --user -r requirements.txt
# 若当前解释器是不允许 --user 的虚拟环境：
# python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 scripts/verify.py
```

`scripts/verify.py` 会重新生成 `examples/` 下的确定性配置、样本、命令流和注入配置，运行测试与全部验收场景，并写出：

- `docs/verification-report.md`：人类可读的实际验证结果。
- `artifacts/runs/verification-summary.json`：机器可读汇总。
- `artifacts/runs/replay-report.json`：十步命令流的完整增量账本与一致性载荷。
- `artifacts/runs/*budget*.json`、`quota-exceeded.json`、`injection-run.json`、`injection.stderr`。

常用命令：

```bash
python3 -m rope_audit generate --out-dir examples
python3 -m rope_audit validate \
  --config examples/config.json \
  --samples examples/samples.jsonl \
  --commands examples/commands.jsonl

python3 -m rope_audit run \
  --config examples/config.json \
  --samples examples/samples.jsonl \
  --drain --workers 2 \
  --output artifacts/runs/run.json

python3 -m rope_audit replay \
  --config examples/config.json \
  --samples examples/samples.jsonl \
  --commands examples/commands.jsonl \
  --output artifacts/runs/replay-report.json
```

## 目录

- `rope_audit/config.py`：配置 JSON 校验、行号/字段定位、量化档位和容差校验。
- `rope_audit/samples.py`：确定性样本生成与 JSONL 逐行校验。
- `rope_audit/commands.py`：编辑命令流解析和纯状态应用。
- `rope_audit/quant.py`：Q 格式量化、整数 sin/cos Taylor/象限归约、整数稳定指数。
- `rope_audit/pipeline.py`：统一 RoPE 接口、定点旋转/点积/归一化、float64 顺序真值和误差指标。
- `rope_audit/analytics.py`：训练/外推分层统计与同外推位置的方案交叉对照。
- `rope_audit/engine.py`：内容指纹、缓存键、复用/失效账本、预算调度和串行/线程一致执行。
- `rope_audit/snapshot.py`：临时目录、版本目录、指针原子替换和崩溃恢复判定。
- `rope_audit/injections.py`：独立故障注入配置、stderr 命中报告和错误码翻译。
- `tests/test_rope_audit.py`：校验、量化、缓存、预算、一致性、注入和快照测试。

## 关键语义

### 定点内核

- 量化尺度为 `Q = 2^(frac_bits - level_shift[quant_level])`，量化步长为 `1/Q`。
- 每个角度逐维记录原始值、量化步长、整数结果、反量化值和残差。
- 切换 `frac_bits` 或 `quant_level` 会改变有效位宽、尺度、整数和残差；不是显示层变化。
- sin/cos 使用整数象限归约和整数 Taylor 递推；旋转、点积累加、max-shift、exp 与概率归一化均保存整数中间值。
- 点积按维度序号升序累加，键按 `key_positions` 升序处理；不使用 `numpy.sum`、数组默认 reduction 或全局随机状态。

### 区间划分

当前实现面向因果相对距离：

```text
relative_distance = query_position - key_position
extrapolation iff relative_distance > sample.train_length
```

因此 `relative_distance <= train_length` 计入 training 层，严格超出才计入 extrapolation 层。统计同时输出样本数、键点数、打分均值、相对 float64 真值的最大偏差和相对均值的最大离散。

### 误差归因

四个环节固定为 `position`、`rotation`、`dot`、`normalization`。每样本输出最大绝对误差和最大相对误差；当真值元素精确为零时，该元素改用绝对误差作为相对分量，并设置：

```text
relative_error_branch = truth_zero_uses_absolute
zero_truth_branch = true
```

容差当前作用于最大绝对误差；首个 `abs_error > tolerance` 的环节写入 `first_breach`，包含环节、数值、容差和最差元素位置。外推键还会逐键生成各方案的环节误差、相对误差、固定路径分值/概率和相对基线方案的差值。

### 指纹与复用

缓存键由规范 JSON 的 SHA-256 摘要派生：

- `position`：方案参数指纹 + 位置子向量指纹 + 量化指纹。
- `rotation`：位置键 + 特征向量子指纹。
- `dot`：旋转键。
- `normalization`：点积键。
- `summary`：方案、训练长度、样本训练标记和当前 normalization 键集合。

因此只替换特征向量时，位置编码可命中、旋转及下游失效；只改训练长度时，底层数值产物全部命中、汇总失效；修改一个方案参数只失效该方案引用的单元。切回历史参数/位宽时，如果内容键仍在本地缓存，会记录 `historical_content_key_reused` 命中并重新挂载当前元数据。

### 预算调度

样本成本固定为：

```text
cost_points = dimension * len(key_positions)
```

调度器按样本 ID 字典序和待办优先构造稳定队列：

- 成本超过 `sample_quota_points`：记录 `E_POINT_BUDGET_EXCEEDED`，该样本失败但不中断其他样本。
- 成本超过本批剩余预算：该样本及队列尾部保留到 `pending_sample_ids`，并记录 `E_POINT_BUDGET_PENDING`。
- 下一批先处理 pending，再处理新失效样本。
- 预算只决定哪些稳定样本进入本批；所有批次排空后的缓存键、打分、统计、归因和对照与大预算串行结果一致。

### 快照与恢复

每次提交先创建同文件系统下的 `artifacts/.../tmp-stage-*`，写入各阶段 JSONL、`state.json`、`ledger.jsonl`、`budget.json`、`report.json` 和 manifest；随后写 `COMMITTED`、rename 为不可变版本目录，并原子替换 `current.json`。manifest 同时保存递增 `global_version` 与 `config_hash`。

恢复时：

- 未 rename 的 `tmp-stage-*` 判定为崩溃残留并清理。
- `current.json` 丢失时回退到最高已提交版本。
- 当前 `config_hash` 与配置不一致时返回 `config_hash_mismatch_cold_rebuild`，不加载旧状态。

### 注入

注入文件独立于主配置，可指定 `id`、`sample_id`、可选 `scheme_id`、`stage` 和 `fault_type`。命中时向 stderr 写：

```text
INJECTION_HIT id=... sample=... scheme=... stage=... code=...
```

故障会真实污染输入并由形状/有限值/归一化内核检查抛出，再翻译为注入配置中的业务码。单个 `(scheme, sample)` 单元进入失败清单，其他单元继续执行。

## 验证结论

最新实际运行结果见 `docs/verification-report.md`。其中记录了 10 个测试通过、十步命令流的 hit/miss 账本、增量/空白全量一致性、串行/线程一致性、预算 pending/排空、配额错误、注入 stderr 和快照恢复。
