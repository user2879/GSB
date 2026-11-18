# LoRA 多任务增量训练与诊断工作台

面向本地确定性样本的 LLM 多任务低秩适配（LoRA）增量训练工作台，纯 NumPy +
标准库、CPU 离线运行，不下载任何真实模型权重。核心能力是共享冻结基座下的
**任务级 / 基座级双层依赖指纹与精确失效判定**，并保证编辑序列下增量训练与
全量重建逐字段一致。

## 环境准备

- Python 3.11+（Windows 原生或 WSL 均可）
- 安装依赖：`python -m pip install -r requirements.txt`（numpy、pytest）

## 一分钟上手

```bash
# 1. 生成确定性样本、任务清单与两版冻结基座
python -m lora_bench.cli generate data/input
# 2. 增量训练（再次执行即验证复用，不会重训未失效任务）
python -m lora_bench.cli run data/input workspace/ws --config configs/base_config.json
# 3. 增量 vs 从空白工作区全量重建逐字段对账
python -m lora_bench.cli rebuild data/input workspace/ws workspace/rebuild --config configs/base_config.json
# 4. 异常注入诊断（报告原子写盘）
python -m lora_bench.cli inject workspace/injection
# 5. 完整端到端验收演示
python scripts/run_demo.py
# 6. 测试套件（33 用例）
python -m pytest tests/ -q
```

## 输入产物

`generate` 在输入目录下生成：

- `samples.jsonl`：每行含 `sample_id`、`task`、`input_tokens`、`target_tokens`、`label`
- `tasks.json`：任务名、类别、指标阈值、训练/验证样本 ID 列表
- `base_v1/`、`base_v2/`：逐层权重/偏置 float32 二进制与 `shapes.json`（含基座指纹）

配置变体位于 `configs/`，覆盖不同秩、目标层白名单、并发数与快照提交间隔，
并支持 `task_overrides` 为单个任务设置秩与缩放系数。

## 核心机制

- **双层指纹**：任务身份 = 样本内容指纹 + 秩/缩放 + 目标层 + 基座指纹 + 训练超参；
  失效判定为纯函数，增量与全量重建共用同一逻辑（见 `lora_bench/deps.py`）。
- **失效粒度**：样本/秩/缩放变化仅失效单任务；目标层或基座变化 fan-out 全部任务。
- **阶段化快照**：前向/训练/验证先写暂存目录再原子整体替换；按 `commit_interval`
  冻结中间版本；严格续跑不一致返回 `E_CONFIG_MISMATCH`（见 `lora_bench/snapshot.py`）。
- **并发一致**：线程池按配置并发，所有输出按 `任务名|层序` 稳定键排序，与单线程一致。
- **故障隔离**：单任务异常记录任务名、阶段、异常类型后继续其余任务。

## 文档

- 设计细节：`docs/DESIGN.md`
- 验收记录：`docs/VERIFICATION.md`
- 错误码：`lora_bench/errors.py`（`E_SAMPLE_INVALID`、`E_BASE_SHAPE_MISMATCH`、
  `E_CONFIG_MISMATCH` 等）
