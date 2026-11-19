# 确定性 PEFT 低秩适配组件

这是一个不依赖真实模型权重的本地 PEFT 示例与可集成组件。它使用固定种子的合成基座，完成低秩适配器训练、指纹兼容判定、合并/可逆还原、预算调度、基座升级增量重训、阶段化快照和强制终止恢复。

运行时只依赖 Python 3.11 与 NumPy。仓库内 `wheelhouse/` 已包含 CPython 3.11 的 Windows x64 和 Linux x64 NumPy 1.26.4 wheel，安装过程使用 `--no-index`，不需要在线索引。

## 快速开始（Windows）

先安装 CPython 3.11 x64，然后在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_offline.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_acceptance.ps1
```

也可以直接执行：

```bat
run_acceptance.cmd
```

`setup_offline.ps1` 会创建 `.venv-win`、从本地 wheel 安装 NumPy、验证导入，并生成确定性样本。测试覆盖训练一致性、升级复用、合并可逆性、预算中止、信号退出和强制终止恢复。

运行完整演示：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_demo.ps1
```

## CLI

```bash
python -m peft_lora run --samples samples --workspace work
python -m peft_lora run --samples samples --workspace work --base-dir samples/base_v2
python -m peft_lora inspect --workspace work
```

每次运行在 `workspace/report.json` 输出确定性 JSON 报告，stdout 再输出一份精简 JSON 摘要。业务失败返回退出码 `1`；正常完成或收到可处理退出信号并完成当前阶段后返回 `0`；强制终止不承诺生成退出报告。

## 样本目录

```text
samples/
  base_v1/                    冻结基座 v1
  base_v2/                    仅 layer_1 改动的基座 v2
  tasks.json                  任务清单
  instructions.jsonl          指令样本
  peft_config.yaml            PEFT 配置
```

### 基座二进制布局

`manifest.json` 描述每个参数文件：

```json
{
  "format": "peft-base-bin-v1",
  "fingerprint_algorithm": "base-fp-v1",
  "layers": [{
    "name": "layer_0",
    "weight": {"file": "layer_0.weight.bin", "dtype": "<f8", "shape": [4, 3]},
    "bias": {"file": "layer_0.bias.bin", "dtype": "<f8", "shape": [4]}
  }]
}
```

权重和偏置都是 C order、little-endian float64 原始字节。加载时校验层名唯一、文件大小与 `shape × dtype.itemsize` 一致、形状维度、偏置长度、有限数以及路径不越出基座目录。

### 配置与任务

`peft_config.yaml` 包含目标层白名单、秩/缩放候选、固定学习率、训练轮数、批量、定点量化档位、线程池并发数、两类合并预算和快照提交间隔。任务中的秩与缩放必须为正且属于候选集合；目标层必须在白名单和基座中。

统一业务码：

| 业务码 | 含义 |
| --- | --- |
| `E_BASE_INVALID` | 基座包结构、文件、形状或数值非法 |
| `E_TASK_INVALID` | 任务/样本结构或目标层非法 |
| `E_ADAPTER_CONFIG` | 适配秩、缩放、配置或适配器产物非法 |
| `E_BASE_VERSION_MISMATCH` | 适配器目标层指纹与当前基座不一致，必须重训 |
| `E_MERGE_NOT_REVERSIBLE` | 量化后无法在允许误差内还原基座 |
| `E_MERGE_BUDGET_EXHAUSTED` | 合并预算不足，任务仍留在稳定顺序的待处理队列 |

任务/样本错误进入失败清单后，其余合法任务仍继续加载和执行。

## 关键保证

- 训练初始化只依赖任务稳定键、秩和目标层形状；梯度目标与样本顺序固定，每个任务独立使用确定性 RNG。
- 适配器同时记录整包基座指纹和目标层指纹。整包变化但目标层字节未变时，状态为 `COMPATIBLE_LAYER_UNCHANGED`，可直接复用。
- 合并按任务名字典序进行；权重随后按配置的定点档位量化。还原时从合并权重逐项减去适配器 `scale × B × A`。
- 合并预算按稳定任务顺序分配，允许/拒绝与扣减前后值逐条写入审计账本。被拒绝及后续任务不会写入该次合并负载。
- 阶段目录先完整写入临时目录，再重命名并以 `CURRENT.json` 原子切换。清单包含全局版本、配置哈希、基座指纹、兼容清单哈希及四个阶段版本。
- `SIGINT`、`SIGTERM` 和 Windows `SIGBREAK` 会设置优雅退出标记；当前阶段完成后提交，再以 `0` 退出。
- 强制终止后的重启只信任 `CURRENT.json` 中引用的完整阶段；重启报告包含 `recovery.started_from_version`。

## 测试与文档

```bash
python scripts/run_acceptance.py
```

- 架构、指纹、训练与快照协议：`docs/DESIGN.md`
- 实际运行结果与宿主限制：`docs/VERIFICATION.md`
- 机器可读验收摘要：`artifacts/acceptance_summary.json`
