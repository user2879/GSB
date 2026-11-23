# 跨模态图文检索诊断工作台（retbench）

面向本地确定性样本的跨模态图文检索诊断工作台：固定种子生成 PNG 几何图元样本与
受控词表文本断言，完成确定性特征抽取、定点量化、跨模态对齐、三通道召回融合、
规模自适应策略选择、阶段级精确失效账本与快照化诊断报告。

详细设计见 `docs/tech-note.md`。

## 快速开始

```bash
# 离线创建虚拟环境并安装 NumPy/Pillow（wheel 位于 vendor/wheels）
bash scripts/bootstrap.sh

# 运行 31 个单元测试
.venv/bin/python -m unittest discover -s tests -v

# 端到端验收：生成样本、全量/增量等价、异常与并发报告
.venv/bin/python -m retbench.acceptance sample_accept --samples 12

# CLI：在工作区上输出 reports/full_report.json、incremental_report.json、
# command_history.json、snapshot_timeline.json、equivalence.json
.venv/bin/python -m retbench.runner sample_accept
```

## 输入与产物

- `config.yaml`：分辨率、受控词表、分块/特征维度、对齐权重、通道开关与候选上限、
  定点位宽与量化档位、规模档位、延迟预算、线程池、快照提交间隔与预置策略表。
- `commands.json`：规模切换、权重调整、量化切换、单图/文本替换、通道增删命令流。
- `injections.json`：固定故障注入清单（区域越界、词表外实体），用于失败清单验收。
- `samples/`：`*.png`、`*.regions.json`、`*.text.json` 三元组。
- `reports/`：逐查询归因、边界样本、量化审计、决策记录、复用/重算清单、失败明细。

## 错误码

- `E_SAMPLE_INVALID`：样本校验失败，进入失败清单且不阻断其他样本。
- `E_CONFIG_INVALID`：配置/命令流结构非法，致命。
- `E_INTERACT_BUDGET_EXCEEDED`：交互预算内无法完成，返回最近过期摘要。
- `E_SNAPSHOT_STALE`：快照配置哈希漂移，清理后转入全量重建。

## 约束

仅使用 Python 标准库、NumPy 与 Pillow；不使用 Docker；不访问网络或未授权目标。
YAML 解析在无 PyYAML 时自动回退到内置零依赖迷你解析器（与 PyYAML 输出已对齐验证）。
