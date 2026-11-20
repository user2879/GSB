# GPU 配额与分片调度领域组件

面向 Windows 本地环境的确定性 GPU 资源池配额与任务分片调度组件。纯 Python 实现，仅依赖 NumPy 与标准库 sqlite3，无需真实 GPU、容器或外部服务。组件以可集成包 `gpu_scheduler/` 与完整示例（样本、测试、演示、文档）交付。

## 能力概览

- 资源池/任务/分片 JSON 声明的强校验，非法输入整体拒绝并报字段路径（`E_SPEC`，细分 `E_MEMORY/E_QUOTA/E_CYCLE/E_DEPEND`）。
- 全局 GPU 配额、池并发上限、任务显存预算三重约束下的确定性并行分片调度。
- 基于优先级与等待时长的稳定排队，以及优先级严格不降序、可恢复的确定性抢占。
- 任务内与任务间依赖图的稳定拓扑失效传播，成功分片版本与产出摘要保持不变。
- 整数定点统计（比例向零截断、均值向下取整）与显式单值/比较/聚合误差界、稳定决胜标注。
- SQLite 账本单事务轮次写入、链式摘要、版本化会话快照、版本漂移整体替换与只读备份。
- 强制终止后仅依据已提交轮次恢复，在途分片标 UNKNOWN 重跑，并逐分片重放比对（`E_REPLAY`）。
- 四类异常注入、JSONL 结构化日志与 Tkinter 诊断界面（无 Tkinter 时控制台降级）。

## 快速开始

```powershell
py -3 -m pip install -r requirements.txt
py -3 demo.py
py -3 -m unittest discover -s tests
```

分步命令与 Windows 细节见 `docs/WINDOWS_USAGE.md`，架构与误差界口径见 `docs/DESIGN.md`，实际验证记录见 `docs/VERIFICATION.md`。

## 目录结构

```
gpu_scheduler/   可集成领域组件（引擎、定点、账本、快照、重放、注入、UI、CLI）
samples/specs/   确定性样本声明（session/pools/tasks JSON）
tests/           31 个 unittest 用例
docs/            设计、Windows 使用、验证结果
demo.py          端到端演示
workdata/        运行期产物（账本、快照、备份、日志、注入副本），已 git 忽略
```

## 业务码

| 码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明强校验失败（整体拒绝） |
| `E_MEMORY` | 显存预算越界 |
| `E_QUOTA` | 配额份额/全局 GPU 约束违规 |
| `E_CYCLE` | 分片或任务依赖成环 |
| `E_DEPEND` | 依赖引用缺失或不可恢复依赖问题 |
| `E_LEDGER` | 账本格式/内容摘要不符 |
| `E_REPLAY` | 强制终止恢复后重放不一致 |
