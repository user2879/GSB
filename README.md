# vrep：本地确定性视觉表征与检索增量编辑应用

面向本地程序生成 PNG 图像库的分块特征抽取、定点量化、图像嵌入、相似度检索，
重点解决图像与表征配置反复编辑时的**增量更新、精确失效判定与预算调度**问题。
仅依赖 Python 标准库、NumPy、Pillow，不使用 GPU、网络或容器。

## 快速开始

```bash
python -m pip install -r requirements.txt

# 生成确定性样本（samples/）
python -m vrep.cli generate-samples --samples samples

# 全量重建 vs 增量执行
python -m vrep.cli run --samples samples --workspace workspace/full \
    --config samples/config.json --commands samples/commands.json --mode full
python -m vrep.cli run --samples samples --workspace workspace/incr \
    --config samples/config.json --commands samples/commands.json --mode incremental

# 逐字段一致性比对
python tools/compare_ledgers.py \
    workspace/full/store/state/ledger.json \
    workspace/incr/store/state/ledger.json

# 一键验收（一致性 + 预算待办 + 注入业务码）
python tools/run_acceptance.py

# 全量测试（49 个用例）
python run_tests.py
```

Windows PowerShell / CMD 中将行尾 `\` 换成 `^`（CMD）或写在同一行即可。

## 目录

- `vrep/`：应用包（加载校验、特征/量化、嵌入/索引、预算、存储、快照、注入、引擎、CLI）
- `tools/`：一键验收与账本比对
- `tests/`：按 8 项业务要求编号的 unittest（`test_01` ~ `test_10`）
- `docs/TECHNICAL_DESIGN.md`：架构、设计决策与要求映射
- `docs/TASK_SPEC.md`：原始任务书
- `samples/`：由生成器产出的图像、真值、配置、命令流、注入配置
- `workspace/`：运行期工作区（已 gitignore）

## 核心保证

- **确定性**：投影/嵌入矩阵由配置指纹派生独立 RNG，不依赖全局随机状态；
  半向上定点量化，同图同配置逐元素一致，并发度变化不影响结果。
- **精确失效**：特征/嵌入/索引三级缓存键绑定内容指纹、配置子集指纹与量化档位，
  复用与重算逐条入账本。
- **逐字段一致**：任意编辑序列后的增量结果与空白全量重建在嵌入向量、缓存键、
  检索顺序、边界样本、失败清单、待办队列上完全一致。
- **可审计预算**：单批特征点预算稳定预订，超预算分块持久化为部分特征束并标记
  `E_POINT_BUDGET_PENDING`，待办图像不进索引。
- **崩溃安全**：阶段产物临时目录 + 整体替换 + 快照指针，强制终止后仅恢复完整快照；
  信号退出完成当前提交后返回退出码 0。
