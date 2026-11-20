# 指令微调数据构造诊断工作台（sft-diag）

面向本地确定性样本的指令微调数据构造与增量失效诊断工作台。支持确定性加权
采样、四维度质量打分、子词 Jaccard 近重复剔除、基于内容指纹的精确失效判定、
阶段化原子快照与并发一致性，保证分批增量运行与全量重建逐字段一致。

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install numpy jsonschema && .venv/bin/pip install -e .
.venv/bin/python scripts/generate_samples.py
.venv/bin/python scripts/acceptance_check.py
```

## 目录结构

```
src/sft_diag/        工作台源码（加载、采样、打分、去重、快照、编排、CLI）
scripts/
  generate_samples.py  固定种子样本生成程序（无随机源）
  acceptance_check.py  一键验收脚本
samples/basic/       本地确定性样本目录（六类输入）
tests/               27 个 unittest 测试
docs/                架构、需求追踪、验证结果、环境搭建说明
```

## 核心特性

- **确定性**：numpy PCG64 + 内容指纹派生种子，重复运行产物逐字节一致；
- **精确失效**：采样/打分/去重三阶段缓存键 + 模板/槽位引用索引，未失效
  候选零重算；
- **逐字段一致**：内置影子全量构造对照，任何差异写入诊断报告并标注来源；
- **快照恢复**：临时目录 + `os.replace` 原子提交，强制终止后仅依据完整
  快照恢复，基础输入变化返回 `E_CONFIG_MISMATCH`；
- **并发一致**：线程池并发度受配置限制，输出顺序与单线程按 ID 执行完全相同。

详见 `docs/ARCHITECTURE.md`、`docs/REQUIREMENTS.md`、`docs/VERIFICATION.md`。
