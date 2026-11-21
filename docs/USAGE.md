# 使用说明

## 环境准备（Windows / Linux / macOS 通用）

1. 安装 Python 3.10+，确认 `python --version` 与 `pip --version` 可用。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

依赖只有 `numpy` 与 `PyYAML`，不需要 GPU、模型权重或网络。

## 生成本地确定性样本

```bash
python tools/generate_samples.py --out samples
```

生成 `samples/render_config.yaml`、`samples/prompts.jsonl`、
`samples/edit_plan.json` 以及 `samples/images/` 下的初始 PNG。样本覆盖单批、
多批、两档分辨率、提示词替换、种子/引导/步数修改、档位增删、初始图替换、
补丁增删（含 no-op 与越界失败批次）。

## 运行渲染

```bash
# 增量渲染：按编辑计划逐批渲染，仅重算失效阶段
python run_workbench.py render \
  --config samples/render_config.yaml \
  --prompts samples/prompts.jsonl \
  --plan samples/edit_plan.json \
  --workdir work/inc

# 空白全量渲染：清空工作目录后按最终输入一次性渲染
python run_workbench.py full \
  --config samples/render_config.yaml \
  --prompts samples/prompts.jsonl \
  --plan samples/edit_plan.json \
  --workdir work/full

# 强制单线程串行基准
python run_workbench.py baseline \
  --config samples/render_config.yaml \
  --prompts samples/prompts.jsonl \
  --plan samples/edit_plan.json \
  --workdir work/baseline

# 查看工作目录清单摘要（版本号、配置哈希、节点数、游标）
python run_workbench.py inspect --workdir work/inc
```

## 运行测试与端到端验证

```bash
python -m unittest discover -s tests
python tools/validate_workbench.py
```

验证结果写入 `work/validation/validation_summary.json`，每次运行的诊断报告
位于各工作目录的 `report.json`，持久化清单位于 `manifest.json`。

## 工作目录产物

```
<workdir>/
  manifest.json     全局版本、配置哈希、节点缓存键/内容指纹、步序帧序游标
  report.json       每批失效/复用清单、累计重算数、失败码、恢复点版本
  store/<node>/     各节点 latent.bin / embedding.bin / frame.png / pixels.bin
  tmp/              原子提交暂存区（启动时清理）
```
