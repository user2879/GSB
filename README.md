# mdet-render：确定性定点多模态条件渲染包

`mdet-render` 是一个只依赖 Python 标准库与 numpy 的本地 Python 包。它把受控词表文本、RGB PNG 图像和分块运动矢量表编码为定点条件，经过融合、潜变量初始化、逐去噪步生成、逐帧解码与分块行 PNG 输出，形成可重复的多模态渲染管线。

实现重点不是“能出图”，而是以下可复核约束：

- 所有数值路径落在显式定点整数域，随机源由内容指纹派生，不读取 numpy 全局随机状态。
- v1 定宽行与 v2 JSON 可双向转换；v2 → v1 的舍入残差、精度标签、未知字段进入旁路 sidecar，重新升级可精确还原。
- 条件、融合、潜变量、每个去噪步、解码计划和每个分块行都有内容寻址缓存键及上游引用。
- 修改单帧提示词/图像、融合权重、去噪步数、分块行数、量化档位或租约参数时，只重算依赖真正变化的产物。
- 本地执行器租约包含到期毫秒和单调递增代次；过期检查会立即释放工作单元并产生 `E_LEASE_EXPIRED`，不使用 sleep 延迟回收。
- 低预算调度按稳定顺序在失效去噪步间分配点数；旧待办优先，未处理步次保留为 `E_POINT_BUDGET_PENDING`。
- 增量重放与从空白全量渲染在像素、PNG 字节、分块摘要、缓存键与调度顺序上一致。

## 环境与安装

要求 Linux / WSL Ubuntu、Python 3.11+、numpy。运行测试另需 pytest。项目不依赖容器、网络或外部服务。

```bash
python3 --version
python3 -m pip install -e '.[test]'

# 旧版 pip 若未完成 PEP 517 构建环境，可使用：
# python3 -m pip install -e . --no-deps --no-build-isolation
# python3 -m pytest -q
```

如果系统 Python 缺少 Tkinter，可安装 `python3-tk`；本包核心渲染与测试不需要 GUI。

## 快速开始

```bash
# 1. 生成确定性样本、编辑命令流与租约参数
python3 -m mdet_render init-samples --root samples

# 2. 严格校验配置、词表、PNG、运动表，并试跑整条编辑命令流
python3 -m mdet_render validate \
  --config samples/config.json \
  --commands samples/commands.json

# 3. 首次增量渲染 + 8 条编辑逐条重放
python3 -m mdet_render render \
  --config samples/config.json \
  --commands samples/commands.json \
  --replay-edits \
  --cache-dir samples/render_cache \
  --ledger samples/ledger/render.jsonl \
  --out samples/out

# 4. 完整演示：增量/全量逐字节比较、粗粒度失效审计、租约回收
python3 -m mdet_render demo --root samples
```

样本目录还包含 `intermediate_v2.json`、`intermediate.v1` 与对应 sidecar，可直接验证双版本往返：

```bash
python3 -m mdet_render v2-to-v1 \
  --input samples/intermediate_v2.json \
  --output /tmp/intermediate.v1 \
  --sidecar /tmp/intermediate.v1.sidecar.json

python3 -m mdet_render v1-to-v2 \
  --input /tmp/intermediate.v1 \
  --output /tmp/intermediate.restored.json \
  --sidecar /tmp/intermediate.v1.sidecar.json
```

`demo` 生成的证据位于：

- `samples/demo/demo_report.json`：比较明细、各编辑轮账本摘要、租约回收事件。
- `samples/demo/ledger/incremental.jsonl`：每次渲染的复用/失效与调度批次。
- `samples/demo/incremental_output/` 与 `samples/demo/full_output/`：逐帧和逐分块 PNG。
- `samples/demo/results.jsonl`：逐编辑结果序列。

## CLI

```text
mdet-render init-samples  生成样本工作区
mdet-render validate      校验配置和资源；失败返回非零业务码
mdet-render render        全量或增量重放渲染
mdet-render demo          端到端一致性与租约演示
mdet-render lease-demo    单独验证租约代次、过期与真实回收
mdet-render v2-to-v1      v2 JSON -> v1 定宽行 + sidecar
mdet-render v1-to-v2      v1 定宽行 -> v2 JSON（可提供 sidecar）
```

业务错误统一输出 `{"ok": false, "code": "E_...", ...}` 并返回非零退出码。点预算未完成返回 `E_POINT_BUDGET_PENDING`。

## Python API

```python
from mdet_render import Renderer, Workspace, load_commands

workspace = Workspace.load("samples/config.json")
renderer = Renderer(
    workspace,
    cache_dir="samples/runtime_cache",
    ledger_path="samples/ledger/runtime.jsonl",
)
try:
    first = renderer.run(output_dir="samples/out")
    for command in load_commands("samples/commands.json"):
        workspace = workspace.apply_command(command)  # 事务式；失败不修改原状态
        renderer.workspace = workspace
        result = renderer.run(output_dir="samples/out")
finally:
    renderer.close()  # 释放租约工作单元
```

命令操作包括 `replace_prompt`、`replace_image`、`delete_image`、`set_weight`、`set_denoise_steps`、`set_decode_block_rows`、`set_quantization`、`set_lease`，以及测试用的 `set_point_budget`、`set_max_workers`、`set_guidance_strength`。

## 项目结构

```text
mdet_render/
  config.py       配置、枚举、维度与分块相容性校验
  conditions.py   词表/提示词/PNG/运动表加载与定点编码
  png.py          严格 PNG 解码（CRC、IHDR、过滤器）与确定性编码
  fixed.py        定点缩放、量化、舍入与存储范围
  formats.py      v1 定宽行 / v2 JSON / 旁路 sidecar
  lease.py        本地执行器、代次、TTL、真实回收
  cache.py        内容寻址产物库、manifest 与 JSONL 账本
  renderer.py     管线、并发调度、预算、失效判定与 PNG 输出
  workspace.py    可编辑状态与事务式命令
  samples.py      确定性样本和编辑流生成器
  demo.py         增量/全量/租约端到端演示
tests/            39 项自动化测试
samples/          已生成的样本、命令、账本与演示证据
TECHNICAL.md      格式、缓存键、调度和确定性不变量说明
```

详细技术约定见 [TECHNICAL.md](TECHNICAL.md)。
