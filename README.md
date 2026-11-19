# 确定性图像修复与风格迁移 Studio

本工作区实现了一个纯本地、无 GPU、无容器依赖的 Python 3.11 图像修复与风格迁移应用。应用包含程序化样本生成、PNG/配置校验、定点确定性填充、量化 LUT 风格映射、内容寻址增量缓存、受预算约束的交互调度、分块并发渲染、原子快照与强制终止恢复。

## 快速开始

```bash
# 当前工作区已包含 .deps；在标准 Windows Python 3.11 中先安装依赖
python -m pip install -r requirements.txt

# 生成两个固定分辨率的确定性样本
python scripts/generate_samples.py --root samples

# 确定性全量验证模式：每条命令等待渲染完成，供逐像素对照
python -m repair_studio run --samples samples --output output/sync --mode sync

# 交互模式：每条命令在配置预算内返回；必要时返回 E_RENDER_BUDGET_PENDING
python -m repair_studio run --samples samples --output output/interactive --mode interactive

# 仅依据完整原子快照恢复，并输出恢复点版本号
python -m repair_studio recover --output output/sync

# 自动化测试与完整验收
python -m unittest discover -s tests -v
python scripts/verify_acceptance.py
```

入口也可写作 `python run.py ...` 或 `python -m repair_studio ...`。`repair_studio/__init__.py` 会在存在项目内 `.deps` 时加入本地依赖路径；标准 Windows 环境安装 `requirements.txt` 后不需要该回退。

## 样本格式

样本根目录下每个子目录是一份样本；也可直接把单样本目录作为 `--samples`。默认文件为 `image.png`、`style.png`、`config.json`、`commands.json`，路径可在配置中覆盖。

```json
{
  "schema_version": 1,
  "seed": 20261120,
  "width": 192,
  "height": 128,
  "image_file": "image.png",
  "style_file": "style.png",
  "commands_file": "commands.json",
  "mask": {"x": 48, "y": 32, "width": 64, "height": 48},
  "repair": {"radius": 3, "iterations": 40},
  "style": {
    "quant_levels": 16,
    "channel_weights": [0.35, 0.75, 0.55],
    "preserve_mask": true
  },
  "interaction": {
    "latency_budget_ms": 30,
    "chunk_rows": 8,
    "thread_workers": 4,
    "snapshot_interval": 2
  },
  "fixed_bits": 16
}
```

合法范围与约束：

- `mask` 必须完全位于图像内且宽高为正，否则为 `E_MASK_OUT_OF_RANGE`。
- `fixed_bits` 为 4 至 20；`quant_levels` 为 2 至 256 的 2 的幂。
- `repair.radius` 为 1 至 64，`repair.iterations` 为 1 至 4096。
- `chunk_rows` 为 1 至图像高度，`thread_workers` 为 1 至 128。
- 缺字段、类型错误、分辨率不一致、分块行越界等均为 `E_CONFIG_INVALID`。
- PNG 容器头、PIL 解码或容器格式错误均为 `E_IMAGE_DECODE`。

样本加载失败会进入按 `(sample_id, code, message)` 排序的失败清单，并继续处理其他样本。

## 交互命令

`commands.json` 可为数组，也可为 `{"commands": [...]}`：

```json
[
  {"op": "set_repair_radius", "value": 4},
  {"op": "set_style_weights", "weights": [0.2, 0.7, 0.5]},
  {"op": "set_quant_levels", "value": 8},
  {"op": "set_mask_rect", "rect": {"x": 44, "y": 28, "width": 72, "height": 54}},
  {"op": "set_preserve_mask", "value": false},
  {"op": "snapshot"},
  {"op": "exit"}
]
```

支持的操作：`set_mask_rect`/`set_mask`、`set_repair_radius`/`set_radius`、`set_repair_iterations`、`set_style_weights`、`set_quant_levels`、`set_preserve_mask`、`snapshot`、`exit`。无效命令返回 `E_CONFIG_INVALID` 或 `E_MASK_OUT_OF_RANGE`，保持上一有效参数并继续后续命令。

## 渲染流水线

四个阶段均持久化到 `output/cache/v1/<stage>/<cache_key>/`，并先写临时目录、校验文件哈希后整体原子替换。

1. **mask**：由源图内容指纹与矩形参数生成布尔掩码。
2. **fill**：只从当前已知区域向未知掩码推进；每层用半径内已知像素的距离平方反比权重。权重为 `max(1, 2^20 / d²)`，通道值与分子、分母均为整数，均值使用 half-up 整数除法；元数据记录 `fixed_bits`、`rounding=half_up`、权重规则、已填与未填像素数。
3. **style**：由风格参考图的整数均值/标准差、三个 Q16 通道权重、量化档位构建 LUT。输入像素先量化到 LUT 索引，再整体查表；量化档位改变会改变 LUT 形状、条目哈希和实际输出像素。
4. **composite**：保留策略开启时在掩码区恢复源图像素；关闭时让风格结果覆盖掩码区，并记录 `style_override_pixels`。

缓存键由上游缓存键、图像内容指纹、参数指纹、量化档位及算法版本派生。命令只会使受影响阶段及其下游失效。例如修改风格权重复用 `mask`/`fill`，修改修复半径复用 `mask`，修改保留策略只重算 `composite`。

线程池并发按配置的 `thread_workers` 创建并复用。每个分块任务携带稳定键 `image.png|00000003|fill` 形式；合并时按稳定键排序，失败任务也按键排序。线程调度顺序、完成顺序和实际并发不影响合成顺序、像素或阶段摘要。

## 延迟预算语义

`--mode interactive` 启动后台串行渲染工作线程。命令触达后主线程最多等待预算减去安全余量；得到该请求的完成结果时返回：

- `business_code: "OK"`、`result_is_latest: true`；
- 最新参数对应的预览、四阶段摘要、复用/失效清单与完整进度。

若预算内未完成，则返回：

- `business_code: "E_RENDER_BUDGET_PENDING"`、`result_is_latest: false`；
- `latest_completed`：最近一次完成的真实旧预览，带 `params_match_requested`，不会冒充新参数结果；
- `progress`：当前任务阶段、已完成行数、总行数、进度与 `params_match_requested`；
- 受影响区域和最近完成预览的实际复用/失效审计。

交互命令流结束后会等待当前最终参数完成、写最终 PNG、提交最终快照，再正常退出。

`--mode sync` 是供验收和自动对照使用的全量模式：每条命令等待完整结果，允许耗时超过预算，因此其 `budget_met` 可能为 `false`，但不使用 pending 语义。延迟验收应使用 `interactive` 模式。

## 快照与退出恢复

每个样本的目录为 `output/snapshots/<sample_id>/`。一次提交包含：

1. 在 `.tmp-snapshot-*` 中写入全部阶段产物副本、`final.png`、`manifest.json` 并完成哈希与 fsync；
2. 将临时目录整体重命名为 `gen-XXXXXXXX`；
3. 以临时文件 + `os.replace` 原子切换 `CURRENT` 指针。

`manifest.json` 记录 `global_version`、配置哈希、参数指纹、已完成命令游标、最终 PNG 哈希及每个阶段产物的缓存键/文件哈希。提交前会验证当前指针和清单；状态未变化时返回 `unchanged`，绝不会逐文件覆盖旧代。

- `SIGINT`/`SIGTERM` 被视为可处理退出：完成当前命令，排空最终状态，提交快照与诊断报告，然后以整数退出码 `0` 退出。
- 强制终止（如 `SIGKILL`）不承诺退出报告。之后 `recover` 仅接受 `CURRENT` 指向且清单/全部文件哈希完整的快照，忽略 `.tmp-snapshot-*` 和旧代未发布目录，并输出 `recovered_version`；同时把恢复 PNG 写入 `output/recovered/<sample_id>/final.png`。

## 诊断报告

每次正常运行写入 `output/reports/diagnostics-<run_id>.json`，包含：

- 每条命令的操作、`elapsed_ms`、`budget_ms`、`budget_met`、业务码；
- `reused_stages`、`invalidated_stages` 及计数；
- 四阶段摘要、最终 PNG 哈希、分块任务键和实际观测并发；
- 加载/命令/渲染失败清单及排序结果；
- 运行级业务码计数、样本状态、完成游标和快照提交记录。

恢复报告写入 `output/recovered/recovery-report.json`，并明确 `exit_report_written: false`。

## 审计一致性的边界

运行时的 `reused_stages`/`invalidated_stages` 是命令历史相关的执行事实；仅凭最终参数不可能反推出中间复用过哪些阶段。因此验收对“从空白工作区按最终参数一次性全量渲染”的对照采用：

- 最终 PNG 像素逐像素一致；
- 四阶段 `cache_key`、内容哈希、量化档位、LUT、任务键等阶段摘要逐字段一致；
- `artifact_versions` 与快照阶段版本逐字段一致；
- 增量运行的逐命令复用/失效清单单独审计，并验证恰好等于对应命令的“受影响阶段 + 下游”集合。

这一区分避免了把历史审计列表伪装成最终参数的纯函数，同时保留了可逐项核对的全部确定性证据。`scripts/verify_acceptance.py` 会输出所有比较结果。

## 实际验证结果

最近一次完整验收见 `artifacts/verification.json`，结果为 `PASS`，耗时约 48 秒：

- 两个样本的最终增量 PNG 与全新全量渲染逐像素一致：`procedural_a` 为 `7f5263b2f57c4b279ce2f017be34e156abde3b471c6994ccd8c0e8e6ad86e174`，`procedural_b` 为 `6fe20cd267418e11ba5f3b191fc71fca350b75a67705796fe360773875e5e4b6`。
- 两个样本的四阶段摘要与 `artifact_versions` 逐字段一致。
- `procedural_a` 的 10 条命令逐条与对应状态的全量渲染像素、摘要一致；复用/失效集合符合下游规则。
- 1 线程与 4 线程最终像素、阶段摘要一致；观测最大并发为 4，任务键按行号稳定排序。
- 30ms 预算下 10 条交互命令全部返回 `E_RENDER_BUDGET_PENDING`，最大响应耗时 28.418ms，全部带真实旧预览与进度；排空后的最终 PNG 与全量参考一致。
- 正常快照恢复成功，恢复点版本为 6；强制终止测试确认不生成退出报告，恢复只使用完整快照。
- `python -m unittest discover -s tests -v`：14 项测试全部通过。

## 代码结构

- `repair_studio/config.py`：配置/命令解析、严格类型与范围校验、规范化指纹。
- `repair_studio/imaging.py`：PNG 头与解码校验、内容哈希、风格统计。
- `repair_studio/rendering.py`：定点填充、量化 LUT、分块并发、内容寻址阶段产物。
- `repair_studio/cache.py`：原子缓存读写与文件哈希验证。
- `repair_studio/engine.py`：同步/交互调度、预算响应、逐命令诊断。
- `repair_studio/snapshot.py`：整代快照、原子 CURRENT、恢复校验。
- `repair_studio/samplegen.py`、`scripts/generate_samples.py`：确定性样本生成。
- `scripts/verify_acceptance.py`：端到端验收与证据生成。
- `tests/`：错误码、效果敏感性、逐命令等价、并发、预算、命令错误、快照/信号/强杀恢复测试。
