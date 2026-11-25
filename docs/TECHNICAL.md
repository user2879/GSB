# FixedTex 技术说明

## 1. 目标与边界

FixedTex 在 Linux/WSL 上实现程序化纹理与材质的分层定点渲染。核心渲染路径只使用 NumPy `int64`/`int8` 整数数组；浮点数仅出现在 JSON 参数校验、配置编译和预览/诊断报告展示中。GUI 不是渲染依赖，Tkinter 仅作为环境预检项。

工程布局：

```text
fixedtex/               渲染核心与 CLI
  config.py             配置、材质、命令、注入加载校验
  texture.py            PNG 容器与像素模式校验、扫描、解码
  fixed.py              定点舍入、量化、缩放与数组摘要
  noise.py              SplitMix64 坐标哈希 + 双线性 value noise
  material.py           四通道映射、逐通道混合、最终 RGBA 合成
  renderer.py           增量批次、预算调度、失效判定、账本
  scheduler.py          确定性线程池结果归并
  snapshot.py           目录级原子快照、版本/哈希恢复
  session.py            命令会话、后台批次、预览、提交
  diagnostics.py        四阶段误差传播与两档量化归因
  verification.py       验收场景编排
samples/                固定种子 PNG、配置、材质、命令、故障注入
tests/                  28 项单元/集成/CLI/强杀测试
tools/generate_samples.py
artifacts/              运行时输出、快照、账本、验证报告（本地）
```

## 2. 输入格式与校验

渲染配置的规范字段见 `samples/config.json`：

- `resolution_tiers`：严格递增的正整数列表。
- `resolution`：必须属于 `resolution_tiers`。
- `noise.layers[]`：每层含正的有限 `frequency`、`amplitude` 与整数 `seed`。
- `fixed.fraction_bits`：允许 `8/12/16/20`。
- `fixed.quant_step`：允许 `1/2/4/8/16`。
- `fixed.tie_rule`：`ties_to_even/ties_to_lower/ties_to_upper`。
- `budgets.layer_points`：单层每个批次最多消费的像素点。
- `budgets.batch_points`：所有层每个批次共享的点预算。
- `preview_budget_ms`、`thread_pool_size`、`commit_interval_commands`。
- `error_tolerance_quant_steps`：误差阈值以量化步为单位，0 表示 `max(1, quant_step/4)`。

材质规范字段见 `samples/material.json`。四个通道 `diffuse/roughness/metallic/normal` 各自声明 `base_weight` 和若干 `{index, weight}`。权重在加载时用 ties-to-even 编译为千分整数 `weight_q`，渲染阶段只使用整数权重。材质 metadata 固定写入 `tie_rule` 和 `equal_weight_rule`。

校验层不会因单个坏项丢弃整个输入：

- 非法频率/振幅保留层位置并采用可继续处理的安全值，同时报告 `E_FREQ`/`E_AMPLITUDE`。
- 非法分辨率档位被跳过，其余合法档位继续做单调性检查。
- 非法位宽、量化档位、预算、线程数等使用安全默认值并逐项报告。
- 缺失材质通道会生成“仅基础权重”的占位通道，其他合法通道继续保留。
- 非法命令行/命令对象被跳过，命令流中其余合法命令继续执行。
- PNG 先校验签名、IHDR/IEND、chunk CRC、8-bit 像素模式、分辨率，再交给 Pillow 解码为 `uint8` RGB；支持 `RGB/RGBA/L`，拒绝 palette 与非法头。

CLI 使用整数退出码：`0` 成功，`2` 校验/业务输入错误，`3` 渲染/预算/注入失败，`4` 快照/恢复错误，`5` 内部异常。每个错误都带 `E_xxx` 业务码。

## 3. 定点噪声与量化

第 `i` 层指纹：

```text
SHA256(KEY_VERSION, "layer-fingerprint",
       texture_content_key, resolution, global_seed, algorithm,
       layer_index, frequency_milli, amplitude_milli, layer_seed)
```

指纹不包含小数位宽，因此主路径与“高一位宽参考路径”共享同一噪声图案。指纹前 64 bit 作为 SplitMix64 种子。每个像素的 lattice 四个角点由 `(x, y, seed, corner_salt)` 整数哈希得到，再做两次定点双线性插值。运行时不调用 `random`、`numpy.random` 或全局随机状态。

噪声输出范围 `[0, 2^F]`。每层得到：

- `raw`：量化前整数；
- `quant_step`：本层使用的量化步长；
- `quantized`：按 `quant_step` 和 tie rule 舍入后的整数；
- `residual = raw - quantized`；
- `decisions`：down/up/tie-down/tie-up 的 `int8` 标记。

因此切换量化档位会真实改变中间整数、残差和缓存键，而不是只改变显示。

## 4. 材质通道与混合

基础纹理到通道的映射：

- 漫反射：RGB，每个分量 `u8 -> [0, 2^F]`。
- 粗糙度：`2^F - luma`。
- 金属度：RGB 的 max-min（饱和度代理）。
- 法线扰动：luma 的整数中心差分 `gx/gy` 与常量 z 组成三分量。

通道混合只接收该通道声明的层。例如 `roughness` 的混合任务拿不到 `metallic` 的中间数组。支持三种模式：

- `weighted`：`Σ(value * weight) / Σweight`，使用配置的 tie rule 舍入。
- `max`：候选值先按 `weight/1000` 定点缩放，再按层序号稳定遍历取最大；相等值由 `equal_weight_rule` 决定保留低位还是高位序号。
- `layered_over`：按层序号逐个做定点 alpha 混合，每一步都记录舍入余数。

最终 RGBA 合成使用四个通道：粗糙度/金属度生成高光项，法线扰动偏移漫反射 RGB，粗糙度与金属度共同生成 alpha。所有中间量都夹紧到 `[0, 2^F]`。

## 5. 缓存键与精确失效

所有键都是稳定 JSON 的 SHA-256：

```text
texture_key     = H(texture dimensions/mode/content bytes)
layer_param_key = H(texture_key, layer params, resolution, seed, algorithm)
layer_key       = H(texture_key, layer_param_key, fraction_bits, quant_step, tie_rule)
channel_key     = H(texture_key, referenced layer keys, channel weights,
                    blend mode, tie/equal rules, fraction_bits, quant_step)
composite_key   = H(all channel keys, blend mode, tie rule,
                    fraction_bits, quant_step, resolution)
```

失效规则：

- 替换纹理：所有层键变化，层/通道/最终产物重算。
- 调整某层频率或振幅：只改变该层参数键及其下游通道；其他层直接复用。
- 调整某通道权重：所有层键复用；仅该通道键变化，其他通道复用。
- 切换混合模式：层复用；通道与最终合成重算。
- 切换量化档位：所有层键变化（因为量化结果和残差必须重算），通道与最终合成随之下游变化。

每批账本按稳定顺序记录 `texture -> layers(升序) -> channels(固定四通道顺序) -> composite`。每行包含 `key`、`action`（`load/reuse_hit/recompute/recompute_complete/resume_partial/pending/failure`）、`reason`、量化步长、位宽、引用关系、点数与业务码。这样可以直接审计“为什么复用/为什么失效”。

## 6. 预算调度与确定性

渲染批次先串行生成计划，再把计算任务提交线程池。层按升序获得连续像素块：

```text
allocation = min(remaining_layer_points, layer_points_budget, remaining_batch_budget)
```

每层每批最多消费 `layer_points`，所有层共享 `batch_points`。未完成部分保存在 `partials`，待办队列标记 `E_POINT_BUDGET_PENDING`。下一次命令/`advance` 从连续位置续算。

线程池大小严格取 `min(config.thread_pool_size, task_count)`。任务只返回自己的数组片段，不写入共享结果；主线程按计划序号合并。层、通道任务完成后统一按固定顺序构建账本、失败清单和待办清单。因此并发度、任务完成顺序和预算耗尽时机不会改变逐字段结果。

## 7. 预览与提交

每次合法命令后，渲染批次在后台线程启动。会话最多等待 `preview_budget_ms`：

- 批次在预算内完成：返回最新摘要，`stale=false`，包含完成层数、通道摘要、受影响层、点进度、待办/失败/业务码。
- 批次未完成：立即返回上一份可用摘要，标记 `stale=true`、`stale_reason=E_PREVIEW_STALE`，并同时给出新命令预计受影响的层。

到达 `commit_interval_commands` 时，如果批次已完成则立即提交；否则记录待提交状态，在下一条命令开始前或 `finalize/recover` 时等待批次完成并提交，避免把不完整阶段写成当前快照。

## 8. 快照与强杀恢复

快照写入 `<state>/snapshots/.tmp-<uuid>/`，包括：

- `manifest.json`：版本、源配置/材质哈希、有效配置哈希、纹理内容键、层/部分层/通道/合成元数据、待办、失败、注入命中、账本行数、所有产物 SHA-256。
- `artifacts/*.npz`：层 raw/quantized/residual/decisions、部分层、通道、最终合成、纹理像素。
- `config.json`、`material.json`、`ledger.jsonl`、`outputs/*.png`。

临时目录内容 `fsync` 后整体 rename 成 `snapshots/vNNNNNNNN`；随后原子替换 `CURRENT` 指针。强杀发生在 rename 前或 pointer 替换前时会继续使用旧指针，不会出现半快照。

启动时：

1. 读取 `CURRENT` 并校验其版本与 manifest 版本一致。
2. 校验 manifest 中每个产物的 SHA-256。
3. 比较当前源配置/材质哈希。不一致则以当前文件重建，并报告 `E_CONFIG_CHANGED_REBUILD`。
4. 源配置/材质一致则恢复有效配置、材质、纹理、层、部分层、待办队列和账本。
5. 若纹理文件内容在会话外被替换，报告 `E_TEXTURE_EXTERNAL_CHANGE` 并清空渲染产物，下次批次按键自然全量失效。

恢复后的续跑与使用相同有效输入的全量重建逐像素、逐键一致。

## 9. 误差传播与量化归因

`build_error_report` 同时计算：

- 主路径：配置位宽 `F` + 层量化。
- 参考路径：`F+1` 位宽、不对层做量化。

比较四个阶段：

1. `layer_noise_synthesis`：主路径 raw 与参考 raw 下采样到 F 的逐像素差。
2. `layer_quantization`：主路径残差绝对值。
3. `material_mixing`：主路径通道整数与参考通道下采样的逐像素差。
4. `channel_composite`：主路径 RGBA 与参考 RGBA 下采样的逐像素差。

容差按量化步派生：`steps>0 ? step*steps : max(1, step//4)`。`stage_diffs.npz` 保存每层、每通道和逐像素 `first_exceed_stage`；JSON 报告给出每阶段最大/平均偏差、首超坐标与全局首超阶段。

`compare_quant_steps` 对两档量化分别做完整主路径，逐像素比较最终 RGBA。分类优先级：

1. `boundary_decision`：该像素出现量化或混合 tie 决断。
2. `quantization_rounding`：任一引用层的量化整数变化。
3. `mixing_rounding`：通道混合整数变化。
4. `channel_composition`：仅最终合成变化。

`quant_compare.npz` 保存分类图、tie mask、两档输出以及层合成/量化/混合/最终合成四类逐像素偏差；JSON 给出类别计数和前 64 个不一致像素的各环节偏差样例。

## 10. 故障注入

`samples/injections.json` 使用：

```json
{
  "layer": 2,
  "stage": "layer_quantize",
  "channel": null,
  "delay_ms": 0,
  "code": "E_INJECTED_LAYER_QUANTIZE",
  "message": "..."
}
```

阶段可为 `noise_synthesis`、`layer_quantize`、`material_mix`、`channel_compose`。故障在真实任务内部抛出 `BizError`，不跳过执行流程。失败被记录为含 `layer/stage/channel/code/injected` 的稳定排序报告，其他可独立层与通道继续处理；依赖失败层的通道记录 `E_CHANNEL_DEPENDENCY`，最终合成不会伪造成功产物。

## 11. 常用命令

```bash
# 兼容性/环境预检
python3 -m fixedtex env

# 校验并输出合法项与全部 E_xxx
python3 -m fixedtex validate --config samples/config.json --material samples/material.json

# 命令流渲染 + 最终提交
python3 -m fixedtex run --config samples/config.json --material samples/material.json \
  --commands samples/commands.jsonl --state-dir artifacts/demo --fresh

# 仅消费一个预算批次，观察 E_POINT_BUDGET_PENDING
python3 -m fixedtex run --config samples/config_low_budget.json --material samples/material.json \
  --commands <one-render-command.jsonl> --state-dir artifacts/low --fresh --no-drain --no-commit

# 从最新快照继续并提交
python3 -m fixedtex recover --config samples/config.json --material samples/material.json --state-dir artifacts/demo

# 四阶段误差与量化归因
python3 -m fixedtex diagnose --config samples/config.json --material samples/material.json \
  --state-dir artifacts/demo --output-dir artifacts/diagnostics --quant-steps 4,8

# 完整验收
python3 -m fixedtex verify --output-dir artifacts/verification
```

## 12. 已知环境限制

当前执行账户无 sudo，系统 `python3-tk` 不能通过 apt 安装。已用 `apt-get download` 将 Tk/Tcl 及 X 依赖解包到本地，验证 Tkinter 8.6.12 可导入并创建 Tk 根窗口；但没有把这些二进制库写入仓库。渲染、CLI、测试和验收全部为无头运行，不依赖 Tk。
