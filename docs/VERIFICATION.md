# 实际验证记录

验证日期：2026-11-26（Asia/Shanghai）  
工作目录：`/home/zsl/feedback/uidfc5909d3-default-326`  
环境：WSL Ubuntu 22.04，Python 3.11.0rc1，NumPy 1.26.4，Pillow 10.3.0，pytest 8.2.0。

## 1. 自动化测试

命令：

```bash
python3 -m pytest
```

实测结果：

```text
..........................                                               [100%]
28 passed in 12.91s
```

覆盖内容：

- 配置/材质/PNG 容器与像素模式校验，非法项容错与 `E_xxx`。
- 定点舍入、ties-to-even/tie-down/tie-up。
- 坐标哈希噪声跨进程/跨切块确定性。
- 量化档位切换确实改变整数与残差。
- 线程池并发度 1/4 时逐层数组、通道、最终合成、账本、失败清单逐字段一致。
- 单层/单批点预算耗尽、`E_POINT_BUDGET_PENDING` 与多批续算。
- 通道权重编辑只重算目标通道；其他通道账本 `reuse_hit`。
- 单层频率编辑只重算该层和引用它的通道；无引用通道复用。
- 纹理替换使所有层及下游失效；量化档位切换重算全部量化产物。
- 四阶段逐像素误差、首超环节、两档量化分类与 NPZ 产物。
- 预览预算超时返回 `E_PREVIEW_STALE` 陈旧摘要。
- 部分批次快照恢复、配置哈希变化重建、快照 artifact SHA 损坏检测。
- 子进程渲染期间真实 `SIGKILL`，版本不前进，重启后续跑与全量重建一致。
- CLI 退出码、命令流渲染、`--no-drain` 待办码、完整 `verify`。

## 2. 完整验收

命令：

```bash
python3 -m fixedtex verify --output-dir artifacts/verification
```

实测退出码：`0`，最近一次总耗时 `3.94s`。报告：`artifacts/verification/report.json`。

| 验收项 | 结果 | 关键证据 |
|---|---:|---|
| 增量渲染 vs 全量重建 | 通过 | 5/5 层全字段相等，4/4 通道相等，composite key 相同，PNG 像素相同 |
| 完整快照恢复 vs 全量重建 | 通过 | 恢复来源 `snapshot:v00000002`，PNG 像素相同，快照根 `output.png` 相同 |
| 部分批次快照恢复 | 通过 | 快照时待办 10 项，续跑后 composite key/array 全等 |
| 阶段故障隔离 | 通过 | 两个注入均命中；失败清单稳定；层 0/1/3/4 仍完成 |
| 四阶段误差传播 | 通过 | 首超阶段 `layer_noise_synthesis`，首超像素 3402 |
| 两档量化归因 | 通过 | 3163 像素不一致，四类归因均输出 |

增量与全量最终键：

```text
incremental composite = 9b6c4662abb2cf94fb71622eead86d6971ed8cc3edcc984d4af8f807c8ee63f5
full rebuild composite = 9b6c4662abb2cf94fb71622eead86d6971ed8cc3edcc984d4af8f807c8ee63f5
PNG pixels equal       = true
incremental PNG sha256 = 3c5323d337a94a00c46e8bb0983cdf899b122074c01c6da93fa31ac4e59fce21
```

验收样本执行的命令序列：通道权重调整、层频率/振幅调整、`weighted -> max`、量化档位 `4 -> 8`、纹理替换、同位宽重写、通道权重调整、量化档位 `8 -> 4`、`max -> weighted`、显式提交。最终 `output.png` 与使用最终有效配置重新全量渲染的图像逐像素一致。

## 3. 缓存失效与复用账本

完整验收的增量账本位于：

```text
artifacts/verification/state_incremental/ledger.jsonl
```

共 99 行。按命令序号汇总的实际动作：

| 序号 | 编辑 | 层动作 | 通道动作 | 其他 |
|---:|---|---|---|---|
| 1 | roughness 权重 | 5 层全算 | 4 通道全算 | composite 全算 |
| 2 | layer 1 频率/振幅 | 1 层重算，4 层复用 | 3 通道重算，1 通道复用 | composite 重算 |
| 3 | 切到 max | 5 层复用 | 4 通道重算（blend_mode_changed） | composite 重算 |
| 4 | quant 4 -> 8 | 5 层重算（quant_step_changed） | 4 通道重算 | composite 重算 |
| 5 | 替换纹理 | 5 层重算（base_texture_changed） | 4 通道重算 | texture load |
| 6 | 重写同 frac bits | 5 层复用 | 4 通道复用 | composite 复用 |
| 7 | diffuse 权重 | 5 层复用 | 仅 diffuse 重算；3 通道复用 | composite 重算 |
| 8 | quant 8 -> 4 | 5 层重算（quant_step_changed） | 4 通道重算 | composite 重算 |
| 9 | 切回 weighted | 5 层复用 | 4 通道重算（blend_mode_changed） | composite 重算 |

第 7 条直接验证了“分层权重只影响部分通道”：目标通道 reason 为 `channel_weight_changed`，另外三通道 reason 为 `cache_key_match`，所有层为 `reuse_hit`。

## 4. 误差传播与量化归因

诊断产物：

```text
artifacts/verification/diagnostics/stage_diffs.npz
artifacts/verification/diagnostics/error_report.json
artifacts/verification/diagnostics/quant_compare.npz
artifacts/verification/diagnostics/quant_compare.json
artifacts/verification/diagnostics/diagnostics.json
```

参考路径为 `fraction_bits=13` 的无量化路径；主路径为 `fraction_bits=12, quant_step=4`。容差为 1 个定点单位（`error_tolerance_quant_steps=0` 的规则 `max(1, step//4)`）。

实测首超：

```text
first_exceed_stage = layer_noise_synthesis
first_exceed_count = 3402 / 4096 pixels
```

量化档位 4 vs 8：

```text
mismatch_pixels          = 3163
boundary_decision        = 2860
quantization_rounding    = 297
mixing_rounding          = 6
channel_composition      = 0
```

`quant_compare.npz` 同时保存：

- `classification`、`mismatch`、`tie_mask`；
- `quant_a`、`quant_b` 两档最终 RGBA；
- `synthesis_deviation`、`quantization_deviation`、`mixing_deviation`、`composite_deviation` 四类逐像素偏差。

JSON 中另有每个不一致样例的 x/y、分类、四类偏差数值和两档 RGBA 值。

## 5. 故障注入隔离

注入配置：`samples/injections.json`。

实测：

```text
expected hits = [E_INJECTED_LAYER_QUANTIZE, E_INJECTED_MATERIAL_MIX]
actual hits   = [E_INJECTED_LAYER_QUANTIZE, E_INJECTED_MATERIAL_MIX]
failures      = [E_CHANNEL_DEPENDENCY, E_INJECTED_LAYER_QUANTIZE, E_INJECTED_MATERIAL_MIX]
completed     = layers [0, 1, 3, 4]
```

层 2 在量化阶段真实抛错；层 3 的噪声仍完成；通道依赖层 2 的部分记录 `E_CHANNEL_DEPENDENCY`；对 layer 1/diffuse 混合阶段的故障真实命中并阻断该通道。其余可独立层与通道继续计算。失败列表按阶段、层号、通道、业务码稳定排序。

## 6. 强杀恢复

测试 `test_sigkill_before_commit_keeps_previous_snapshot` 的实际流程：

1. 完成一次渲染并提交快照，记录 `VERSION`。
2. 启动子进程，加载该快照，应用新层频率命令并注入 4 秒阶段延迟。
3. 子进程在后台批次执行期间收到 `SIGKILL`。
4. 检查 `VERSION` 未变化，说明未产生半提交。
5. 新进程重新打开同一快照，续跑并完成。
6. 与全量重建逐层 raw/quantized/residual/decisions、逐通道数组、composite key/array 对比，全部相等。

完整验收中的部分快照恢复也验证了待办队列持久化：第一批后待办 10 项提交，恢复后续跑，最终 composite key/array 与全量重建相同。

## 7. 环境预检

`python3 -m fixedtex env` 实测：

```text
Python        3.11.0rc1
NumPy         1.26.4
Pillow        10.3.0
Tkinter       默认解释器缺 libtk8.6.so（系统 python3-tk 未安装，账户无 sudo）
```

为确认不是代码/GUI 设计问题，使用 `apt-get download` 将 Tk/Tcl 及 X 依赖解包到临时本地目录后，实测：

```text
tkinter import ok 8.6
Tk root ok 8.6.12
```

因此 Tk 8.6.12 可用；仅默认解释器未带系统级库路径。本交付的渲染、CLI、测试和验收均不依赖 GUI，可完全无头运行。

## 8. 复现实命令

```bash
python3 tools/generate_samples.py
python3 -m pytest
python3 -m fixedtex verify --output-dir artifacts/verification
python3 -m fixedtex env
```

最终验收图像：

- 增量运行：`artifacts/verification/incremental.png`
- 全量重建：`artifacts/verification/full_rebuild.png`
- 快照恢复：`artifacts/verification/recovered.png`
- 恢复全量对照：`artifacts/verification/recovered_full_rebuild.png`
