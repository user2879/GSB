# 本地确定性全景分割掩码管线

本工作区实现了一个不依赖 Docker、外部账号或在线服务的全景分割像素掩码处理与验证工具。样本、配置、掩码包、快照和验证报告全部保存在本地，核心算法只依赖 NumPy 与 Pillow。

## 快速开始

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python -m panorama_pipeline generate --out samples
python -m panorama_pipeline run \
  --sample samples \
  --out artifacts/verification_report.json \
  --work artifacts/verification

python -m unittest discover -s tests -v
```

当前环境系统 Python 缺少 `ensurepip`，首次建环境使用了官方 `get-pip.py` 引导；这不影响项目本身。安装完成后，运行与验证均不需要网络。

交互式调参：

```bash
python -m panorama_pipeline interactive --sample samples
```

支持的命令包括 `threshold 0.4`、`retention confidence 0.3`、`formats v1,v2`、`metadata v1:2 0.97 false`、`fuse`、`status` 和 `quit`。也可通过 `--commands commands.json` 回放 JSON 命令序列。

双向转换：

```bash
python -m panorama_pipeline convert --sample samples --direction v1-to-v2 --out /tmp/v2.json
python -m panorama_pipeline convert --sample samples --direction v2-to-v1 \
  --out /tmp/v1.bin --sidecar /tmp/v1_sidecar.json
```

## 输入格式

- `samples/images/panorama_000.png` 等：64x48 RGBA PNG，`packed=(instance_id<<16)|class_id`，四个字节按大端填入 RGBA。`0/0` 表示背景。
- `samples/instances.json`：实例 ID、类别、可选置信度/crowd；v1 缺失的置信度使用 `confidence_default`，缺失 crowd 使用 `default_crowd`。
- `samples/masks_v1.bin`：单文件二进制。28 字节头为 `<8sHHIII>`，依次为 magic、版本、flags、宽、高、实例数；每个实例为 `<III>` 的 ID/类别/行程数，随后是 `<II>` 的起始像素与长度。解析校验非负、非零长度、有序、不重叠、不越界。
- `samples/masks_v2.json`：`format_version=v2`，每个实例包含 ID、类别、置信度、crowd、RGBA 调色板和 base64 像素掩码；严格校验 base64、像素数等于 `width*height`、字节只能是 0/1。
- `samples/config.json`：IoU 阈值、像素保留模式与覆盖率下限、默认置信度/crowd、调色板、交互预算、并发数、快照间隔和边界 epsilon。

任一实例解析失败返回 `E_MASK_FORMAT_INVALID` 并进入稳定排序的失败清单；其他有效实例仍可继续融合。

## 处理语义

**转换**：v1 升 v2 时补齐默认置信度、crowd 与类别调色板；v2 降 v1 时把置信度、crowd 和调色板写入 sidecar，并逐实例生成降级损失清单。转换后重新解析，实例数、类别集合、实例 ID 与逐像素掩码都必须一致。

**融合**：仅比较 v1/v2 同类别实例。匹配对先按候选优先级排序，再一对一裁决，顺序为 IoU 降序、面积加权置信度降序、pair key 字典序。匹配对的掩码取加权支持的并集，置信度按掩码面积加权。像素冲突最终按 crowd 策略、IoU 降序、置信度降序、实例 ID 字典序裁决。支持 `prefer_non_crowd`、`confidence`、`drop_crowd` 三种保留策略。

**边界诊断**：每次候选匹配同时保存 float64 参考 IoU/覆盖率和 6 位小数的快速判定值。若值靠近阈值或覆盖率下限，诊断进入报告；若两者判定不同，记录 `E_BOUNDARY_SENSITIVE` 并标注偏差来源。

**增量失效**：实例缓存键由规范化行程指纹、类别、元数据指纹和格式版本共同派生；融合依赖指纹由参与实例缓存键、IoU 阈值和保留策略派生。元数据替换只重解码目标实例，保留策略变化只重算融合，格式切换只重解码新增格式。事件报告包含 `decoded_instances`、`invalidated_instances`、`reused_instances` 与解码计数。

**预算内交互**：命令在单消费者队列中执行；提交线程最多等待配置预算。预算内完成则返回 `status=complete,is_latest=true`。否则立即返回最近一次完整预览，并带 `E_INTERACT_BUDGET_PENDING`、当前命令序号、请求参数快照版本、预览来源命令序号和当前进度；陈旧预览被强制标记为 `is_latest=false`。

**快照与恢复**：每次提交先写不可变临时代际目录，再原子改名到 `generations/gen-N`，最后原子切换 `current` 符号链接。manifest 记录全局版本、格式版本、配置哈希、实例结果版本、融合结果版本、文件哈希和版本变化。恢复时校验 manifest、文件哈希、实例内嵌版本和融合版本。未发布的半写代际不会被恢复。

**并发确定性**：解码、转换和匹配使用 `max_workers` 线程池。每个任务携带 `stage:source:zero-padded-instance-id` 稳定排序键；结果按同一键回收，失败清单按数值实例 ID、阶段和业务码排序，因此并发输出与单线程按实例 ID 执行一致。

## 模块

- `codec.py`：双格式解析、统一中间表示与失败清单。
- `conversion.py`：v1/v2 双向转换和逐像素复核。
- `fusion.py`：匹配、crowd 策略、稳定裁决与边界诊断。
- `pipeline.py`：缓存键、融合依赖和三类增量失效。
- `interactive.py`：预算受控预览与 PENDING 语义。
- `snapshot.py`：代际快照、原子发布、恢复校验和优雅退出。
- `verification.py`：PNG、转换、交互、并发、快照和全量重建验收。
- `tools/generate_samples.py`：固定种子样本生成器。

## 实际验证结果

本工作区已用固定种子 `20261120` 的本地样本完成验证，结果写入 `artifacts/verification_report.json`。最终类别栅格为 `artifacts/verification/artifacts/final_category_grid.npz`，实例清单为 `artifacts/verification/artifacts/panorama_instances.json`，可视化结果为 `artifacts/verification/artifacts/final_panorama.png`。

- 3 张 64x48 PNG 可读写；`panorama_000.png` 与 v1 掩码逐像素一致。
- v1 解析 8 个实例，v2 解析 8 个实例，无格式失败。
- v1→v2 与 v2→v1 均逐像素一致；降级损失清单记录 8 个 v2 实例。
- 最终识别 6 个跨格式匹配对，类别分布为背景 1273、类别 1 574、类别 2 213、类别 3 781、类别 4 143、类别 5 88。
- 边界样本 `v1:2|v2:2` 的 float64 IoU 为 0.3333333333333333，快速值为 0.333333；参考判定为匹配、实际判定为不匹配，因此产生 `E_BOUNDARY_SENSITIVE`。
- 8 条主交互命令均即时完成；独立预算探针注入 80ms 工作、预算 5ms，确定返回 `E_INTERACT_BUDGET_PENDING`，预览与原响应均为 `is_latest=false`，后台完成后才转为 `is_latest=true`。
- 多次增量交互后的类别栅格与从更新后源元数据全量重建的结果逐像素一致，SHA-256 均为 `6db6b2744269b0e7e72ee724d3924a73de56d25f8fa1a55acb9eba1fa2c0addc`，16 个实例缓存键一致。
- 4 个交互代际快照均通过 manifest 与实例版本校验；强制 SIGKILL 发生在 `current` 指针发布前，重启恢复到此前已发布代际且像素一致，随后成功提交下一代。
- SIGTERM 优雅退出测试完成提交并以整数退出码 0 退出。
- 13 个单元/集成测试全部通过，覆盖固定种子复现、损坏/越界行程、base64 长度、缺失字段、并发失败排序、预算 PENDING、三类失效、转换降级、快照损坏和临时代际忽略。

最近一次完整验收为 `overall_pass=true`。唯一业务诊断码是预期中的 `E_BOUNDARY_SENSITIVE`。
