# 本地确定性生成式图像编辑互操作工作台

这是一个完全离线、无 GPU、无容器依赖的 Python 3.11+ 工作台，用固定种子样本验证编辑指令与编辑结果在 **v1 定宽文本** 与 **v2 JSON** 两种交换版本之间的互操作、精度损失、区域级增量失效和逐像素一致性。

核心能力：

- v1 定宽字段 / v2 JSON 解析，统一为可校验的中间表示。
- v2→v1 降级旁路元数据，附带坐标残差、强度残差、精度标签、旋转与羽化；v1→v2 可恢复为与降级前逐字段一致。
- 全整数定点渲染：旋转矩形掩码、羽化 alpha、由外向内邻域填洞、属性重着色、量化档位 LUT、PNG 合成。
- 六阶段内容寻址缓存：`parse → mask → repair → attribute → style → compose`。
- 区域级精确失效、阶段级复用、单批像素预算、稳定顺序待办队列。
- 线程池并发渲染，并与串行结果逐像素一致。
- 临时工作目录整体提交、原子 `CURRENT` 指针、快照恢复与续跑。
- v1/v2 跨版本偏差直方图、边界归因、`E_BOUNDARY_SENSITIVE` 诊断。
- 独立注入文件和命令序列，不需要修改源码制造异常。
- 静态 HTML + PNG 可视化报告，无外部前端资源。

## 快速开始

### Linux / WSL

```bash
python3 --version                    # 需要 >= 3.11
python3 -m pip install --no-index --no-build-isolation -e . --no-deps   # NumPy/Pillow 由本地环境或 wheelhouse 提供
export PYTHONPATH=src
python3 scripts/generate_samples.py                  # 可选：样例已随仓库生成
python3 -m pytest -q
python3 -m image_edit_workbench validate-injections --root .
python3 -m image_edit_workbench run \
  --root . \
  --commands scenarios/incremental_sequence.jsonl \
  --reset-state
```

打开 `report/index.html` 即可查看离线可视化报告；机器可读报告在 `report/diagnostics.json`。

### Windows PowerShell

```powershell
py -3.11 --version
py -3.11 -m pip install --no-index --no-build-isolation --find-links .\wheelhouse -e . --no-deps
$env:PYTHONPATH = "src"
py -3.11 scripts\generate_samples.py
py -3.11 -m pytest -q
py -3.11 -m image_edit_workbench validate-injections --root .
py -3.11 -m image_edit_workbench run --root . --commands scenarios\incremental_sequence.jsonl --reset-state
```

也可以直接运行：

```powershell
.\scripts\run_windows.ps1
```

所有路径、原子替换、硬链接降级复制和线程池逻辑均兼容 Windows。当前容器实际执行环境是 WSL2/Ubuntu；本仓库无法声称“已在 Windows 本机执行”，详见 `docs/VERIFICATION.md`。

## 目录

```text
src/image_edit_workbench/   工作台实现
  codec_v1.py               v1 定宽编解码
  codec_v2.py               v2 JSON 编解码
  convert.py                双向转换、降级残差、旁路恢复
  renderer.py               整数掩码、填洞、属性、风格、合成
  pipeline.py               阶段缓存、预算调度、线程池、图像合成
  snapshot.py               不可变快照与原子 CURRENT 指针
  runner.py                 命令状态机、冷/热等价执行、续跑
  diagnostics.py            跨版本诊断、注入校验
  reporting.py              JSON/HTML/PNG 报告
samples/                    固定种子 PNG、清单、v1/v2 交换文件、旁路
config/default.json         渲染与调度配置
injections/                 独立异常样本与用例清单
scenarios/                  增量、预算、异常命令序列
tests/                      14 项自动化测试
report/                     已生成的最终诊断与可视化报告
docs/                       格式、架构与实测记录
```

## 命令入口

```text
python -m image_edit_workbench generate-samples
python -m image_edit_workbench validate-injections
python -m image_edit_workbench convert --from-version 2 --to-version 1 ...
python -m image_edit_workbench convert --from-version 1 --to-version 2 --sidecar-in ...
python -m image_edit_workbench run --commands scenarios/incremental_sequence.jsonl --reset-state
python -m image_edit_workbench run --resume          # 从快照情景状态继续，优先处理 pending
```

`run --batch` 每次只执行一个预算批次；`--no-verify-each-command` 可跳过逐命令冷启动对照以缩短试运行时间。

## 数据流

1. 固定种子生成 3 张 `256x256` PNG、6 条编辑指令、v1 定宽交换文件、v2 JSON 交换文件和降级旁路。
2. 解析交换文件并用清单校验图名、区域序号、图像边界和字段完整性。
3. 为每个区域计算六阶段缓存键；命中时直接加载持久化数组，不执行对应像素算法。
4. 调度器在 `recompute_region_budget` 内按稳定顺序分配区域；预算不足的区域进入 pending，并记录 `E_RECOMPUTE_BUDGET_PENDING`。
5. 线程池并发执行受影响区域，结果按稳定键排序后合成图像。
6. 命令序列按 `snapshot_commit_interval_commands` 提交不可变快照；清单记录配置哈希、交换版本指纹、旁路哈希和已完成区域。
7. 每条命令都与同状态冷启动全量结果比较像素指纹、阶段摘要、内容指纹和 pending 顺序。
8. 最终报告同时执行 v1/v2 独立渲染、偏差直方图、边界归因和注入用例校验。

## 关键语义

- **定点渲染**：像素与风格混合使用 `int64`；坐标/三角/羽化使用整数缩放；舍入固定为 `half_up`。浮点只参与三角常数初始化。
- **内容指纹**：阶段键由上游摘要、参数指纹和交换版本派生；调度配置（线程数、预算、快照间隔）不进入渲染内容指纹。
- **真实像素变化**：量化档位、定点小数位宽、旋转和羽化都会改变缓存键、阶段数组与输出像素。
- **快照**：阶段文件先写入 `.work`，完成清单后整体重命名为新快照，再原子替换 `CURRENT.json`。
- **续跑**：快照内保存 `scenario_state`；`--resume` 恢复区域、配置、交换版本和 pending 队列，下一批优先处理旧待办。
- **失败隔离**：单条 v1 截断、单条 v2 结构缺失或单区域越界只进入失败清单，其他合法区域继续渲染。

详细格式与缓存公式见 `docs/FORMATS.md`。

## 已实测结果

本次最终验收命令：

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m image_edit_workbench run --root . \
  --commands scenarios/incremental_sequence.jsonl --reset-state
```

- 自动化测试：`14 passed`。
- 11 条命令全部满足：最终 PNG 像素、各阶段摘要、内容指纹、pending 顺序与冷启动全量结果一致。
- 最终内容指纹：`52b2cd48b9b9b4c3904f0b6cce209469a19f6d9749da1d904556b84c861f6310`。
- 最终 pending 队列为空；全过程最大并发任务数 4，配置上限 4。
- v1/v2 旁路恢复：6/6 命中、6/6 字段一致；修补方式映射 6/6 恢复。
- 风格强度最大偏差：`0.0005`；5 个区域被归因为边界敏感并记录 `E_BOUNDARY_SENSITIVE`。
- 4 个独立注入用例全部命中预期业务码。
- 快照提交 6 次，提交间隔 2 条命令。

异常命令序列的独立报告在 `report/abnormal/index.html`，其中 `INJ-MIXED-V1` 命中 1 条截断错误，同时 6 个合法区域继续渲染并与冷启动全量结果一致。完整证据见 `report/diagnostics.json`、`report/index.html` 和 `docs/VERIFICATION.md`。

## 能力边界

“全量冷启动”的原始 `reused_regions` / `recomputed_regions` 执行日志与增量运行**从定义上不可能相同**：增量运行会真实复用，冷启动必须重新计算。报告保留 `raw_reuse_lists_equal` 明确展示这一差异；验收等价性以像素、阶段摘要、内容指纹、待办顺序和归一化语义为准，不伪造成两个不同执行策略的原始日志相同。

跨版本渲染本来就可能不同：v1 无法表达浮点坐标、千分比以外强度和旋转/羽化。工作台不会把这种信息损失隐藏为“相同”，而是通过旁路、偏差直方图和边界归因显式呈现。
