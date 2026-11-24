# 竖排古籍版面识别与增量编辑工具

这是一个从零实现的本地确定性应用，面向固定分辨率 PNG 竖排古籍合成版面。它完成：

- PNG/真值 JSON 严格校验与灰度解码；
- 从右到左的栏检测；
- 栏内字符投影切分、字符边框回归；
- 粘连块动态规划最优分割；
- 内置 16×16 点阵字模的多模板定点匹配；
- 逐字符定点置信度传播与低置信原因；
- 内容寻址缓存、栏/字符级精确失效；
- 单批模板点数预算、跨批续跑；
- 原子阶段快照、启动续跑/陈旧检测；
- 线程池并发下的稳定排序与确定性报告。

实现只依赖本机文件、`numpy` 与 `Pillow`，不访问网络、账号或外部数据集。

## 环境

已在以下环境验证：

- Python 3.11.0rc1（代码使用 3.10+ 语法）
- numpy 1.26.4
- Pillow 10.3.0

Windows 安装示例：

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS 对应激活命令为 `source .venv/bin/activate`。

## 快速开始

```bash
# 1. 生成确定性的 3 张 PNG、同名真值和配置
python -m gsb_layout generate-samples --samples samples

# 2. 从空白状态全量执行
python -m gsb_layout run \
  --samples samples \
  --config samples/config.json \
  --state state \
  --report reports/report.json \
  --full

# 3. 无改动增量重跑（应全部命中缓存）
python -m gsb_layout run \
  --samples samples \
  --config samples/config.json \
  --state state \
  --report reports/report.json

# 4. 执行编辑序列 + 每步全量对照 + 无改动全复用检查
python -m gsb_layout verify \
  --samples samples \
  --config samples/config.json \
  --workdir .verify \
  --report reports/verification.json
```

所有入口均为 `python -m gsb_layout ...`，不依赖 IDE 或工作目录中的可执行脚本。

## 目录

```text
gsb_layout/
  algorithms.py       投影、DP 分割、回归、定点匹配、置信度
  cache.py            内容寻址缓存与原子 JSON 写入
  cli.py              python -m 命令行
  config.py           配置校验、别名、参数指纹
  errors.py           业务错误码
  io_layer.py         PNG/真值扫描与严格校验
  pipeline.py         阶段调度、预算、并发、缓存追踪、报告
  sample_generator.py 确定性样本生成
  snapshot.py         阶段快照、清单、陈旧检测
  templates.py        内置点阵字模与模板版本
  verification.py     增量/全量验收阶梯
tests/                15 个标准库 unittest 测试
samples/              本地 PNG、真值、配置
reports/              最终运行报告与验收报告
```

## 输入契约

### PNG

- 扫描目录内所有后缀为 `.png` 的文件，按 `(文件名 casefold, 文件名)` 排序。
- 校验 PNG 签名、`IHDR` 长度/字段、位深、颜色类型、压缩/滤波/交错字段。
- 支持灰度 `L/1/I/I;16` 及可转为灰度的 `RGB/RGBA/P`；统一转为二维 `uint8`。
- 若配置提供 `expected_width/expected_height`，尺寸必须匹配。
- 单图失败不会中断其余图像；失败码为 `E_IMAGE_INVALID`。

### 真值 JSON

可用文件名：`<stem>.truth.json`、`<stem>.layout.json`、`<stem>.json`。示例：

```json
{
  "image": "page_001.png",
  "columns": [
    {
      "reading_index": 1,
      "bbox": [405, 32, 430, 86],
      "chars": [
        {"reading_index": 1, "bbox": [407, 32, 429, 56], "text": "日"},
        {"reading_index": 2, "bbox": [408, 57, 426, 79], "text": "月"}
      ]
    }
  ]
}
```

校验内容：

- 所有 bbox 必须为四个整数，且在图像范围内；
- 同栏字符按几何自上而下位置排序后，阅读序号必须严格递增；
- 同栏字符 y 高度区间不得重叠；
- 字符内容必须为非空字符串；
- 栏/字符阅读序号不得重复，真值 `image` 与文件名一致；
- 失败码为 `E_TRUTH_INVALID`，其余图像继续处理。

`reading_index/reading_order/index/order`、`text/char/character/label`、`bbox/box` 均作为兼容别名读取。

## 识别配置

最终样本配置见 `samples/config.json`：

```json
{
  "binarization_threshold": 160,
  "row_projection_window": 5,
  "char_min_height": 14,
  "char_max_height": 30,
  "split_penalty": 12,
  "match_metric": "iou",
  "template_weights": [4, 3, 2],
  "fixed_bits": 10,
  "quant_levels": 16,
  "confidence_tolerance": 0.62,
  "thread_count": 4,
  "match_point_budget": 2000000,
  "snapshot_interval": 1,
  "expected_width": 480,
  "expected_height": 360,
  "border_residual_tolerance": 5,
  "confidence_low_q": 0,
  "confidence_high_q": null
}
```

- `match_metric`：`l1`、`hamming`、`iou`、`cosine`。
- `fixed_bits` 定义定点比例 `Q=2^fixed_bits`；匹配分与置信度均使用整数传播。
- `template_weights` 必须与内置模板变体数（当前 3）一致。
- 支持常见字段别名，如 `threshold`、`projection_window`、`min_height`、`max_height`、`metric`、`threads`、`width/height`。

## 算法说明

### 栏检测

1. 用 `gray < binarization_threshold` 得到墨迹二值图。
2. 在 x 方向累加墨迹得到竖直投影。
3. 用 `row_projection_window` 做整数滑动平均。
4. 以低密度谷切出连续带，再用原始 ink 支撑边界收紧为整数半开区间 `[x0,x1)`。
5. 按 `x0` 从大到小排序，得到从右到左的栏序号。

### 字符投影切分

- 在每栏 x 范围内沿 y 方向投影，经同一窗口平滑并按空白谷切成候选段。
- 所有候选段保留，`height < char_min_height` 标记 `under_min`，`height > char_max_height` 标记 `over_max`，均进入回归，不直接丢弃。
- 排序、切分和边界收紧全部使用整数逻辑，重复运行逐字段一致。

### 边框回归

对候选段统计墨迹紧边界、每行墨迹数与墨迹重心：

- 目标中心由 3 份重心与 2 份几何中心合成；
- 目标高度限制在 `[char_min_height, char_max_height]`；
- 再结合上下留白与紧边界扩展/裁剪；
- 输出整数 bbox、紧 bbox、重心定点值、中心偏差、高度偏差和回归残差。

### 粘连块最优分割

高度超过 `char_max_height` 时使用精确动态规划：

- 枚举可行段数；
- 每个候选切点代价 `valley_cost + balance_cost`：
  - `valley_cost = (当前行平滑投影 / 最大投影 × 1000)^2 / 100`；
  - `balance_cost = |切点 - 均分理想点| × (4 + split_penalty) × 128`；
  - 另对额外段数乘 `split_penalty` 支付段数代价。
- 累计代价相等时按切点序列字典序选更靠上的方案，并设置 `tied=true`。
- 每个字符记录实际罚项、切点数、切点、逐切点代价、逐段代价和并列标记。
- 修改罚项会重新进入 DP 并真实改变切点；单元测试也覆盖了不同罚项产生不同切点及并列取上界。

### 多模板定点匹配

- 内置 20 个 16×16 点阵字模，每个字符 3 个变体（原形与轻微旋转）。
- 字符块与模板先紧裁墨迹，缩放到模板网格，再按 `quant_levels` 量化。
- 每个变体产生 `[0,Q]` 定点分；按 `template_weights` 加权整数求和后整除权重和。
- 最高分模板为识别字符；同分按 Unicode 码点升序决策，并记录所有并列字符。

### 置信度

固定顺序：

1. `score_q`；
2. 减边框残差罚项；
3. 减粘连分割代价罚项；
4. 裁剪到 `[confidence_low_q, confidence_high_q]`。

以下任一情况标记低置信，并给出结构化原因：

- `TIE_BREAK`：模板并列；
- `BORDER_RESIDUAL`：回归残差过大；
- `SPLIT_BLOCK`：字符来自粘连块分割；
- `LOW_MATCH_SCORE`：匹配分低于容差；
- `LOW_CONFIDENCE`：最终置信度低于容差。

## 缓存与精确失效

缓存键由阶段名、算法版本、图像/子图内容指纹、参数指纹和字模模板版本派生：

| 阶段 | 内容指纹 | 主要参数 |
| --- | --- | --- |
| 栏检测 | 整图 SHA-256 | 阈值、投影窗口 |
| 投影切分 | 栏二值图摘要 | 投影窗口；阈值通过二值图体现 |
| 边框回归 | 候选段二值图摘要 | 实际生效的高度约束；仅 over-max 才含罚项 |
| 模板匹配 | 字符块摘要 | 度量、权重、定点位宽、量化档位、模板版本 |

由此：

- 只改高度约束：切分复用；仅受边界夹取影响的回归重算；
- 只改分割罚项：栏与切分复用；仅 over-max 回归及其下游重算；
- 只切换匹配度量：栏/切分/回归复用；仅模板匹配重算；
- 只替换一张图：该图内容指纹变化，其余图复用；
- 报告在 `cache.reuse_recompute` 中逐栏、逐字符列出 `columns/segments/regression/template_match` 的 `reuse/recompute/pending`、缓存键和稳定任务键。

## 点数预算

- 单个字符的比对点数：`bbox_width × bbox_height × 模板变体数`。
- 任务顺序：图像名 → 从右到左栏序号 → 自上而下字符序号。
- 缓存命中的字符不扣预算。
- 一旦下一个未缓存任务放不下，本轮后续未缓存任务全部保留为 pending，避免乱序消耗。
- pending 项输出 `E_MATCH_BUDGET_PENDING`，下一轮前面的已完成项直接缓存命中，pending 队列优先继续。
- 报告包含 `budget.sequence`、扣减后余额、pending 顺序及稳定任务键。

## 快照与恢复

`state/` 下有两类持久状态：

- `state/cache/<stage>/<key>.json`：每个阶段产物先写同目录临时文件、`fsync`，再用 `os.replace` 原子替换；
- `state/snapshot/`：`manifest.json` 与 `stages/{columns,segments,boxes,matches}.json`。

清单包含：

- `format_version`、`pipeline_version`；
- `global_version`；
- `config_hash`；
- `completed_images`、`image_hashes`；
- 四个阶段产物版本；
- 预算与 pending 数量。

启动时比较快照格式/流水线版本与配置哈希。配置哈希不一致时输出 `E_SNAPSHOT_STALE` 诊断并重建快照，同时仍可复用内容寻址缓存中真正未受影响的产物。`snapshot_interval` 控制布局阶段的逐图检查点，最终四阶段产物再整体提交。

## 并发确定性

- 栏检测按图像提交线程池；模板匹配按字符提交线程池。
- 工作线程数不超过 `thread_count`。
- 每个阶段任务都有稳定键，例如 `page_001.png|0002|0003|match`。
- 即使线程完成顺序不同，结果按稳定任务序列收集；预算扣减在提交线程前按稳定顺序完成。
- 图像、栏、字符、失败、预算序列和 pending 顺序均固定。

## 报告字段

`reports/report.json` 包含：

- 运行模式、全局版本、配置哈希、模板版本、陈旧诊断；
- 每张图的 image SHA、尺寸、模式、真值；
- 每栏整数 `[x0,x1)`、候选段及高度分类；
- 每字符 bbox、文本、置信度定点值与浮点展示、完整传播明细；
- 低置信原因统计、匹配/分割并列数量与明细；
- 边界候选清单（under-min/over-max、分割动作、切点数）；
- 按 `F0001...` 编号且稳定排序的失败清单；
- `cache.reuse_recompute` 逐栏逐字符清单；
- `budget` 扣减序列与 pending；
- `threading` 任务键与线程数；
- `snapshot` 全局版本和四阶段产物版本；
- `semantic_digest`：增量/全量比较使用的逐字段语义摘要。

## 验收与实测结果

执行：

```bash
python -m unittest discover -s tests -v
python -m gsb_layout verify --samples samples --config samples/config.json \
  --workdir .verify --report reports/verification.json
```

本机实际结果：

- 15 个测试全部通过（本机约 20 秒）；
- 3 张 480×360 样本、10 栏、29 个字符；
- 样本字符识别 28/29（96.55%），所有字符 bbox IoU ≥ 0.665（平均 0.811）；
- 粘连块分割字符均带 `SPLIT_BLOCK` 低置信标记；
- 全量运行无失败项；
- 编辑阶梯包含：阈值调整、最小/最大高度修改、分割罚项修改、匹配度量切换、单图替换；
- 每一步增量结果与同输入空白全量结果 `semantic_digest` 完全一致；
- 无改动重跑 68 个阶段产物全部复用、0 重算；
- 罚项步骤仅重算 over-max 回归，以及切点真实改变后的 2 个下游匹配；度量步骤仅重算匹配；高度步骤栏/切分全复用；
- 单图替换仅使被替换图重算，其余图复用；
- 低预算测试跨 8 批完成，最终语义与一次高预算全量运行一致；
- 1 线程与 4 线程得到相同语义、预算序列、pending 顺序和失败顺序。

最终空白全量语义摘要为 `86ef4030c83eae6b0afaba5c3d2ab6e6e884481f64ea41ec00bf68a61346857e`；编辑阶梯最终语义摘要为 `8944b58aed20a5f3578ec49bf0afa0279991b077ce7bdcccd4c014357d30177b`（后者输入已包含单图替换）。完整机器可读结果保存在 `reports/report.json`、`reports/verification.json` 与 `reports/test-results.txt`。

### 关于“复用清单逐字段一致”的验收口径

缓存热态的增量运行必然包含 `reuse`，空白全量运行必然全部为 `recompute`，因此缓存动作清单本身不能与全量运行字面相等。本项目将“结果逐字段一致”落在 `semantic_digest` 覆盖的完整图像业务对象上：图像状态/真值、栏区间、候选段、字符 bbox、识别文本、完整置信度传播明细、低置信标记/原因、完整回归与分割记录、并列记录、缓存键等；同时单独断言编辑步骤的失效集合和复用集合符合预期。`reuse_recompute` 作为运维审计元数据独立比对。除该运行态清单外，增量与全量的业务字段完全相同。

## 已知边界

- 点阵字库是为本地合成样本内置的 20 个字符；真实历史字体不在任务范围内。
- 字符“日/口”等低分辨率近似形在强噪声下可能产生低分误识，报告会通过低置信原因显式暴露，而不会静默宣称高置信。
- `char_min_height`、`char_max_height` 若无法构成任何合法分割段，分割器会保留原候选并记录未应用分割，报告边界候选可定位该配置问题。
- 所有样本和输出均为确定性生成；不同 Pillow 主版本的重采样细节可能影响图像生成，但识别/缓存/预算/快照逻辑本身只依赖整数与稳定排序。
