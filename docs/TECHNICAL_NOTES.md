# 竖排古籍数字化首个可运行版本

本目录是无外部数据、无 Docker、无网络依赖的确定性实现。样本 PNG、版面真值、标注员多版本关键点均由固定种子生成；NumPy/Pillow 负责数组与图像，CLI/Python 标准库负责编排、账本、快照和报告。

## 快速运行

```bash
python3 -m pip install -r requirements.txt
bash scripts/run_acceptance.sh
```

也可分步运行：

```bash
python3 -m vertical_ocr.cli sample --root demo_workspace
python3 -m vertical_ocr.cli validate --root demo_workspace
python3 -m vertical_ocr.cli run --root demo_workspace
python3 -m vertical_ocr.cli acceptance --root artifacts/acceptance-run
```

可选 Tk 浏览器：

```bash
sudo apt install python3-tk
python3 -m vertical_ocr.cli view --root demo_workspace
```

当前精简运行环境可能缺少 `libtk8.6.so`；缺少时 `view` 返回安装提示，不影响 CLI、测试和验收。

## 目录与模块

- `vertical_ocr/generator.py`：固定版面计划、内置字模渲染、轻微旋转/笔画抖动、PNG/真值/多版本关键点生成、命令编辑。
- `vertical_ocr/glyphs.py`：内置点阵/笔画字模、三个点阵变体、模板归一化和模板指纹。
- `vertical_ocr/validation.py`：PNG 签名/格式/固定分辨率/`L` 模式解码，真值边框、栏区间、同栏序号连续性、整数关键点和包含关系校验。
- `vertical_ocr/config.py`：配置、命令流校验、参数指纹。
- `vertical_ocr/exchange.py`：v1 定宽行、v2 JSON、v3 键值文本解析/序列化/互转，显式 `MISSING` 和旁路元数据。
- `vertical_ocr/ledger.py`：JSONL 账本、旁路存储、原子 JSON 写。
- `vertical_ocr/vision.py`：竖直栏投影、栏内投影候选、过滤原因、粘连块 DP 分割、墨迹重心/留白边框回归。
- `vertical_ocr/matching.py`：模板缩放归一化、定点整数权重融合、置信度量化步长/残差、预算成本。
- `vertical_ocr/reading.py`：关键点权威裁决、RTL/TTB 几何串联、跨栏/错序/缺号冲突。
- `vertical_ocr/cache.py`：内容寻址缓存、scope 旧键、命中/重算/失效事件。
- `vertical_ocr/scheduler.py`：稳定顺序预算准入、待办优先、线程池并发。
- `vertical_ocr/snapshot.py`：阶段产品临时目录写入、版本目录整体提交、`current.json` 指针和恢复摘要。
- `vertical_ocr/injection.py`：五类真实输入故障隔离套件。
- `vertical_ocr/diagnostics.py`：25 项一键验收。
- `tests/test_core.py`、`tests/test_pipeline.py`：解析、裁决、DP、预算、校验和端到端缓存测试。

## v1 定宽格式

字段宽度依次为：

```text
image_no 4 | char_no 4 | x1 5 | y1 5 | x2 5 | y2 5 | confidence_permille 5 | order_code 4
```

总宽 37。空白字段是该版本无法表达或旁路尚未恢复的显式缺失，不会按零处理。字段过短、整数含非法字符、v2/v3 缺键均抛 `E_FIELD_TRUNCATED`，错误对象带 `version` 和 `field`。

## 降级旁路

v3 → v1 使用半舍五入恢复千分比置信度，坐标按非负整数截断并保存：

- 原始浮点 bbox 与四个截断残差；
- 原始浮点置信度和舍入残差；
- 精度/完整性标签；
- 关键点整数坐标、命名关键点版本、来源标注版本、字符和阅读顺序。

旁路位于 `<root>/ledger/sidecar_store.json`，逐条降级/恢复写入 `<root>/ledger/conversion_ledger.jsonl`。相同业务键使用确定性 `SC-<sha256>` ID，v1 → v3 可从旁路还原浮点原值。

## 确定性视觉链路

1. `gray < binarize_threshold` 得墨迹。
2. x 方向投影 + `column_projection_window` 平滑，密度谷/连续墨迹 run 输出右到左栏。
3. 栏内 y 投影 + `projection_window` 平滑，合并笔画 run 为字符候选。
4. 低于 `char_min_height` 记录 `height_below_min`；宽高比越界记录 `width_ratio_out_of_bounds`。
5. 高于 `char_max_height` 的候选进入 DP：代价为 `1 - valley_depth + split_penalty`；总成本相同时选择字典序更靠上的切点集合。
6. 仅在所属栏 x 区间和候选 y 区间内统计墨迹重心/留白并回归整数 bbox，不跨栏借像素。

## 定点匹配与缓存

观测块先在回归 bbox 内取墨迹、保持比例补方并归一到 16×16；每个字模有基准、加粗、微移三个变体。融合权重乘 `2^fraction_bits` 转整数，按变体累加 `similarity_points * fixed_weight`，再除以权重和与 `256` 得置信度。最高分识别；精确并列时按 Unicode 码位升序。

置信度按 `quantization_level` 转定点整数，记录：

```text
confidence_fixed, quantization_step = 1 / quantization_level, quantization_residual
```

缓存键覆盖栏检测、投影切分、候选/边框回归和模板匹配。栏/字符阶段使用区域内容指纹，因此新增栏不会改变未动栏的区域键；整图指纹用于栏检测和追踪。修改阈值、高度约束、分割罚项、匹配权重/度量、量化档位、模板版本或替换图像时，只有依赖键变化的 scope 重算。普通非粘连块的键不包含 `split_penalty`，所以纯分割罚项变化不会重算它们；高粘连候选的键包含分割参数。

## 阅读顺序

权威关键点排序键固定为：

```text
version_no DESC, annotated_ms DESC, annotator ASC
```

最终顺序优先级为：

```text
权威关键点几何位置 > bbox 中心 > 已校验真值 reading_no > 稳定字符 ID
```

冲突类型为 `cross_column`、`wrong_order`、`missing_number`，均写入逐图报告和快照。预算待办会形成显式缺号，下一批补齐后消除。

## 预算、并发与快照

单字符成本：

```text
(bbox_width * bbox_height) * template_variant_count
```

调度器按 `(image_name, rtl_column, char_index)` 稳定排序。预算不足时，当前字符及其后项目进入 `E_MATCH_BUDGET_PENDING`；待办队列写入 `ledger/budget_queue.json`，下一批先消费待办。缓存命中不重复扣模板点数。

栏检测和匹配使用 `ThreadPoolExecutor(max_workers=min(config.max_workers, workload))`，不会超过配置并发。

快照包含 `01_load_validation`、`02_column_detection`、`03_character_matching`、`04_reading_assembly` 和 `budget_queue`。提交时先写 `snapshots/.staging-*`，再原子改名到 `snapshots/versions/...` 并更新 `current.json`。清单记录全局版本、配置哈希、模板版本、关键点指纹、完成图像、累计预算和待办。版本或配置哈希不匹配时报告 `E_SNAPSHOT_STALE` 并重建。

`--kill-before-commit` 在临时阶段产品写完、正式改名和指针更新前发送 `SIGKILL`。因此强杀后 `current.json` 仍指向上一个已提交快照；新版本启动会明确报告 stale，而不是承诺生成被杀进程的退出报告。

## 诊断报告

主报告：`<root>/reports/last_report.json`。

包含三版本回执数、逐图字符和阅读序号、冲突计数、缓存命中/重算/失效事件、待办清单、预算、业务码计数、失败清单、快照恢复范围和 `elapsed_ms` 单调时钟耗时。

注入报告：`artifacts/acceptance-run/injections/injection_report.json`，逐项给出 `INJ-PNG-001` 到 `INJ-FONT-005` 的命中状态和映射码。
