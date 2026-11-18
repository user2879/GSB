# 跨模态对齐与检索管线技术说明

面向本地确定性图文样本的离线处理管线，仅依赖 Python 3.10+ 标准库与
NumPy。所有缓存键、快照版本号、报告内容均由输入内容与配置派生，不含
时间戳、机器名或随机盐值，因此结果可复现、可审计。

## 1. 目录与模块

| 模块 | 职责 |
| --- | --- |
| `xmodal/png_codec.py` | 纯标准库 PNG 编解码（五种行过滤器、CRC/容器头/尺寸校验） |
| `xmodal/samplegen.py` | 固定种子 `RandomState` 生成几何图元+噪声 PNG、区域真值、结构化文本 |
| `xmodal/features.py` | 样本加载校验、分块统计、文本哈希特征、统一 `FeatureRecord` |
| `xmodal/formats.py` | v1 紧凑二进制包与 v2 JSON 包读写器、统一中间表示互转 |
| `xmodal/alignment.py` | 加权余弦、减最大值稳定 softmax、对齐得分、检索与同分字典序决断 |
| `xmodal/convert.py` | v1/v2 互转、精度标签、截断残差 sidecar、`E_FORMAT_LOSSY` 审计 |
| `xmodal/cache.py` | 内容/配置/版本指纹派生的缓存键与精确重算范围判定 |
| `xmodal/snapshot.py` | 阶段临时目录整体替换、清单、信号门、恢复点判定 |
| `xmodal/injection.py` | 六类真实输入故障注入、注入编号与业务码映射 |
| `xmodal/pipeline.py` | 五阶段编排、线程池、异常隔离、增量复用、快照提交 |
| `xmodal/report.py` | 确定性 JSON 验证报告 |
| `xmodal/cli.py` | `generate` / `run` / `recovery` 命令行入口 |

## 2. 特征抽取（业务要求 1）

- 图像解码为 uint8 后按 `block_size` 分块，每块输出
  `3 个 RGB 均值 + 1 个 Sobel 梯度强度 + B 个梯度主方向直方图`，
  再按 `quant_levels` 均匀量化，维度为 `4 + orientation_bins`。
- 文本把实体、属性、断言 token 化，每个 token 用 sha256 派生槽位与正负
  号，累加到 `text_dim` 向量后做 L2 归一化；不使用 Python 随机哈希。
- PNG 头、尺寸、JSON 结构、断言字段任一不合规即抛 `E_SAMPLE_INVALID`，
  失败样本进入失败清单且不阻断其他样本。

## 3. 交换格式与统一表示（业务要求 2、4）

- v1：48 字节头部（magic/版本/各维度/偏移表位置）+ 6 项 uint32 偏移表 +
  样本名、float32 图像块矩阵、float32 文本向量、权重段、精度标签 JSON。
- v2：`实体/块名 -> 浮点数组` 字典，附带 `precision_labels` 与对齐权重。
- 两者都解析为同一 `FeatureRecord`；同记录写 v1/v2 再读回，图像块与文本
  向量逐元素一致，权重差异仅为 float32 舍入并由精度标签解释。
- v2->v1 时四舍五入写 float32，残差写入 `*.sidecar.json`；非有限值或超出
  float32 上限输出 `E_FORMAT_LOSSY` 并记录具体键（如 `text.vector[0]`）。

## 4. 对齐打分与检索（业务要求 3）

- 图像块按均值/梯度/方向切片，文本向量按槽位确定性折叠到对应维度。
- 每组件计算块级余弦，做减最大值的稳定 softmax 后加权汇总；最终得分
  裁剪到 `[-1, 1]` 并保留 9 位小数，保证跨格式、跨并发度稳定。
- 检索候选按得分降序、同分按样本名字典序排序，返回前 K；每条文本查询的
  目标命中取同名样本图像得分，目标得分 `<` 阈值时标记
  `E_RETRIEVAL_LOW_SCORE`，全量候选清单保留在报告 `audit_candidates`。

## 5. 增量失效与一致性（业务要求 5、6）

- 缓存键 = 上游内容指纹 + 配置指纹 + 格式版本号的 sha256 派生值。
- 样本替换只失效该样本全链路；权重变更只失效 align/retrieve；分块尺寸
  变更失效 extract 及其下游；格式切换只失效 convert。
- 未失效产物直接从已提交快照读取，不执行对应计算；报告 `invalidation`
  与 `reuse` 记录逐样本清单，`reuse_stats` 汇总数量。
- 线程池按 `(样本名, 阶段名)` 派生稳定排序键；workers=1/4/8 的
  `content_hash` 完全一致；增量运行与空白全量运行的业务字段逐字段一致。

## 6. 快照与中断恢复（业务要求 7）

- 阶段产物先写 `stages/<stage>.tmp/`，再通过同文件系统 `os.replace`
  整体替换为 `stages/<stage>/`；复用样本旧产物在提交前并入临时目录，
  快照仍保持单次原子 rename。
- `manifest.json` 记录全局版本号、配置哈希、规范哈希、已完成/有效样本、
  各阶段产物版本与全部缓存键。
- SIGTERM/SIGINT 在样本边界触发优雅退出：先完成当前阶段原子提交，再以
  退出码 0 结束；SIGKILL 后重启只承认完整替换目录，清理 `.tmp` 并依据
  清单与实际阶段目录取交集恢复，恢复点写入报告（不承诺退出报告）。

## 7. 异常注入与报告（业务要求 8）

| 注入编号 | 故障 | 阶段 | 业务码 |
| --- | --- | --- | --- |
| INJ-001 | 截断 PNG 字节 | validate | E_SAMPLE_INVALID |
| INJ-002 | 文本断言字段缺失 | validate | E_SAMPLE_INVALID |
| INJ-003 | v1 偏移表越界 | convert | E_SAMPLE_INVALID |
| INJ-004 | float32 非有限值 | convert | E_SAMPLE_INVALID |
| INJ-005 | 阈值恰好等于边界得分 | retrieve | E_RETRIEVAL_LOW_SCORE |
| INJ-006 | 未知格式版本号 | convert | E_SAMPLE_INVALID |

报告 `report.json` 含逐查询检索结果（含审计候选）、对齐得分分布、跨格式
偏差摘要、失效/复用统计、快照版本号、恢复点、注入命中与失败清单。

## 8. 运行方式

```bash
python3.10 -m xmodal.cli generate --input demo_output/input --count 8
python3.10 -m xmodal.cli run --input demo_output/input \
  --workdir demo_output/work --config config/alignment.json \
  --spec config/format_spec.json [--injections config/injections.json] \
  [--format-version 1|2] [--workers N]
python3.10 -m xmodal.cli recovery --workdir demo_output/work
python3.10 -m unittest discover -s xmodal/tests -p "test_*.py"
bash scripts/demo.sh
```
