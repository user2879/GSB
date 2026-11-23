# 竖排古籍版面识别（本地确定性合成图像）

这是一个纯 Python 的竖排古籍版面识别应用。它从本地固定分辨率 PNG 与真值 JSON 出发，完成栏检测、栏内字符投影切分、边框回归、粘连块最优分割、24×24 点阵多模板定点匹配和逐字符置信度计算，并以内容寻址的阶段缓存与代际快照支持可复现的增量重算、预算待办和故障注入隔离。

项目不访问网络、不使用 Docker，样本由内置字模和固定种子生成。当前验证环境为 CPython 3.11、NumPy 1.26.4、Pillow 10.3.0、pytest 8.2.0；完整结果见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 快速开始

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ancient_vlayout generate --samples samples
.venv\Scripts\python -m ancient_vlayout run `
  --samples samples `
  --state-dir .state\normal `
  --output artifacts
.venv\Scripts\python -m pytest
```

Linux/macOS 将 `.venv\Scripts\python` 换成 `.venv/bin/python` 即可。

故障注入模式使用单独的状态目录，避免与正常运行互相污染：

```powershell
.venv\Scripts\python -m ancient_vlayout run `
  --samples samples `
  --state-dir .state\injections `
  --output artifacts\injections `
  --injections
```

单批运行（保留预算待办）加 `--once`；默认会持续批处理到待办清空或无可进展。

## 目录

```text
ancient_vlayout/
  config.py       配置归一化、哈希、阶段参数指纹
  loader.py       PNG 区块/CRC/尺寸/模式校验、灰度解码、truth 校验
  imaging.py      二值化、投影、2x2 开运算、边框回归
  pipeline.py     栏检测、候选段、DP 粘连分割、父块追溯
  matching.py     24x24 点阵、多模板定点融合、置信度量化
  engine.py       并发调度、增量缓存、预算账本、快照提交
  snapshot.py     代际目录、原子 current 指针、恢复扫描
  injection.py    四类独立故障注入
  samples.py      固定种子样本与 truth/config/injections 生成
tests/            19 个本地验收用例
samples/          已生成的 4 页 PNG、truth、config、injections
artifacts/        实际运行结果、报告与诊断
docs/             设计与验证说明
```

## 输入契约

每张版面图为 `600x720`、PNG、`L/RGB/RGBA` 像素模式，解码后统一为 `uint8` 灰度。真值文件优先命名为 `<stem>.truth.json`，也可为 `<stem>.json`：

```json
{
  "schema_version": 1,
  "image": "page_01.png",
  "width": 600,
  "height": 720,
  "columns": [
    {
      "index": 0,
      "bbox": [474, 115, 526, 494],
      "chars": [
        {"reading_index": 0, "bbox": [479, 119, 523, 173], "char": "口"}
      ]
    }
  ]
}
```

校验包括整数半开区间边框、图像/栏内范围、栏索引与栏内阅读序号连续唯一、字符内容非空。失败图片按文件名字典序进入 `failures`，错误码为 `E_IMAGE_INVALID` 或 `E_TRUTH_INVALID`；注入目标还可能产生 `E_CONFIG_INVALID`。

## 配置

`samples/config.json` 的主要字段：

| 字段 | 作用 |
| --- | --- |
| `binarization_threshold` | 灰度二值化阈值 |
| `projection_window` | 投影闭塞小缝隙的窗口 |
| `min_char_height` / `max_char_height` | 候选段与粘连块高度约束 |
| `split_penalty` / `split_valley_weight` | DP 分割罚项与投影谷权重 |
| `match_metric` | `overlap`（带 1 像素容差的对称 F1）或严格 `l1` |
| `fusion_weights` | 三个点阵变体的定点融合权重 |
| `fixed_point_bits` | 定点小数位宽，真实参与分数/置信度计算 |
| `confidence_quantization_steps` | 置信度量化档位数 |
| `ma_point_budget` | 单批模板比对点数预算 |
| `thread_pool_workers` | 栏检测与字符匹配线程池上限 |
| `snapshot_interval` | 快照/检查点间隔元数据，批末仍整体原子提交 |
| `injection_manifest` | 样本目录内独立注入清单 |

## 结果与报告

`run` 写出五个文件：

- `recognition.json`：稳定排序的逐图、逐栏、逐字符边框、识别字符、置信度、栏引用、候选段、父块和待办项。
- `report.json`：快照全局版本、配置/模板/各阶段版本、缓存命中/重算/失效原因、预算账本、并发上限与任务排序。
- `diagnostics.json`：失败清单、短段丢弃依据、DP 并列/不可行切点、缺失模板与注入命中。
- `run_summary.json`：便于人工检查的计数摘要。
- `snapshot_manifest.json`：快照全局版本、配置哈希、字模模板版本、代际目录与各阶段产物版本。

每个匹配结果同时保留 `confidence_raw_fixed`、`confidence_fixed`、`confidence_value`、`confidence_digits`、`confidence_quantum` 与量化档位。切换 `fixed_point_bits` 会改变定点整数、量化后数值与保留位数。

## 增量语义

缓存键由阶段相关的图像内容指纹、参数指纹与上游键组成；匹配阶段额外绑定字模模板版本与可用模板集合。修改匹配度量只重算匹配，修改高度只重算切分/边框/匹配，修改分割罚项只重算边框/匹配，单图替换只影响该图及其下游。未失效产物保留原键并记为命中。

从同一编辑序列稳定后的增量结果与空白全量重建，在逐栏区间、逐字符边框、识别字符、置信度、失败/待办顺序上逐字段一致。单次运行的 `cache.hits`/`cache.recomputed` 本身依赖运行前状态，因此验证时与相同编辑序列的独立重放比较，而不是与冷启动全量的缓存事件比较；两类清单均固定排序。

更多算法与不变量见 [docs/DESIGN.md](docs/DESIGN.md)。
