# 候选框交换与跨帧匹配验证工具

这是一个完全离线、无真实检测模型权重的本地工具，用于验证 v1 定宽文本与 v2 JSON 两种候选框交换格式之间的互操作、精度损失、跨帧轨迹状态、增量失效和阶段快照恢复。

项目包含固定种子合成数据生成器、双版本解析/统一表示、双向转换与旁路残差、类别内 IoU 轨迹串联、内容指纹缓存、原子阶段快照、真实故障注入、并发执行和完整 JSON/Markdown 验收报告。

## 环境

- Python 3.10+
- `numpy`
- `Pillow`

当前验证环境为 Python 3.11.0rc1、numpy 1.26.4、Pillow 11.0.0。工具不访问网络，也不需要模型权重。离线安装可使用本地 wheel：

```bash
python -m pip install --no-index --find-links ./wheels -r requirements.txt
```

Windows 上可将 `python` 替换为 `py -3`。

## 快速开始

```bash
# 1. 生成固定种子样本
python -m candidate_exchange generate --out data/sample --seed 20261122 --force

# 2. 运行完整验收（会执行单/多线程、6 步编辑、注入、强杀恢复等）
python -m candidate_exchange validate --data data/sample --out reports

# 3. 单次运行
python -m candidate_exchange run \
  --data data/sample \
  --out runs/demo \
  --snapshots runs/demo_snapshots \
  --workers 4 \
  --snapshot-interval 2

# 4. 生成真实注入样本并运行
python -m candidate_exchange inject \
  --data data/sample \
  --out data/injected \
  --injections data/sample/injections.json \
  --force

python -m candidate_exchange run \
  --data data/injected \
  --out runs/injected \
  --snapshots runs/injected_snapshots
```

也可以直接使用脚本：

```bash
python scripts/generate_samples.py --out data/sample --force
python scripts/run_validation.py --data data/sample --out reports
```

## 实际验收结果

当前工作区已生成并验证：

- 3 个序列、21 帧、99 个候选框
- 18/18 项自动检查通过
- 单线程与 4 线程最终帧、轨迹、失败清单、偏差结果逐字段一致
- 增量编辑后最终状态与空白全量重建一致，最终 cache-key 账本投影一致
- v2→v1 坐标最大偏差 `0.4000000000000057`，得分最大偏差 `0.0002999999999999947`
- 7/7 注入项命中，业务码分别为 `E_IMAGE_INVALID`、`E_FIELD_TRUNCATED`、`E_COORD_INVALID`、`E_NON_FINITE`、`E_CLASS_UNKNOWN`、`E_MATCH_INVALID`、`E_SERIALIZE_INVALID`
- 独立进程在 matching 阶段快照后以退出码 `75` 终止，未写最终报告；重启恢复点版本为 `4`，最终提交版本为 `9`

详细结果见 `reports/validation_report.json` 与 `reports/validation_summary.md`。

## 进程退出码与业务码

进程退出码只表达运行状态：

| 退出码 | 含义 |
| ---: | --- |
| `0` | 流程完成；即使报告中有业务失败也返回 0 |
| `2` | 配置、IO 或命令行输入错误 |
| `3` | 未预期内部异常 |
| `75` | 模拟强制终止；该次运行不写最终退出报告 |

`E_xxx` 业务错误码写入 JSON 报告或 stderr，不映射为进程退出码。主要业务码：

- `E_FIELD_TRUNCATED`：v1 定宽字段不足/错位，或必要字段缺失
- `E_VERSION_MISMATCH`：v1/v2 的候选数量、类别、坐标或得分不能配对
- `E_COORD_INVALID`：坐标非单调或越出图像
- `E_SCORE_INVALID`：得分不在 `[0, 1]`
- `E_CLASS_UNKNOWN`：类别码不在码表
- `E_NON_FINITE`：权重等数值非有限
- `E_IMAGE_INVALID`：PNG 截断或图像不可验证
- `E_MATCH_INVALID` / `E_SERIALIZE_INVALID`：匹配或序列化阶段故障

## v1 定宽格式

文件：`candidates_v1.txt`。每行宽 `75` 字节，注释行以 `#` 开头：

| 字段 | 偏移 | 宽度 | 语义 |
| --- | ---: | ---: | --- |
| `series_name` | 0 | 16 | 左对齐序列名 |
| `frame_index` | 16 | 8 | 十进制帧号 |
| `class_code` | 24 | 6 | 整数类别码 |
| `x1/y1/x2/y2` | 30/38/46/54 | 各 8 | 整数像素坐标 |
| `score_pm` | 62 | 5 | 千分比得分，0..1000 |
| `missing` | 67 | 8 | 前四位为 `C/X/Y/S` 缺失标记，如 `C-------` |

任一字段不足、额外尾部数据或必要字段为空均输出 `E_FIELD_TRUNCATED`。

## v2 JSON 格式

文件：`candidates_v2.json`。根对象包含 `format_version: "v2"` 和 flat `frames` 列表。每个候选包含：

```json
{
  "class_code": 1,
  "class_name": "pedestrian",
  "x1": 30.2,
  "y1": 25.4,
  "x2": 60.7,
  "y2": 67.3,
  "score": 0.8241,
  "precision_label": "f32",
  "extra": {}
}
```

解析器也接受按序列分组的 `series[].frames[]` 形式。统一表示把坐标归一为配置指定小数位宽的定点整数，把得分归一为微单位整数；报告中的框为整数像素框，得分为千分比。

## 双向转换

- v1→v2：整数坐标和千分比得分精确转浮点，`precision_label` 为 `v1_exact`；类别码通过 `class_map` 映射为名称。
- v2→v1：使用 `ROUND_HALF_UP` 还原整数坐标和千分比；每个候选的四轴坐标残差、得分残差、精度标签和 `extra` 元数据写入独立 sidecar。
- 两个方向都会真实写出文件并重新解析；报告给出坐标/得分绝对偏差直方图、最大值、类别还原统计。v1→v2 应为零偏差；v2→v1 的残差用于证明精度损失被明确保留。

## 匹配与候选池

- 只在同一序列、同一类别内按 IoU 贪心匹配。
- 匹配成功的轨迹追加帧、更新中心与逐帧速度估计。
- `max_gap_frames` 用帧号差判断，超过则封闭。
- 活跃池超过 `candidate_pool_capacity` 时，按最后得分升序选出逐出对象；同分时选择坐标字典序尾部，即保留 `score_desc_then_coord_lex` 顺序。
- 被逐出轨迹在允许空档内再次命中时重开，并写 `reentered_pool=true` 与 `eviction_count`。
- 轨迹 ID 由序列、类别、起始帧和候选规范内容确定性派生，不依赖线程完成顺序。

## 增量缓存与一致性

每帧缓存键由以下内容派生：

- 图像字节内容指纹
- v1/v2 候选规范内容
- 检测参数指纹
- 当前交换版本/格式指纹
- 该帧旁路元数据指纹
- 定点小数位宽

轨迹缓存键依赖轨迹首末帧之间所有帧缓存键，以及 IoU、最大空档、逐出规则和小数位宽等匹配配置指纹。因此：

- 切换 v1/v2：格式指纹变化，全部帧失效
- 修改 IoU：帧复用，轨迹全部重算
- 替换单帧图像或单帧候选：仅该帧及依赖它的轨迹重算
- 修改候选池容量：检测参数指纹变化，帧和轨迹均重算

编辑序列的最终帧、候选顺序、轨迹 ID/构成、框、得分、偏差、失败清单与空白全量重建一致。`reused/recomputed` 是运行操作标签，空白全量与增量按定义不同；验收额外比较最终 cache-key 依赖账本投影，确保可审计的失效和复用依赖相同。逐条 ledger、原因和命中配置均写入报告。

## 阶段快照

快照根目录结构：

```text
snapshots/
  CURRENT.json
  versions/
    000000000001/
      manifest.json
      stages/detection.json
      stages/matching.json
      stages/serialization.json
```

每次提交先在 `.tmp-*` 写完整目录，fsync 后原子重命名；manifest 记录全局版本、父版本、配置哈希、格式指纹、旁路元数据哈希、输入指纹、各阶段版本和每个产物 SHA-256。`CURRENT.json` 只在目录提交后原子替换。

恢复时扫描所有带有效 manifest 且产物哈希匹配的版本，忽略半写临时目录。强制终止后重启只使用已完整替换的检测/匹配快照继续，输出恢复点版本；终止进程不会补写终止前最终报告。

## 故障注入

`injections.json` 是模板；`inject` 命令复制工作区后施加真实输入变化，并写入 `active_injections.json`：

- 截断 PNG 字节
- 清空 v1 定宽类别字段并设置缺失位
- 同时越界 v1/v2 坐标
- 在逐帧 detection override 中写入非有限融合权重
- 同时替换为码表外类别码
- 匹配阶段真实触发指定帧异常
- 序列化阶段真实触发指定帧异常

每个注入项编号、期望码、实际码和阶段写入报告。失败帧不进入轨迹，其余帧继续处理。

## 测试

```bash
python -m unittest discover -v
```

测试覆盖解析截断、双向转换残差、增量/全量一致、单/多线程一致、注入隔离、快照恢复与 CLI 退出码。

## 目录

```text
candidate_exchange/   核心包与 CLI
data/sample/          固定种子样本、双版本候选文件、配置和注入模板
reports/              已验证报告及逐步骤审计产物
scripts/              样本生成与验收入口
tests/                unittest 测试
docs/DESIGN.md        详细设计说明
```
