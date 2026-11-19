# detbox — 本地合成图像候选框交换与匹配验证工具

面向本地确定性合成 PNG 序列，验证候选框在 **v1 定宽文本** 与 **v2 JSON** 两种交换
版本之间的互操作、精度损失、跨帧轨迹一致性，以及版本切换下增量运行与空白全量
重建的逐字段等价性。不依赖网络数据、真实账号或 GPU；不使用容器。

## 1. 环境准备（Windows / Linux 均可，Python ≥ 3.10）

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pillow pytest
python -m pip install -e .
```

Linux（本机实测 Python 3.10，系统缺 `ensurepip` 时先用 `get-pip.py` 引导）：

```bash
python3.10 -m venv --without-pip .venv
.venv/bin/python get-pip.py            # 仅在 ensurepip 不可用时需要
.venv/bin/python -m pip install numpy pillow pytest
.venv/bin/python -m pip install -e .
```

实测版本：Python 3.10.12、numpy 2.2.6、Pillow 12.3.0、pytest 9.2.0。

## 2. 目录与模块

```
src/detbox/
  errors.py        业务码与异常类型（E_CAND_FORMAT / E_CONFIG_MISMATCH / ...）
  models.py        UIR 候选/帧/轨迹数据类、码表、v1 定宽列布局、四舍五入
  util.py          内容哈希、原子写、目录整体替换、有限数校验
  config.py        检测参数 / 匹配配置 / 注入配置加载与校验
  samplegen.py     固定种子 PNG 序列、真值框、配置与畸形样本生成
  exchange.py      v1/v2 解析、序列化、双向转换与 v2→v1 旁路元数据
  detect.py        滑窗检测、颜色/边缘特征、融合得分、NMS、候选池逐出
  match.py         类别内 IoU 贪心匹配、轨迹串联/空档封闭、匹配池逐出回放
  fingerprint.py   帧内容/参数/格式/旁路指纹与三级缓存键
  injection.py     真实输入故障注入并映射业务码
  store.py         三阶段快照、原子替换、manifest、信号处理与恢复清理
  pipeline.py      检测→匹配→序列化流水线、增量复用、失败隔离
  report.py        跨版本验证报告
  cli.py           命令行入口
scripts/           数据演示、交换样本生成、验收驱动
tests/             pytest 测试（30 个用例）
```

## 3. 两种交换格式

- **v1（定宽文本，每行固定 46 字符 + 换行）**：序列名(10) 帧序号(5) 整数类别码(3)
  左上 x1/y1、右下 x2/y2（各 5）千分比得分(7) 0/1 缺失位(1)。
- **v2（JSON）**：逐序列嵌套帧，浮点 box、`[0,1]` 浮点得分、命名类别、精度标签
  与可选 `ref_score` 高精度参考得分。
- v1→v2：整数坐标/千分比得分精确换算为浮点，标签 `upgraded-from-v1`，码表映射命名类别。
- v2→v1：坐标四舍五入（half-away-from-zero）、得分还原千分比；截断残差、被丢弃
  精度标签、`ref_score`、类别映射写入旁路 sidecar，sidecar 参与匹配/序列化指纹与快照。

## 4. 常用命令

```bash
# 生成数据集（图像/真值/检测参数/匹配配置/注入配置/畸形样本）
detbox generate --dataset data/sample

# 增量运行（v1 或 v2），输出原子写报告
detbox run --dataset data/sample --workdir work/run1 \
  --detection-params data/sample/detection_params.json \
  --match-config data/sample/match_config.json \
  --injections data/sample/injections.json \
  --format v1 --report work/run1/report.json

# 从空白工作区一次性全量重建
detbox rebuild ...   # 参数同 run

# 续跑判定（不一致输出 E_CONFIG_MISMATCH，退出码 2，拒绝续跑）
detbox check --dataset data/sample --workdir work/run1 \
  --detection-params ... --match-config ...

# 施加一次受控编辑（写 edit_journal.jsonl 并修改输入）
detbox edit --dataset data/sample --workdir work/run1 --edit-file edit.json

# 单文件双向转换
detbox convert --dataset data/sample \
  --input in.txt --input-format v1 --output out.json --output-format v2
```

便捷脚本：`scripts/run_demo.py`（双格式端到端）、`scripts/make_exchange_samples.py`
（交换样本与 sidecar）、`scripts/acceptance_check.py`（增量/全量等价验收）。

## 5. 增量失效与一致性设计

- 检测缓存键 = 帧像素内容指纹 + 检测参数指纹 + 匹配配置指纹（内容寻址；不含格式
  指纹，因此切换 v1/v2 **不会重复执行检测**）。
- 轨迹缓存键 = 该序列全部帧候选指纹 + 检测/匹配/格式/旁路指纹；序列化键再聚合所有
  轨迹指纹。单帧图像/候选变化只失效该帧→所属序列轨迹→序列化下游。
- 版本切换通过格式版本指纹识别（而非逐文件覆盖）；融合权重、IoU 阈值、候选池容量
  变化分别经参数指纹使对应阶段失效，未失效产物直接复用。
- 三阶段产物写入 `snapshot/<stage>` 的暂存目录后 `os.replace` 整体提交；manifest
  记录全局/快照版本号、配置哈希、格式与旁路指纹、各阶段版本。启动清理上次强杀残留
  的 `.staging-*`/`.old-*`，仅依据完整替换产物恢复；不承诺生成退出报告。
- 收到 SIGINT/SIGTERM（Windows 为 SIGBREAK）时置位，完成当前帧产物提交后以退出码 0 结束。

## 6. 异常注入与业务码

独立 `injections.json` 按 序列/帧/阶段/故障类型 注入真实输入故障，注入项编号进入
报告命中清单并映射业务码：

| 故障 | 阶段 | 业务码 |
| --- | --- | --- |
| 截断 PNG 字节 | detect | `E_IMAGE_TRUNCATED` |
| 非有限融合权重 | detect | `E_WEIGHT_NONFINITE` |
| 定宽字段缺失 / 坐标越界 | serialize | `E_CAND_FORMAT` |
| 类别码不在码表 | match/serialize | `E_CLASS_UNKNOWN` |
| 候选损坏 | match | `E_CAND_FORMAT` |
| 外部状态/配置不一致 | 启动恢复 | `E_CONFIG_MISMATCH` |

单帧异常记录 序列/帧/阶段/异常类型/业务码 到失败清单并继续其余序列；失败帧不进入
轨迹与序列化输出。

## 7. 验证结果（本机实测）

```bash
.venv/bin/python -m pytest -q          # 30 passed
.venv/bin/python scripts/acceptance_check.py
```

验收报告 `work/acceptance/acceptance_report.json` 关键结论：

- `incremental_full_rebuild_equal = true`：编辑序列（v1/v2 切换、单帧候选替换、
  图像替换、IoU 调整、容量变更）后增量结果与空白全量重建在整数框、千分比得分、
  类别码、轨迹归属上逐项一致。
- 6 个注入项全部命中并映射到预期业务码；6 个故障帧被隔离且不进入轨迹。
- v2→v1→v2 坐标偏差直方图全部落在 0 桶（检测框坐标本为整数）；最大得分偏差
  约 5e-4（千分比量化），类别映射全部可还原。
- 重复运行命中缓存、`recomputed=[]`；`detbox check` 在同输入下 `resume=true`，
  外部篡改后返回 `E_CONFIG_MISMATCH`。
