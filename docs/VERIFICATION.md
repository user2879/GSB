# 实际验证结果

## 环境

- OS：Linux（容器内 Ubuntu，等价于 Windows 10/11 + Python 3.11 的纯 Python 路径，
  无平台相关 API；原子替换使用标准库 `os.replace`，在 Windows NTFS 上同样为原子操作）
- Python：3.11.0rc1（虚拟环境 `.venv`）
- 依赖：numpy 2.4.6、jsonschema 4.26.0（仅这两个第三方依赖，无 GPU、无运行期网络访问）

> 注：题目目标环境为 Windows + Python 3.11/3.12。本机系统 Python 缺少
> `ensurepip`，因此先用官方 `get-pip.py` 在用户目录引导 pip，再以
> `python -m venv --without-pip .venv` 创建虚拟环境并在其中安装 pip、
> numpy、jsonschema。复现步骤见 `docs/SETUP.md`。

## 单元与集成测试

命令：`.venv/bin/python -m unittest discover -s tests`

结果：**27 个测试全部通过**，覆盖：

- 加载与 jsonschema 校验、错误码与失败清单顺序（9 个）；
- 确定性采样、float64 打分稳定性、串并行一致、平局决断（5 个）；
- Jaccard、边界 merge/split、代表样本选择（4 个）；
- 五个批次增量 vs 全量逐字段一致、失效/复用、续跑幂等、配置失配拒绝（5 个）；
- 四阶段故障注入崩溃恢复、恢复点批次、清单篡改检测（3 个）；
- 并发度 1/2/4/8 输出不变、产物与失败清单顺序（2 个，合计 27）。

## 一键验收

命令：`.venv/bin/python scripts/acceptance_check.py`

实际输出摘要：

```
=== 1. 确定性：两次全量构造产物逐字节一致 ===
PASS: 全量构造可复现，三阶段产物完全一致
=== 2. 分批增量 vs 全量重建逐字段一致 ===
  batch 1: 候选 42 -> 保留 8，差异字段 0
  batch 2: 候选 42 -> 保留 8，差异字段 0
  batch 3: 候选 42 -> 保留 7，差异字段 0
  batch 4: 候选 42 -> 保留 7，差异字段 0
  batch 5: 候选 42 -> 保留 7，差异字段 0
PASS: 5 个批次增量结果与全量重建逐字段一致
=== 3. 精确失效与复用 ===
PASS: 无变化重跑时重算 0 条，复用 42 条
=== 4. 快照恢复与强制终止 ===
PASS: 强制终止后从批次 2 恢复，临时目录无残留，恢复结果与全量一致
=== 5. 错误码 E_CONFIG_MISMATCH ===
PASS: 基础输入指纹变化时拒绝续跑并返回 E_CONFIG_MISMATCH
=== 6. 加载错误码不中断其余条目 ===
PASS: 坏条目进入失败清单（1 条），其余条目正常加载
=== 7. 并发度不变性 ===
PASS: max_workers=1/4/8 下采样、打分、去重产物完全一致
全部验收通过
```

## 可复现性

两次独立全量构造对以下文件做字节级比较（`cmp`）全部相同：

- `candidates.json`、`scores.json`、`dedup_groups.json`、`report.json`
- `snapshot/manifest.json` 与 `snapshot/stages/{sampling,scoring,dedup}.json`

## 样本数据

`scripts/generate_samples.py` 是无随机源的固定种子程序，从受控词表
（12 个 subject、8 个 action、6 个 format/language、10 个 topic、6 个 tone）
生成 `samples/basic/` 下六类输入，重复执行逐字节一致。分批计划覆盖模板
替换、模板删除、槽位来源增删、目标数调整、模板/槽位权重调整、打分权重与
基准分调整、去重阈值调整、边界规则切换共全部操作类型。
