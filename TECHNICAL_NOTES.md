# 中文文本校对与数值误差诊断应用 — 技术说明

## 1. 概述

本应用面向中文文本校对场景，在**完全离线、确定性合成数据**上实现错别字、重复词、
术语不一致与疑似漏字四类问题的检测、定点化打分、编辑决策与修正文本输出，并对
float32 / float64 两个计算档位做逐级数值误差传播追踪。

- 不使用 Docker 或任何容器；不访问网络、真实站点、账号或外部模型。
- 仅依赖 `numpy` 与 `pyyaml`，Python ≥ 3.10。
- 全部随机数据由固定种子 `SEED=20261118` 生成，可重复复现。

## 2. 目录结构

```
proofread_app/
  errors.py        稳定错误码（E_EMPTY_TEXT/E_PARSE_INVALID/E_RULE_UNKNOWN/
                   E_CONFIG_INVALID/E_CONFIG_MISMATCH 等）与失败清单
  fingerprint.py   SHA-256 内容指纹、派生缓存键、稳定排序键
  models.py        样本/规则/候选/定点中间量/分数分解/审计/编辑记录数据模型
  config.py        YAML 配置加载与严格校验（n-gram 正整数、有限数、阈值范围）
  loader.py        JSONL 样本、规则资源、打分参数加载与全部合法性校验
  detection.py     n-gram 切窗、四类规则匹配、Jaccard 相似度、重叠消解与抑制留存
  scoring.py       双档位量化器、检测分/置信度、加权聚合、阈值与边界优先级决策
  precision.py     四环节差值、首次越界环节、决策翻转与排序矛盾归因（float64 差值）
  editing.py       replace/delete/insert 编辑落地与修正文本（右向左应用）
  cache.py         内容指纹缓存、规则引用索引、分阶段精确失效
  concurrency.py   线程池 + 稳定排序键，保证并发/串行逐字段一致
  snapshot.py      临时目录写入 + os.replace 整体替换、清单版本、恢复点
  signals.py       SIGINT/SIGTERM 可处理退出控制
  pipeline.py      五阶段编排、缓存键派生、双档位审计、快照提交
  report.py        诊断报告与修正文本原子写盘、双档位直方图
  app.py / cli.py  应用装配、五类交互命令、run/replay/recovery/clean 子命令
scripts/
  generate_data.py 固定种子生成样本/规则/打分参数/配置/异常注入样本
  signal_demo.py   优雅退出演示
tests/             21 个 unittest 用例（无第三方测试框架依赖）
config/            proofread.yaml、commands.jsonl、fault_injection.json、invalid_config.yaml
samples/           sentences.jsonl（26 条）、sentences_invalid.jsonl（异常注入）
resources/         rules.json、scoring.json
state/ output/     运行期快照/缓存与报告输出
```

## 3. 五阶段流水线与缓存键

每个样本依次经过：切分(segment) → 检测(detect) → 分项打分(score) →
聚合(aggregate) → 决策(decision)。

缓存键由**上游产物指纹 + 本阶段参数指纹**经 SHA-256 派生，任一上游内容或参数
变化都会改变键。交互命令的精确失效范围：

| 命令 | 失效阶段 | 复用阶段 |
| --- | --- | --- |
| `set_threshold` | detect/score/aggregate/decision | segment |
| `set_quant_levels` | score/aggregate/decision | segment/detect |
| `set_aggregation_weights` | aggregate/decision | segment/detect/score |
| `add/remove_confusion_pair` | 仅命中/引用该规则的样本全部下游 | 其余样本全部阶段 |
| `replace_sample` | 仅该样本全部阶段 | 其他样本全部阶段 |

规则引用索引 `rule_usage` 记录每条规则被哪些样本引用，未失效产物直接复用、
不重算；复用/重算清单写入报告 `cache` 字段。

## 4. 定点化与边界决策

- 量化器先将分数映射到 `quant_levels` 个档位，再按 `fixed_point_bits` 定点化，
  保留 `raw / quantized_int / scale / dequantized` 全部中间量。
- 总分 `= 检测量化值×detect 权重 + 置信度量化值×confidence 权重`。
- 总分严格大于阈值 → apply；严格小于 → skip；**恰好相等** → 按错误类型固定
  优先级决断并标注边界样本，同优先级按起始位置升序。
- 合成数据中 R003 在 float64 下总分精确等于阈值 `0.2876953125`，而 float32
  舍入后略低，形成真实的跨档位决策翻转（S006、S022）。

## 5. 误差传播追踪

对每条样本分别以配置档位（float32）与高一档（float64）完整计算四个环节，
差值统一在 float64 下计算并保留 9 位有效数字（≥6 位要求）：

- 输出每环节 `abs_diff / rel_diff / exceeds_tolerance` 与首次越界环节。
- 决策翻转（两档位 apply/skip 不一致）与排序矛盾（聚合误差小于分项误差，
  即定点量化抵消）分别显式标注并给出归因文字。

## 6. 并发一致性

候选检测与分项打分进入线程池，任务携带由 `(样本ID, 起始位置, 阶段名)` 派生的
稳定排序键；future 回收后统一排序，与单线程按样本 ID 升序结果逐字段一致。
已验证并发度 1/2/4/8/16 下候选顺序、抑制顺序、分数表、失败清单、复用清单等
8 类序列完全相同。

## 7. 快照与退出语义

- 阶段产物先写临时目录，再 `os.replace` 整体替换；清单记录全局版本号、配置哈希、
  规则指纹、已完成样本游标与各阶段版本。
- 续跑时配置哈希不一致输出 `E_CONFIG_MISMATCH` 并拒绝续跑。
- SIGINT/SIGTERM：完成当前样本阶段提交后以退出码 0 退出。
- 强制终止后仅依据完整替换的快照恢复；恢复点会实际解析快照，损坏文件计入
  `corrupt_stages` 并标记 `recoverable=false`。

## 8. 验证结果（实际执行）

- 真值校对：22/22 条带真值样本的修正文本逐字命中。
- 单元测试：`python3 -m unittest discover -s tests` → 21 passed。
- 并发一致：并发度 1/2/4/8/16 逐字段一致。
- 增量复用：调参后仅受影响阶段重算，其余阶段缓存命中复用。
- 回放确定：同一命令序列两次执行，corrected/diagnostics 的 SHA-256 完全相同。
- 异常注入：E_EMPTY_TEXT/E_PARSE_INVALID/E_RULE_UNKNOWN/E_DUPLICATE_ID 进入
  失败清单；非法配置抛 E_CONFIG_INVALID；配置漂移抛 E_CONFIG_MISMATCH。
- 优雅退出：SIGTERM 后提交当前样本、退出码 0、恢复点正确。

## 9. 常用命令

```bash
python3 scripts/generate_data.py                      # 重新生成确定性数据
python3 -m proofread_app --base-dir . run --clean     # 空白全量首跑
python3 -m proofread_app --base-dir . run --resume    # 依据快照续跑
python3 -m proofread_app --base-dir . replay --commands config/commands.jsonl
python3 -m proofread_app --base-dir . recovery        # 查看恢复点
python3 -m proofread_app --base-dir . clean           # 清理状态
python3 -m unittest discover -s tests                 # 运行全部测试
```
