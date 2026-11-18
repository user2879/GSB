# 本地确定性 LLM 评测与幻觉检测管线

一个完全离线、确定性、可审计的数据处理管线。给定相同的输入与配置，重复运行
产生逐字段一致的产物与 JSON 报告；增量运行（含规模档切换、阈值调整、规则权重
变更、偏好对替换）与从空白工作区的全量重建结果一致。

## 目录结构

```
llm-eval-pipeline/
├── eval_pipeline/
│   ├── config.py        # 配置 / 规则表 / 注入配置加载与校验
│   ├── errors.py        # 稳定业务码与 PipelineError
│   ├── util.py          # FNV-1a 稳定哈希、内容指纹、原子写
│   ├── loader.py        # 两份 JSONL 逐行解析与失败清单
│   ├── features.py      # 子词切分、哈希桶计数、成对奖励打分
│   ├── hallucination.py # 实体 / 数值 / 引用三类一致性规则
│   ├── selector.py      # 规模自适应策略选择器（预置策略表 + 候选评分）
│   ├── metrics.py       # 批量与流式两条等价指标算法路径
│   ├── cache.py         # 内容指纹缓存、精确失效与复用计数
│   ├── snapshot.py      # 阶段化快照、原子替换、血缘校验
│   ├── runner.py        # 携带稳定排序键的线程池执行器
│   ├── generator.py     # 固定种子本地样本生成（受控词表，无网络）
│   ├── pipeline.py      # 编排、异常隔离与报告组装
│   └── cli.py           # generate / run / verify / clean 命令
├── tests/test_pipeline.py
├── requirements.txt
└── samples/             # 由 generate 命令产出
```

## 快速开始

```bash
pip install -r requirements.txt
# 1) 确定性生成样本（偏好对、评测样本、规则表、配置、注入配置）
python3 -m eval_pipeline generate --sample-dir samples/baseline \
    --preferences 600 --evaluations 300 --seed 42
# 2) 运行管线
python3 -m eval_pipeline run --sample-dir samples/baseline --workdir work/baseline
# 3) 比对两份报告的核心结果字段（忽略缓存计数等运行态字段）
python3 -m eval_pipeline verify --normalized report_a.json report_b.json
```

崩溃恢复演练（在处理若干条后注入崩溃，再续跑）：

```bash
python3 -m eval_pipeline run --sample-dir samples/baseline --workdir work/crash --crash-after 37
python3 -m eval_pipeline run --sample-dir samples/baseline --workdir work/crash
```

严格续跑（配置/规则/档位/数据血缘变化时拒绝续跑并返回 `E_CONFIG_MISMATCH`）：

```bash
python3 -m eval_pipeline run --sample-dir samples/baseline --workdir work/x --strict-resume
```

## 确定性设计

- 哈希使用 64 位 FNV-1a（`util.stable_hash`），不依赖 Python 进程随机化的 `hash()`。
- 子词切分为纯函数：精度 1 仅词元，精度 2 加相邻二元组，精度 3 加三元组。
- 奖励打分按桶下标固定顺序做浮点求和，记录双精度参考差值 `reward_diff_ref`（`float.hex()`）。
- 采样按内容哈希阈值进行，不使用运行时 RNG；样本生成使用固定种子 LCG。
- 所有 JSON 以排序键 + 固定分隔符规范序列化后再计算指纹与比对。

## 业务码

| 业务码 | 触发条件 |
| --- | --- |
| `E_EMPTY_TEXT` | prompt/chosen/rejected 或 context/answer 为空白 |
| `E_BAD_JSON` | JSONL 行非法、缺字段或时间戳不可解析 |
| `E_DUP_ID` | pair_id / sample_id 重复 |
| `E_CONFIG_INVALID` | 并发数非正、档位非单调、桶数与维度不一致等 |
| `E_RULE_WEIGHT` | 幻觉规则权重越界（隔离为失败清单，不中断整体） |
| `E_TIER_LIMIT_ZERO` | 规模档上限为零 |
| `E_NON_FINITE` | 注入产生 NaN/Inf 特征或奖励差值 |
| `E_THRESHOLD_BOUNDARY` | 阈值恰好落在边界的注入 |
| `E_CONFIG_MISMATCH` | 快照版本/配置/规则/档位/数据血缘不一致，拒绝续跑 |
| `E_STAGE_FAILED` | 单条样本在某阶段抛出未预期异常（隔离后继续） |

## 缓存与快照语义

- 缓存键 = 阶段 + 上游内容指纹 + 参数指纹 + 规模档标识（`cache.FingerprintCache.derive_key`）。
- 偏好对替换仅改变该条及下游键；规则权重变更只影响幻觉检测与指标；未失效条目直接复用。
- 快照按 features/scoring/hallucination/metrics 四阶段以临时文件 + `os.replace` 原子落盘，
  含阶段名、配置哈希、规则表哈希、数据指纹、规模档、游标、产物摘要与版本号。
- `cache` 与 `resume` 属于运行态审计计数，不参与核心结果的逐字段一致性比对。

## 异常注入

注入配置 `injection.json` 默认全部 `enabled: false`。可用注入项：

1. `non_finite_feature`：将指定偏好对奖励差值替换为 NaN，记录 `E_NON_FINITE` 与注入编号。
2. `rule_weight_out_of_range`：在备用规则表中把权重设为越界值，命中 `E_RULE_WEIGHT`。
3. `tier_limit_zero`：在备用配置中把档位上限设为 0，命中 `E_TIER_LIMIT_ZERO`。
4. `threshold_boundary`：阈值恰好等于边界，命中 `E_THRESHOLD_BOUNDARY`。

## 规模自适应策略

预置策略表 `selector.STRATEGY_TABLE` 按规模档给出特征精度、哈希桶放大系数与
幻觉阈值修正量；选择器输出候选评分、参考规模指标与最终理由，保证同规模同结论。

## 验证

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖业务码、确定性哈希与打分、三类幻觉规则、策略随规模切换、流式/批量指标
等价、缓存复用与失效、编辑序列增量 vs 全量逐字段一致、并发 vs 串行一致、
快照崩溃恢复一致、严格续跑拒绝与异常注入隔离。实际运行结果见 `VALIDATION.md`。
