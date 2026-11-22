# Embedding 表与 Gather 算子定点诊断工作台（EGW）

面向推荐/检索场景的 Embedding 表与 Gather 算子诊断工具。聚合完全在定点整数域完成，
同时维护一条 float64 参考路径，对「行量化 → 取行 → 定点累加 → 均值除法 → 最终舍入」
五个环节逐级比对，定位误差首次超差的环节，并归因到量化截断或累加舍入。

## 环境要求

- Python 3.11+（已在 Python 3.11 验证）
- 依赖：`numpy`、`PyYAML`

```bash
python --version
pip install -r requirements.txt
```

不使用 Docker 或其他容器；所有样本由固定种子（`SEED=20261122`）在本地生成，无外部网络访问。

## 目录结构

```
embedding_gather_workbench/
├── egw/
│   ├── config.py        # 配置加载/校验/规范化配置哈希
│   ├── errors.py        # 业务错误码与失败清单排序
│   ├── quantize.py      # 定点量化、饱和累加、确定性除法舍入
│   ├── storage.py       # 行记录校验、行/块/布局三级指纹、编辑命令
│   ├── requests.py      # Gather 请求加载校验
│   ├── engine.py        # 双路径 Gather、五环节误差、失效账本、预算并发调度、快照
│   ├── layout_probe.py  # 块布局累加顺序可观测性探针
│   ├── samplegen.py     # 固定种子样本与异常注入
│   ├── scenario.py      # 编辑序列编排、增量/全量运行
│   └── cli.py           # generate / run / verify 命令行
├── tests/               # unittest 测试（19 个）
├── samples/             # 生成的本地 JSONL/YAML 样本
├── snapshots/           # 阶段快照（临时目录整体替换提交）
└── out/                 # 诊断报告与验收摘要
```

## 快速开始

```bash
# 1. 生成确定性样本（含非法行、非有限向量、未知聚合模式、零预算配置注入）
python -m egw generate --out-dir samples

# 2. 单次诊断（应用编辑序列后全量重建）
python -m egw run \
  --config samples/config.yaml \
  --rows samples/rows.jsonl \
  --requests samples/requests.jsonl \
  --edits samples/edits.jsonl \
  --snapshot-dir snapshots/run \
  --report out/report.json

# 3. 验收：单线程(1)与配置并发(3)各跑一次，增量 vs 全量逐字段对比
python -m egw verify \
  --config samples/config.yaml \
  --rows samples/rows.jsonl \
  --requests samples/requests.jsonl \
  --edits samples/edits.jsonl \
  --work-dir out

# 单元测试
python -m unittest discover -s tests -v
```

## 定点模型

- 量化步长 `step = 2^-fractional_bits`，量化为向零取整后按 `quant_level` 有符号档位截断。
- 量化残差 `residual = float_value - dequant(int_value)`，按行内维度序号升序保留。
- 累加器带饱和上限 `±2^(quant_level-1+headroom)`，因此分块归并为非结合运算，
  块布局变化可产生可观测的定点差异（见 `layout_probe.py`）。
- 均值除法支持 `TRUNCATE / FLOOR / HALF_UP / HALF_EVEN` 四种确定性舍入，
  余数恒满足 `total = quotient*count + remainder`。
- 全程无随机源；相同档位/位宽下重复运行逐元素一致。

## 错误码

| 码 | 触发条件 |
| --- | --- |
| `E_ROW_INVALID` | 行 JSON 解析失败、字段缺失、词 ID 重复、槽位冲突、负坐标、请求字段非法 |
| `E_DIM_MISMATCH` | 行向量维度与配置不一致 |
| `E_NONFINITE_VECTOR` | 向量含 NaN/Inf |
| `E_AGG_MODE_UNKNOWN` | 聚合模式不在配置枚举内 |
| `E_CONFIG_INVALID` | 配置/编辑命令非法（含预算非正） |
| `E_ID_MISSING` | Gather 序列中的词 ID 不存在（记录请求与位置后继续） |
| `E_POINT_BUDGET_PENDING` | 单批行访问点数预算不足，请求保留待办 |

## 失效与复用

- 行指纹由「原始内容字节 + 定点整数 + 档位 + 维度」派生；块指纹由偏移升序的行指纹
  有序列表与块布局派生；布局指纹再对块指纹有序聚合。
- 单行替换仅失效直接引用该词 ID 的请求；整块替换/追加新块失效引用受影响块的请求；
  档位/位宽切换清空全部请求缓存；聚合模式切换仅失效对应请求。
- 未失效请求直接复用上轮定点结果，绝不重算；复用/重算清单按请求 ID 稳定排序。

## 调度与快照

- 线程池最大并发数取自配置且不会超过；调度器按请求 ID 稳定顺序消耗单批点数预算，
  超预算请求以 `E_POINT_BUDGET_PENDING` 进入待办。
- 每个请求任务携带 `request_id::stage` 稳定排序键，最终请求顺序、行顺序、失败清单
  顺序与单线程按请求 ID 执行完全一致。
- 阶段产物先写临时目录，再 `os.replace` 整体替换；`manifest.json` 记录全局版本号、
  配置哈希、量化档位、块布局指纹与各阶段产物版本，启动时以版本号+配置哈希判定续跑。
