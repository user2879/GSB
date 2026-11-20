# 本地电商查询理解与意图解析服务

纯 Python（标准库 + NumPy + pytest）的确定性查询理解服务，覆盖查询归一化、
意图分类、实体槽位抽取、查询改写、规模自适应策略选择、并发一致性、
增量缓存快照与可复现实验报告。无外部网络服务、无容器、无外部语料。

## 环境准备

要求 Python ≥ 3.11。离线环境可用本地 wheel 安装依赖：

```bash
python3 -m pip install --user numpy pytest
python3 -m pytest --version
```

## 一分钟快速开始

```bash
# 1. 生成四个规模档的确定性合成数据（固定种子 20261120）
python3 -m qu generate --data-dir data/generated

# 2. 运行多档实验并输出 JSON 报告
python3 -m qu run --data-dir data/generated --work-dir artifacts/run

# 3. 运行测试套件
python3 -m pytest -q
```

报告写入 `artifacts/run/experiment_report.json`，每档快照写入
`artifacts/run/snapshot_<tier>.json`。

## 常用命令

```bash
# 只跑 tiny/small 两档，调整并发与延迟预算
python3 -m qu run --tiers tiny,small --workers 8 --budget-ms 10

# 连续规模扫档：自定义 12k 查询 / 12k 词表 / 18k 实体 / 220 规则
python3 scripts/scale_up.py --data-dir data/custom \
    --custom sweep:12000:12000:18000:220
python3 -m qu run --data-dir data/custom --work-dir artifacts/sweep
```

## 输入数据格式（JSONL，每行一个对象）

查询样本：

```json
{"query_id": "q-000001", "query": "星河 手机 红色",
 "context": {"session_pos": 0}, "labels": {"intent": "phone_digital"}}
```

意图词表 `{"intent","term","weight"}`；实体词典
`{"entity_type":"brand|category|attribute","term","entity_id","weight"}`；
改写规则 `{"rule_id","type":"synonym|category_expand|drop_recover",
"trigger_term","replacement","weight","ref_terms":[...]}`。

权重必须为有限数，规则 `ref_terms` 引用的词条必须存在，否则分别报
`E_INVALID_WEIGHT` / `E_RULE_DANGLING`。

## 业务码

| 业务码 | 触发场景 |
| --- | --- |
| `E_INVALID_JSON` | JSONL 行无法解析或非对象 |
| `E_EMPTY_QUERY` | 空查询或归一化后仅剩标点 |
| `E_DUPLICATE_QUERY_ID` | 查询 ID 重复 |
| `E_INVALID_WEIGHT` | 词表/词典/规则权重非有限数 |
| `E_RULE_DANGLING` | 规则引用了不存在的词条 |
| `E_CONFIG_MISMATCH` | 快照配置哈希/档位/版本不符，拒绝续跑 |
| `E_PIPELINE_STAGE` | 单样本处理阶段异常的兜底码 |

## 目录结构

```
qu/            服务源码（模块说明见 DESIGN.md）
tests/         34 个 pytest 用例，覆盖八项业务要求
scripts/       规模放大辅助脚本
data/          生成的合成数据（默认不入库）
artifacts/     实验报告与快照输出
DESIGN.md      算法取舍、等价性保证与实测数据
```

## 核心保证

* **剪枝不改结果**：三条实体路径与两条意图路径产出同一候选/命中集合，
  每档用不剪枝参考路径对确定性子集逐字段复核。
* **完全确定性**：无随机源；同输入同配置多次运行、不同并发度逐字段一致。
* **可审计策略**：决策记录含规模指标、12 个候选策略估计代价、最终选择
  与理由，策略指纹用于缓存作用域。
* **可恢复**：原子快照 + 内容指纹增量失效；配置不符拒绝续跑。

详细算法、公式与实测数据见 [DESIGN.md](DESIGN.md)。
