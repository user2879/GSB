# 验证记录

环境：Ubuntu 22.04，Python 3.11.0rc1，NumPy 1.26.4，无容器/无外部服务。

## 用例矩阵

| 测试文件 | 覆盖需求 |
| --- | --- |
| `tests/test_loader.py` | 句子/词典/概率表/配置校验、重复 ID、非法字符、重复词形累加、非法配置枚举 |
| `tests/test_pipeline.py` | FMM 最长匹配与同长决断、确定性指纹、OOV 两规则边界差异与硬上限、维特比确定性、并列字典序、定点量化单调 |
| `tests/test_engine.py` | 增量=全量逐字段、命令分块(1/2/4)等价、精确失效、概率表全量失效、预算挂起与排空、待办不污染快照、并发(1/2/4/8)一致、稳定排序键、快照恢复、SIGKILL 后恢复 |
| `tests/test_invalid_and_docs.py` | 非法场景端到端、失败清单稳定顺序、产物字段完整性 |

## 实测命令与结果

```bash
python -m unittest discover -s tests -q
# Ran 28 tests ... OK （约 3.7 秒）

python scripts/verify_equivalence.py samples/base
# chunk_size=1: identical
# chunk_size=2: identical
# chunk_size=1000: identical
# EQUIVALENCE_OK

PYTHONPATH=src:. python scripts/demo_budget.py samples/base
# first response: E_INTERACT_BUDGET_PENDING（completed=0, pending=24）
# 多轮排空后 pending=[]，最终报告与无预算全量报告逐字段一致
```

## 非法注入场景（samples/invalid）

- 句子：空文本、重复 ID、受控表外字符（拉丁字母）。
- 词典：零词频、非数值词频、未声明词性、缺字段。
- 概率表：`start_logp.n = NaN`（非有限对数值）。

以上条目均进入对应错误码失败清单，合法条目继续参与处理，报告
`success_count == total_sentences`、`failure_count` 与注入数量一致。

## 强杀恢复验证

子进程先完成全量并提交 v1 快照，再执行交互命令、逐命令提交，随后对自身发
`SIGKILL`。新进程仅依据 `current` 完整快照恢复，报告恢复点版本号，继续处理
待办后成功句数等于总句数；`tmp/` 中的半成品候选目录不参与恢复。
