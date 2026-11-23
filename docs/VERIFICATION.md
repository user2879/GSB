# 验证结果记录

## 测试套件
- `python3 -m pytest tests/ -q` -> 15 passed

## CLI 全量+15 条交互命令

- 完成工单: 8 / 12
- 升级工单: 3，边界工单: ['T0007']
- 注入命中: ['INJ-004:E_FIXED_WIDTH', 'INJ-003:E_RULE_UNKNOWN', 'INJ-002:E_RULE_CYCLE', 'INJ-001:E_EVIDENCE_INVALID']
- 失败清单顺序: [('T0008', 'E_FIXED_WIDTH'), ('T0009', 'E_RULE_UNKNOWN'), ('T0010', 'E_RULE_CYCLE'), ('T0011', 'E_EVIDENCE_INVALID')]

## 三版本序列化往返

- v1: 偏差桶=[0] 计数=[8] 最大绝对偏差=0 原因码还原=8/8
- v2: 偏差桶=[0] 计数=[8] 最大绝对偏差=0 原因码还原=8/8
- v3: 偏差桶=[0] 计数=[8] 最大绝对偏差=0 原因码还原=8/8
