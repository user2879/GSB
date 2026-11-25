# 验收运行结果

- 总状态：通过
- 端到端耗时：4026.097 ms
- 样本/实体/抑制/修复：10 / 111 / 8 / 0
- 增量与空白全量逐字段一致：True（指纹 `a283321911942694`）
- 快照往返一致：True；版本探测码：`NER-E4001`
- 预算陈旧摘要：True（`NER-E3002` → fresh）
- SIGKILL 后恢复并 parity：True（kill code -9）
- 异常注入：6/6，业务码 {"NER-E1002": 1, "NER-E1003": 1, "NER-E1004": 1, "NER-E2001": 1, "NER-E2004": 1, "NER-E2005": 1}

## 命令级失效/复用

| op | affected | invalidated | recomputed | reused |
|---|---:|---:|---:|---:|
| upsert_lexicon | 0 | 0 | 0 | 40 |
| replace_sample | 1 | 4 | 4 | 36 |
| set_emission_weight | 10 | 30 | 30 | 10 |
| set_transition_weight | 10 | 20 | 20 | 20 |
| set_quant | 10 | 30 | 30 | 10 |
| upsert_lexicon | 4 | 16 | 16 | 24 |
| remove_lexicon | 0 | 0 | 0 | 40 |
| set_quant | 10 | 30 | 30 | 10 |
