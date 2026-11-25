# 实际验证结果

- 平台: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
- Python: 3.11.0rc1
- NumPy: 1.26.4
- SQLite: 标准库 sqlite3（无外部数据库服务）

## 测试

- `python3 -m unittest discover -s tests -v`：9 组全部 OK。
- 覆盖 E_SPEC 首错定位、逐字节确定性、去重加权、误差归因、双配额调度、
  局部失效、账本整体替换、崩溃恢复与 E_REPLAY、NumPy 整数交叉校验。

## 误编、归因、恢复三类结论

- 基线输出: data/baseline.json
- 基线 SHA-256: 81610e408fae6ccdeeb12aee049692d8b8226d73d4a4d4861c4c94a0b796cd58
- 引用图摘要: c588dc00d827d41a371b20c61dab7643d369c6222e33c6aa98435f41e3219429
- 图节点/边: 192 / 304
- 编辑 weights: 失效轮次 23/24，未失效保持检查 1 项，全部一致=True，账本整体替换=True
- 编辑 rules: 失效轮次 24/24，未失效保持检查 0 项，全部一致=True，账本整体替换=True
- 编辑 precision: 失效轮次 24/24，未失效保持检查 0 项，全部一致=True，账本整体替换=True
- 编辑 channel-weight: 失效轮次 22/24，未失效保持检查 2 项，全部一致=True，账本整体替换=True
- 崩溃恢复: 已提交批次 1，未知并重跑轮次 22，重放状态: all committed rounds matched ledger byte-for-field

## 重复运行一致性

- `tools/verify_byte_identity.py` 两次独立运行输出 SHA-256 相同。
- 复算摘要: 81610e408fae6ccdeeb12aee049692d8b8226d73d4a4d4861c4c94a0b796cd58
- 复算摘要: 81610e408fae6ccdeeb12aee049692d8b8226d73d4a4d4861c4c94a0b796cd58
