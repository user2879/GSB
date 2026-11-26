# 实际验证记录

## 环境

- WSL Ubuntu（Ubuntu-22.04），Python 3.11.0rc1
- NumPy 1.26.4（`python3 -m pip install -r requirements.txt`）
- SQLite 3.37.2（Python 标准库 `sqlite3`）
- 回环绑定验证：`socket.bind(("127.0.0.1", 0))` 成功；样本预留高位端口
  19101/19102，无外部账号、网络服务或数据集

## 自动化测试

命令：`python3 -m unittest discover -s tests`

结果：`Ran 41 tests ... OK`（全部通过）。覆盖矩阵：

| 测试文件 | 覆盖验收点 |
| --- | --- |
| `tests/test_spec.py` | E_SPEC 强校验、首个非法字段、JSON/NDJSON 来源行号、端口/标识唯一 |
| `tests/test_receipt.py` | r1/r2/r3 编解码归一、补齐字段与依据、E_FIELD/E_RANK/E_REF |
| `tests/test_plan.py` | 任意派发/字典顺序下逐字节一致、批标识与内容摘要稳定 |
| `tests/test_fixedpoint.py` | 确定性输入/参数、整数输出、顺序无关摘要、版本变更影响 |
| `tests/test_lease.py` | 单调时钟到期、乐观续约版本冲突、到期/取消唯一归属 |
| `tests/test_end_to_end.py` | 真实子进程、三格式与版本混杂、双上限不超限、排队单调计时 |
| `tests/test_invalidation.py` | 依赖链失效传播、稳定输出、未失效批摘要逐字节不变 |
| `tests/test_ledger.py` | 单事务原子提交、快照整体替换、崩溃 unknown、E_REPLAY 定位 |

## 演示运行（`python3 run_demo.py`）

- 三批计划签名 `238ce4726c85c50c`，批标识由工作进程+批序号+到达序号稳定推导
- 同一运行内三批分别以 r1/r2/r3 回收并归一：r1 补齐 6 个历史字段，
  r2 补齐 3 个，r3 无补齐；每批均绑定真实租约 id
- 将 w1 参数版本由 v0 升至 v1：仅 w1 的两批与请求 r01/r02/r04/r06 失效，
  w2 批次保持未失效且摘要不变
- 账本重放：`replay identical: True`
- 强制终止（首批保留未提交行）：恢复出 1 个 `unknown` 批与 2 个已提交批，
  仅对已提交子集重放且 `replay of committed subset identical: True`

## 验收结论

四项验收点均在本地确定性样本下通过：

1. 三格式回执归一可唯一复核（带来源版本、补齐字段集合与依据）
2. 租约对同一槽位的到期/取消回收唯一归属，双上限下不超限
3. 参数版本→批→请求的失效传播稳定，未失效批摘要逐字节保持
4. 强制终止后仅依据已提交记录恢复，重放差异以 E_REPLAY 定位首处不一致
