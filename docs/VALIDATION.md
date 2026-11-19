# 实际验证记录

以下结果来自本次交付环境中的真实执行，不是占位示例。

## 环境

```text
Python 3.10.12
NumPy 1.21.5
SQLite 3.37.2
Tk 8.6
OS: Ubuntu 22.04 / WSL 本地文件系统
```

Windows 目标环境使用相同的 Python 标准库、NumPy API 与 Tkinter；入口为 `py -3.10 -m batch_audit.cli`。

## 自动化测试

执行：

```text
python3.10 -m unittest discover -v
```

结果：

```text
Ran 28 tests in 6.231s

OK
```

覆盖项：

- 相同种子/维度生成逐位一致；
- 输入与参数摘要不随样本行排列变化；
- 推理与批次位置无关，重复推理逐位一致；
- 投递文件逆序、条目乱序、同槽重投；
- 已提交后仅重排投递文件不会产生新版本或破坏审计；
- 槽位处理顺序正序/逆序编号一致；
- 首错误字段路径与到达序号；
- 准入失败条目可使用外部样本标识；
- 自定义槽位集合可从工作区 manifest 自动继承；
- SQLite 强杀后无退出报告、已提交槽位保留、未知槽位重做；
- 编号口径演进后的整体替换与受影响计数；
- 模型资产演进后的新版本；
- 账本内容摘要、参数数组、输出摘要、投递摘要篡改定位；
- Tkinter 模块在无窗口环境下可导入。

## 完整验收工作区

实际执行：

```bat
py -3.10 -m batch_audit.cli init --workspace workspace --seed 20261120 --samples 12 --input-dim 6 --hidden-dim 8 --slots 0,1,2,3,4
py -3.10 -m batch_audit.cli run --workspace workspace
py -3.10 -m batch_audit.cli audit --workspace workspace
```

实际结果：

```text
generation: 1
status: complete
numbering_version: BA-V1
progress_cursor: 4
completed_slot_count: 5 / 5
assignments: 10
empty slot: 4, error_code=E_EMPTY
overwritten retransmissions: 2 (S0000, S0001)
admission failures: 2 (S0010 policy_denied, S0011 precheck_missing_field)
audit code: AUDIT_OK
audit scope: full
auditable slots: [0, 1, 2, 3, 4]
assignments compared: 10
```

摘要值：

```text
feed_digest:
5ee1f1ac8dbd79d08bcd59a697abae68c6b6cebd83bd94cc961c4886681103c5

input_digest:
01f9303bbd54846c94eb84b99650c48a949a0eccdefc70cc972aaa9b94c4d746

param_digest:
dd72e1650b044996793afddbc5e2dcb882d77e79d36e6137a01e92958cee647a

config_digest:
7cc4d074c235685186b3b24b012882000e8552ae5e2a5e8d680eee6eadf43131

audit recomputed_digest:
38f03afef5f8ac48285fa1a5fbe700c2438b86bc2c2501255b67dc616e30deff

ledger content digest:
22912a544973ab7984d1c277935635b72ae1ec4cef516362ccf2449f7b8f027a
```

槽位结果：

```text
slot 0 committed: S0000(rank 1), S0004(rank 2), S0008(rank 3)
slot 1 committed: S0001(rank 1), S0005(rank 2), S0009(rank 3)
slot 2 committed: S0002(rank 1), S0006(rank 2)
slot 3 committed: S0003(rank 1), S0007(rank 2)
slot 4 empty: E_EMPTY
```

部分实际编号：

```text
S0000 -> BA-V1:S0:R000001:fe81b1eb39cf0e4e8812
S0001 -> BA-V1:S1:R000001:ed6b5b4fbf922d689cc3
S0002 -> BA-V1:S2:R000001:a11b1b357a1ca4abd656
S0003 -> BA-V1:S3:R000001:70bd92ad368d34d44b4d
```

槽位 0 批次输出摘要：

```text
0bc218bd9454bedac6223e261600d19d3ffc89d34ef3b925ed08c2adb8187789
```

同一工作区再次执行 `run`，仍为 generation 1，10 条编号逐一相同。

## 强制终止验证

执行：

```text
python3.10 -m batch_audit.cli run --workspace crash-demo --crash-after-commits 2
```

实际结果：

```text
exit code: 99
report.json exists: no
slot 0: committed
slot 1: committed
slot 2: pending
slot 3: pending
slot 4: pending
```

重启之前审计：

```text
code: AUDIT_PARTIAL
audit_scope: partial
auditable_slots: [0, 1]
assignments_compared: 6
not_auditable_slots: [2, 3, 4]
reason: NOT_COMMITTED_BEFORE_TERMINATION_OR_NOT_YET_EXECUTED
```

重启后：

```text
status: complete
interrupted_attempts: [1]
unknown_slots_before_redo: [2, 3, 4]
redone_slots: [2, 3, 4]
previously committed slot 0/1 assignments: unchanged
```

续跑后审计：

```text
code: AUDIT_OK
audit_scope: full
auditable_slots: [0, 1, 2, 3, 4]
assignments_compared: 10
```

## 版本演进验证

以 `--numbering-version BA-V2` 重新执行后：

```text
previous generation: 1 (superseded)
current generation: 2
replacement_reason: fingerprint_changed
affected_samples: 10
affected_slots: [0, 1, 2, 3, 4]
current IDs: BA-V2:*（与 BA-V1 旧编号集合无交集）
audit: AUDIT_OK (generation 2)
```

另验证快照版本被改为 `99` 后重开：

```text
new generation: 100
replacement_reason: snapshot_version_mismatch
old progress was not merged into generation 100
```

## 首个不一致定位

在验收工作区副本中篡改槽位 `2`、名次 `1` 的 `sample_output_digest`，实际审计退出码为 `3`，输出：

```json
{
  "code": "E_AUDIT",
  "first_mismatch": {
    "actual": "tampered-value",
    "batch_slot": 2,
    "expected": "be18c789fff42c022313d2a942535a0539f3b571f66626730c7e2c6dd0475696",
    "field": "sample_output_digest",
    "sample_id": "S0002"
  },
  "generation": 1,
  "message": "assignment field mismatch"
}
```

其他篡改路径也已验证：

- 投递原始文件变化先定位为 `field=feed_digest`；
- 账本内嵌 `W1` 数组数据变化定位为 `field=arrays[W1].digest`；
- 只改账本内容摘要定位为 `field=ledger_content_digest`。
