# 验证记录

## 环境

| 项 | 实测 |
| --- | --- |
| 操作系统 | WSL Ubuntu-22.04 |
| Python | 3.11.0rc1（满足 >= 3.10） |
| sqlite3 | 3.37.2（标准库可用） |
| NumPy | 1.26.4（`python3 -m pip install --user numpy`） |
| tkinter | 标准库在，但系统缺 `libtk8.6.so` 无法导入；已降级文本控制台 |

tkinter 启用方式（需系统权限，当前环境无免密 sudo）：

```bash
sudo apt install python3-tk
```

## 自动化测试

命令：`python3 -m unittest discover -s tests`

结果：**48 个测试全部通过，耗时约 5 秒**（详细日志见
`artifacts/run/test_output.txt`）。

| 测试文件 | 覆盖内容 |
| --- | --- |
| `test_spec.py` | E_SPEC 七类非法声明、首个非法位置/字段、归并排除 |
| `test_receipts.py` | r1/r2/r3 归一往返、补齐依据、E_FIELD/E_REF |
| `test_plan.py` | 任意顺序下逐字节一致、批大小重切分、覆盖请求排除 |
| `test_inference.py` | 定点前向确定性、参数版本/种子改变摘要 |
| `test_leases.py` | 配额等待、真实单调过期、版本乐观校验、回收幂等 |
| `test_worker_e2e.py` | 真实子进程回环推理、E_PORT、E_TIMEOUT |
| `test_lease_renew_e2e.py` | 续约端到端乐观锁、过期拒续 |
| `test_scheduler.py` | 双重上限峰值审计、等待计时、跨运行确定性 |
| `test_invalidation.py` | 参数/种子/批大小失效传播、全量重算比较 |
| `test_ledger_recovery.py` | 原子提交、快照版本、重放首个差异、unknown 恢复 |
| `test_crash_recovery_e2e.py` | 真实 SIGKILL 后只恢复已提交批并回收配额 |

## 演示场景实测（`python3 demo.py all`）

1. **强校验**：7 个非法样本全部以 `E_SPEC` 报告首个位置与字段，
   例如 `bad_host_literal.json -> /workers/0 host`。
2. **回执归一**：r1 归一补齐 5 个字段并记录依据；负耗时/重复名次 ->
   `E_FIELD`，悬空 `dup_ref` -> `E_REF`。
3. **确定性计划**：反转工作进程与投递顺序后，计划指纹仍为
   `c62eca493d0f2c10a62e1def801afdbd`，逐字节一致。
4. **真实回环调度**：4 批全部经真实子进程 TCP 完成，
   峰值活跃租约=在途批次=3，正好等于全局上限，未超限。
5. **失效传播**：仅 `worker-a` 参数版本 1->2 时，只失效归属它的 2 批；
   `worker-b` 两批 `digest_equal=True`，子集重算与空白全量重算 4 项全等。
   种子变更则 4 批全部失效且仍与全量重算一致。
6. **强制终止恢复**：子进程退出码 -9（SIGKILL），重启后
   2 批 `committed` 可恢复摘要，2 批 `unknown` 先回收配额；
   篡改一行 payload 后独立重放准确定位
   `receipts / b_7d40003d5cc96167bdfe850b:req-005:r1` 的首个摘要差异。

## 验收结论

回执归一、租约配额调度、失效传播、中断恢复四类结论在本地确定性样本上
均可唯一复核，且与从空白全量重算的结果逐项一致。
