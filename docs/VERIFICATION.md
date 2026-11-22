# 验证记录

环境：Python 3.11.0rc1、NumPy 1.26.4、pytest 8.3.3、PyYAML 6.0.2，
全程本地运行、无网络访问、未使用容器。

## 1. 自动化测试

`python -m pytest -q` 结果：**37 passed**。覆盖关系：

| 测试文件 | 覆盖业务要求 |
| --- | --- |
| `test_loader.py` | 要求 1：JSONL/YAML/配置校验、错误码与失败清单 |
| `test_slots.py` | 要求 2：类型校验、四策略、定点加权、非法值隔离 |
| `test_state_machine.py` | 要求 3：合法/拒绝转移、状态保持与后续推进 |
| `test_pollution.py` | 要求 4：替换/删除/FSM 变更的精确污染边界 |
| `test_equivalence.py` | 要求 5：追加/替换/删除/模式变更/策略切换的逐字段等价 |
| `test_scheduler.py` | 要求 6：预算耗尽待办、配额中止、稳定顺序与账本 |
| `test_snapshot.py` | 要求 7：原子阶段提交、续跑、版本/哈希陈旧判定 |
| `test_injection.py` | 要求 8：四类注入命中、隔离、报告章节与等价性 |
| `test_budget_equivalence.py` | 预算续跑与一次性全量的业务字段一致 |
| `test_recovery_e2e.py` | 中断后续跑收敛与全量结果一致 |
| `test_concurrency_determinism.py` | 单/多线程结果一致、账本会话稳定顺序 |

## 2. `python run_demo.py` 实测输出

```
[1] incremental vs full replay equivalence
    field-level identical: True
    reused cached turns  : S001=[1..6], S002=[1..8], S003=[1..7], S004=[1..7]
    fingerprints: inc=5e8aabffbf2d77c9 full=5e8aabffbf2d77c9
[2] budget exhaustion, todo retention and resume
    run #1: consumed=10 remaining=0 pending=['S002','S003','S004']
    run #2: consumed=20 pending=['S002','S003','S004']
    run #3: consumed=25 pending=[]
    ledger gap-free and auditable: True
    injection hits: INJ-BAD-SLOT-01, INJ-QUOTA-01, INJ-TURN-JUMP-01, INJ-UNKNOWN-TX-01
[3] staged snapshot recovery and staleness detection
    after load stage   : stage=load symlink=True
    after advance stage: checkpoints=32 pending=[]
    after settle stage : completed_batches=5 global_version=2
    config change -> E_SNAPSHOT_STALE: pipeline config hash changed; ...
```

## 3. 验收点对照

- **增量与全量一致**：编辑序列后两报告 `report_fingerprint` 相同，
  比较视图（槽位取值/置信度/历史、当前状态、转移轨迹、拒绝、失败清单）
  逐字段相等；`test_equivalence.py` 与 `test_budget_equivalence.py` 断言。
- **预算可续跑**：紧预算（10）首轮即耗尽并保留待办，多轮续跑后
  `pending=[]`、累计消耗 32，账本 `seq` 连续无空洞。
- **快照恢复判定**：三阶段均为符号链接整体替换；配置哈希变化触发
  `E_SNAPSHOT_STALE` 并从头重放，一致时从检查点前缀恢复。

## 4. 复现命令

```bash
python -m dialog_state_pipeline.cli gen-samples --out samples
python run_demo.py
python -m pytest -q
```

## 5. 附加健壮性验证

- 独立进程重复运行产出字节级相同的报告（SHA-256 一致）。
- 空事件流、非 JSON 行、非字典槽位指令、未知槽位、未知转移动作均被
  归类为对应业务码并进入失败清单，不产生未捕获异常。
- 注入配额覆盖使用运行期配置副本，续跑不会误报 `E_SNAPSHOT_STALE`
  （`test_injection_resume_does_not_falsely_flag_stale` 回归保护）。
