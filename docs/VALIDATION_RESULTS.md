# 验证结果记录

- 运行环境：WSL2 Ubuntu，Python 3.11.0rc1，sqlite3/threading/hashlib/json 标准库，
  numpy 1.26.4（可选后端）；当前无头环境未装 tkinter，GUI 采用懒加载，
  CLI/核心/测试不依赖 tkinter（Windows 官方 Python 勾选 tcl/tk 即可启动 GUI）。
- 固定随机种子：`20261122`；样例 8 个对象，热层容量 20224 字节、冷层 80896 字节，
  热层水位 600‰、冷层回收线 200‰、批次上限 2。
- 复现命令：`python3 -m unittest discover -s tier_mover/tests -p 'test_*.py'`
  或 `./run_tests.sh` / `.\run_tests.ps1`；一键演示 `./run_demo.sh` / `.\run_demo.ps1`。

## 1. 自动化测试结果

```
Ran 28 tests in 4.3s
OK
```

| 测试文件 | 用例数 | 覆盖业务要求 |
| --- | --- | --- |
| test_manifest_heat.py | 7 | 要求 1 强校验、确定性摘要、整数热度 |
| test_planner_engine.py | 8 | 要求 2/3/4/8 计划确定性、原子搬运、批次、E_CAPACITY、E_INTEGRITY、rewrite |
| test_versioning.py | 3 | 要求 5 版本单调、stale 失效传播、结论整体替换 |
| test_recovery_replay.py | 6 | 要求 6/7/8 崩溃未知标注、rollback/rebuild、重放、E_VERSION |
| test_cli_demo.py | 4 | CLI 端到端、退出码、演示报告、崩溃恢复链路 |

## 2. 迁移计划与真实搬运（要求 2/3/4）

计划推导（节选自实际输出 `/tmp` 验证工作区）：

```
evict_keys   : obj-alpha, obj-hotel, obj-bravo, obj-foxtrot
retained_keys: obj-charlie, obj-delta, obj-echo, obj-golf
每个 step 含 direction/heat/recency_rank/ranking/size_bytes
plan_digest  : 4710c8f8300d996b35fb8669129dfc3eb8c7497e3e949fcc65243c004a45840c
```

- 同一输入两次 `plan` 输出逐字节一致（`plan_deterministic: true`）。
- `run` 结果：`batches=2`（受 `batch_limit=2` 约束）、`migrations_committed=4`，
  执行后热层 11776B / 冷层 8448B，均未超容量。
- 目录扫描真值与账本一致；对象按 SHA-256 摘要命名，每层唯一存在，无双层归属。
- 账本逐层整数记录：热层 cost=11776×5=58880，冷层 cost=8448×1=8448。

## 3. 三类异常注入业务码（要求 8，进程退出码均为 2）

```
E_CAPACITY : 冷层 cold 余量 767B 不足以接纳 obj-alpha（需 2048B），rejected_keys=["obj-alpha"]
E_INTEGRITY: 层 hot 目录字节数 19200 与账本 20224 不一致（对象被截断）
E_VERSION  : 账本格式版本冲突: 文件版本 999，软件支持版本 3
```

- 注入只改输入文件/参数：容量改 `tiers.json`、截断热层 `.bin`、复制账本改副本版本号。
- 异常后原账本均可重新打开（`status` 正常），满足“账本状态仍可恢复”。
- 三类事件均写入 `logs/events.jsonl`。

## 4. 强制终止恢复（要求 6）

`run --crash-after-key obj-alpha` 以退出码 137 终止（对象已搬运、批次未提交）：

```
inspect : committed_objects=7, unknown_objects=1, unknown_keys=["obj-alpha"],
          aborted_open_batches=[1]
rollback: rolled_back_objects=1，obj-alpha 按摘要回退 hot，目录与账本重新对齐
rebuild : rebuilt_objects=1，obj-alpha 按 cold 重建归属并更新账本
```

恢复后重新 `run` + `replay`：`steps_compared=4, batches_compared=2`，
最终归属/占用/成本与账本逐项一致。组件未生成任何“强制终止退出报告”，
仅输出含已提交/未知/已回退/已重建计数的恢复报告。

## 5. 重放校验（要求 7）

- 正常链路重放：4 个迁移步骤、2 个批次全部比对通过，最终占用与成本如上。
- 篡改账本首条迁移记录摘要后，重放立即报：
  `E_REPLAY`，`object_key=<首个对象键>`，`mismatch_field=content_digest`，
  并携带 basis/expected/actual，定位首个分歧点而非笼统失败。

## 6. 版本演进与陈旧防护（要求 5）

演示场景 07：v1 已生成未执行计划后修改规则（batch_limit 2→3）注册 v2：

```
superseded_plan_fragments 数量 = 1（v1 pending 计划整体 stale）
current_view.active_spec_version = 2，pending_plan = null（stale 不进当前视图）
active_conclusion.spec_version = 2（结论整体替换，旧结论 status=stale）
```

## 7. 一键演示报告

`python -m tier_mover.cli demo <dir>` 产出 7 个场景与汇总
（副本见 `docs/demo_report.json`）：

```
all_business_codes = {E_CAPACITY, E_INTEGRITY, E_VERSION}
scenarios = happy_path, capacity_injection, integrity_injection,
            version_injection, crash_rollback, crash_rebuild, version_evolution
```

## 8. 已知环境说明

- 本机无 tkinter，GUI 未在此环境实际弹窗；`tier_mover/gui.py` 仅在调用
  `python -m tier_mover.gui` 时才 import tkinter，缺失时给出明确安装指引，
  不影响其余全部功能与 28 个测试。
- Windows 上使用官方 Python 3.11+ 安装包（勾选 tcl/tk）即可直接运行 GUI，
  代码仅使用跨平台标准库与 pathlib/os.replace，无 POSIX 专属依赖。


### rebuild 路径的重放步数说明

`rebuild` 把崩溃时未提交的对象归属直接重建到目标层，因此该对象不再需要
重新迁移，恢复后续跑的已提交迁移步数为 3（rollback 路径对象回到源层、
需重新迁移，步数为 4）。两者重放均通过且最终视图与账本逐项一致，
这是两种合法处置语义的预期差异，已由测试与 demo 报告固化。
