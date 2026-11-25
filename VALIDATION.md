# 实际验证记录

验证日期：2026-11-26  
环境：WSL2 Ubuntu 22.04.5，Linux 6.18.33 内核，Python 3.11.0rc1，numpy 1.26.4，Tk 8.6.12  
固定种子：`20261126`

## 自动化测试

```text
$ .venv/bin/python -m unittest discover -v
Ran 26 tests in 5.684s
OK
```

覆盖：输入校验/命令回滚、通道权威/覆盖通道/引用计次、恰等阈值、证据分布影响话题、跨话题归属、定点舍入/量化、逐游标全量增量等价、预算待办、并发度无关、误差首超差、翻转归因、快照版本冲突/续跑、端到端崩溃恢复。

GUI 冒烟检查：

```text
$ .venv/bin/python -m evidence_diag --root . gui --smoke-test
tk_smoke_test = ok
tk_patchlevel = 8.6.12
preloaded = libXft.so.2, libXss.so.1, libtcl8.6.so, libtk8.6.so
```

## 演示结果

```text
$ .venv/bin/python -m evidence_diag --root . demo
all_cursor_results_equal       true
cursor_count                   18
budget_pending_cursor          9
budget_pending_count           7
final_todo_count               0
failure_codes                  [E_INPUT_INVALID]
fixed_point_change_verified    true
diagnostic_first_exceed_count  8
diagnostic_flip_count          1
snapshot_crash_recovered       true
config_mismatch_full_recompute true
```

初始配置指纹：`d3e01bc3eeabc145bd44eb77666a156b99b5c84576b0bd5292044830591e866c`  
最终结果哈希：`9c18598a013bfa42507231d8a6ef513189ef0e03439acc74ed3068c2befac12c`

## 定点误差

- 目标域：8 位小数、量化步长 4、`half_up`，scale=256。
- 参考域：24 位小数、量化步长 1、`half_up`，scale=16777216。
- 容差：`0.0015`。
- 召回：34 个逐轮证据项全部超差，最大绝对差约 `0.006875`。
- 去重：34 项全部超差，最大绝对差约 `0.006875`。
- 话题：34 项，0 项实际归属变化。
- 置信度：42 项（含逐证据贡献与总项），38 项超差，最大绝对差约 `0.015183`。
- 每轮首次超差 stage 均为 `recall`，完整差值与标注见 `reports/diagnostic.json`。
- 第 1 轮发生答案翻转：目标域 `0.578125 >= 0.5625` 判 answer；参考域 `0.567842543... < 0.57` 判 abstain。归因列表包含去重权威、跨话题惩罚和舍入方向，主因为去重合并顺序。

## 恰等阈值

最终轮 1/2 的组合边界分数为 `2971bp`。将阈值精确设为 `2971bp` 后：

```text
exact_equal = true
split = false
decision = continue
equal_rule = continue_on_equal
```

## 预算与失败隔离

- 命令 8 将预算设为 5，命令 9 切换到 8 位/量化 8，触发多个轮次失效。
- 游标 9 只完成 1 轮置信度合成；轮次 2..8 共 7 条待办，成本分别为 4,4,4,4,4,5,5。
- 命令 10 恢复预算 200 后待办被优先消费，最终待办为空。
- 命令 13 设置负通道权重，事务回滚，失败清单仅保留该条 `E_INPUT_INVALID`，后续命令继续执行。

## 快照恢复

- 在命令 10 的阶段文件写完后发送 `os._exit(87)`；当前完整指针仍停在游标 9。
- 遗留 1 个 `.tmp-*` 临时目录，恢复时被忽略/清理。
- 重启返回 `resume/version_config_evidence_command_match`，从游标 9 续跑到 17。
- 恢复结果哈希与全量回放均为 `9c18598a...fac12c`。
- 故意使用错误基准配置哈希请求恢复，返回 `full_recompute/base_config_hash_mismatch`。

完整机器可读证据：`reports/equivalence.json`、`reports/precision.json`、`reports/diagnostic.json`、`reports/recovery.json`、`reports/demo_summary.json`、`reports/environment.json`、`reports/gui_smoke.json`、`reports/tests.log`。
