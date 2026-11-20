# 实际验证结果

验证环境：Python 3.11.0rc1、NumPy 1.26.4、Linux(WSL Ubuntu-22.04)。
代码仅使用跨平台 API，Windows 终端命令相同；本环境无 tkinter，
GUI 入口验证为优雅降级（打印提示、退出码 2），核心逻辑全部经
无头路径验证。

## 单元/集成测试（python -m unittest discover -s tests）

```
Ran 21 tests in 17.928s
OK
```

覆盖：E_FORMAT（未知版本/缺字段/重复键/不支持类型/派生引用）、
三版本归一与只读备份、归一强杀续做（子进程强杀后按检查点续做）、
优先级并发下发与配额上限、E_CONVERGE 偏差定位、节点强杀重启恢复、
E_PORT、E_TIMEOUT 与重试次数、E_PROTO、账本强杀恢复、E_REPLAY
篡改定位、E_PREVIEW、陈旧标记与整体替换、延迟统计与 E_LATENCY。

## 端到端验收（python -m configdist.cli demo）

```
ACCEPTANCE: PASS
```

关键观测（单次运行，可复现）：

- 三版本归一：app_v1.json v1→v3（4 条）、app_v2.json v2→v3（5 条）、
  app_v3.json already-current；旧清单保留为只读 .bak。
- 归一续做：30 条清单升级子进程在第 6 条后被强杀，重启 resumed_from=6
  续做完成，检查点清除，备份只读。
- 优先级并发下发：17 个切片，实际开始顺序前 6 为
  (rho,node-a)(rho,node-c)(alpha,node-a)(alpha,node-b)(alpha,node-c)
  (alpha,node-d)，符合优先级降序+稳定键决胜；全局峰值 3（上限 3），
  每节点峰值 2（配额 2）。
- 回执收敛：17/17 回执与期望摘要逐字节一致，无 E_CONVERGE。
- 界面可交互：下发耗时约 153ms 期间完成 14 次界面查询，
  观测到运行中峰值 3；ui_query P95=0ms、recompute P95=2ms，
  量化上界 ±0.5ms，无 E_LATENCY。
- 权威重算：v5 与编辑后 v6 两个版本均与快速结论逐字节一致（match=true）。
- 陈旧防护：编辑后 17 条旧结论全部标记 stale 且当前视图为空；
  整体替换报告稳定列出变化键（alpha、eta）与各节点旧新摘要。
- 节点恢复：node-a 强杀重启后 hello_ack 携带 applied=5 的持久化状态，
  继续应答。
- 强杀恢复：crash-run 子进程提交批次 1 后以 os._exit(9) 终止，
  恢复报告 unknown_batches=[2] 且不生成退出报告；重跑批次 3 收敛，
  账本重放与权威重算一致。
