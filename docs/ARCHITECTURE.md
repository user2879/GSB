# 架构与不变量

## 模块

| 模块 | 职责 |
| --- | --- |
| `deploy_workbench/spec.py` | 严格 JSON 读取、公式线性化、声明校验、异常注入输入生成 |
| `deploy_workbench/model.py` | 环境、源、绑定、任务及三组版本指纹 |
| `deploy_workbench/engine.py` | 逐环境逐键候选源排序、绑定裁决、链摘要、任务编排 |
| `deploy_workbench/fixed.py` | 向零截断定点量化、INT64 饱和、误差传播、归因、比较决胜 |
| `deploy_workbench/ledger.py` | SQLite 格式版本、版本元数据、环境事务、恢复、只读触发器 |
| `deploy_workbench/snapshot.py` | 快照构造、摘要、重放字段比较、首个差异定位 |
| `deploy_workbench/service.py` | 版本发布、陈旧隔离、整体替换、run 编排、快照恢复 |
| `deploy_workbench/gui.py` | Tk 编辑器、稳定排序视图、热力色、历史只读窗口 |
| `deploy_workbench/cli.py` | 本地 CLI、注入参数、结构化错误输出和日志 |

## 关键不变量

1. 规范 JSON 使用排序键、稳定分隔符和十进制定点文本；同一声明与来源集合多次裁决得到相同字节。
2. 链排序只依赖 `(type_bias, explicit_priority, source_id)`。遍历来源列表的顺序不会进入结果。
3. 生效源必须是排序链中首个满足绑定的源；绑定不会把未参与排序的源插入链。
4. 所有定点乘加使用 Python 整数；只有量化、最终整数除法和最终饱和会改变整数表示。
5. 误差界三段贡献之和等于 `error_bound`，`actual_error <= error_bound` 是提交前可测试的不变量。
6. 一个 run 只读取自己的 `environment_records`。当前视图不会把多个 run 的键拼在一起。
7. 环境记录不可更新或删除；版本变更只修改 run 的当前标记，并追加新 run。
8. 一个环境的 `environment_records` 插入与 `environment_progress=committed` 在同一事务。
9. 恢复只信任已提交记录。`computing`/`pending`/`failed` 在中断恢复中转为 `unknown`。
10. 快照重放先按环境排序、再按键排序、再按声明字段顺序比较，保证首个差异可重复定位。

## 版本指纹

版本不是文件时间戳，而是内容指纹差分的单调计数器：

```text
new_version = old_version + 1  if fingerprint changed
new_version = old_version      otherwise
```

`environment`、`priority`、`scale` 三个计数器独立递增。打开修改后的声明会持久化新计数器并撤下当前标记；重新计算使用同一计数创建新 run。因此版本不会因多次刷新倒退或重复递增。

## 事务边界

```text
begin_run transaction:
  persist version_state
  clear previous run is_current
  insert new run(is_current=1, status=running)
  insert progress rows

per environment:
  transaction A: progress=computing
  resolve + fixed-point compute (outside DB transaction)
  transaction B: insert environment_records + progress=committed

finalize:
  compute content digest from committed records
  set run complete/incomplete
  update ledger_content_digest
```

强制终止可能发生在任意阶段。恢复不会把事务 A 或内存结果当作已提交结论。

## 快照恢复决策

```text
snapshot version tuple != current version tuple
  -> rebuild all environments with a new forced run
snapshot version tuple == current version tuple
  -> resolve all in memory, no ledger mutation
  -> compare completed set, environment, key, task and fields
  -> first difference -> E_REPLAY(environment, key, field, expected, actual, basis)
```

这保证旧版本快照不会增量污染当前视图，同时同版本重放可做逐项审计。

