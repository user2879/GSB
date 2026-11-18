# 实际验证记录

验证日期：2026-11-18（Asia/Shanghai）

## 环境

- Python：3.11.0rc1（满足 3.11+ 核心语法与标准库要求）
- SQLite：3.37.2
- 平台：Linux WSL2（本地路径、无网络访问）
- pytest：当前基础环境未安装，因此全部测试使用标准库 `unittest`
- Tkinter：当前 Linux Python 未安装 `tkinter`，无法在本机打开窗口；`quota_app.gui` 导入、无 Tk 降级和语法编译已验证。Windows 官方 Python 3.11+ 自带 Tkinter，目标平台可直接运行 `python -m quota_app gui`

## 执行命令

```bash
python3 scripts/generate_samples.py --output data
python3 -m unittest discover -s tests -t . -v
python3 scripts/run_verification.py
python3 -m py_compile quota_app/*.py scripts/*.py tests/*.py
```

完整机器可读结果保存在 `artifacts/verification.json`。

## 结果总览

| 项目 | 结果 | 实际输出摘要 |
| --- | --- | --- |
| 单元/集成测试 | 通过 | `Ran 48 tests`，`OK`，退出码 0 |
| v1/v2 自动升级 | 通过 | v1、v2 均重写为 v3，活动修订为 3，生成两个只读备份 |
| 派生与仲裁 | 通过 | `child`: 显式 700 / 派生 1100，生效 700；`mirror`: 显式 5000 / 派生 1000，生效 1000 |
| 快速预览与权威重放 | 通过 | 10/10 事件一致，单事件/聚合误差上界均为 0 |
| 判定分布 | 通过 | `ALLOW=6`、`RATE_LIMITED=2`、`OVER_QUOTA=2` |
| SQLite 批次提交 | 通过 | 4 批、10 条判定、进度游标 10 |
| 账本重放 | 通过 | 逐项比较 10/10，无差异 |
| 强制中断恢复 | 通过 | 遗留 1 个 `unknown` 批次，`exit_report_available=false`，重跑后 10 条判定可用且重放一致 |
| 异常注入 | 通过 | `E_FORMAT`、`E_UPGRADE`、`E_ORDER` 均命中 |
| CLI 端到端 | 通过 | `run` 返回 `OK`，4 批/10 判定；`recover` 重放通过 |
| 脚本编辑 | 通过 | `edit --ingest` 生成修订 4，旧修订标记 STALE，当前视图整体替换并重放 10/10 |
| Tkinter 模块 | 受环境限制 | 模块导入成功；本机无 Tkinter，未执行窗口启动 |

## 关键实测细节

### 升级与检查点

- `config-v1.json` 与 `config-v2.json` 在内存中完整校验后升级为当前 v3 编码。
- 备份路径为 `backups/config-v1.v1.bak`、`backups/config-v2.v2.bak`。
- 每个条目写完后原子更新 `upgrade_checkpoint.json`。
- 测试用子进程在第一个文件重写后执行 `os._exit(86)` 强制终止；重启后识别为 `recovered_completion`，未重复重写，随后继续升级第二个条目。
- 检查点缺失时低版本条目被阻断并返回 `E_UPGRADE`；不可读旧条目报告 `E_UPGRADE` 后其他条目继续。

### 判定样本

样本事件 1-10 的判定序列：

```text
ALLOW, ALLOW, RATE_LIMITED, ALLOW, OVER_QUOTA,
ALLOW, ALLOW, OVER_QUOTA, ALLOW, RATE_LIMITED
```

容量恰好等于上限的测试通过；令牌恰好等于成本时通过。令牌按 1/1000 单位整数存储，`100ms * 2 ops/s = 200` 千分单位，与连续模型无余数，故量化误差上界为 0。

### 账本事务与恢复

- 批次先以独立事务写入 `pending`。
- 判定、事件输入、当前用量、进度游标和 `committed` 状态在同一事务提交。
- 在事务提交前注入运行时中断后，`judgments`、`current_judgments`、`event_inputs`、`current_usage` 均为 0 行，进度仍为 0；仅遗留 `pending` 批次。
- 重启后该批次变为 `unknown`，明确返回 `rerun_required=true` 和 `exit_report_available=false`。
- 重跑后 4 批全部提交，账本输入重放 10/10 一致。
- 另有人工篡改 `current_judgments` 的测试，校验器返回 `E_REPLAY`，并给出卷 `root`、事件 1 和 `decision` 差异字段。
- 人工篡改 `current_usage.cumulative_write_bytes` 同样返回 `E_REPLAY`；游标大于 0 但用量快照缺失时拒绝继续追加批次。

### 异常注入实测

| 注入 | 业务码 | 依据摘要 |
| --- | --- | --- |
| 配置 JSON 截断 | `E_FORMAT` | `invalid JSON: Expecting value`，字段路径含行/列 |
| 删除升级检查点且存在旧条目 | `E_UPGRADE` | `upgrade checkpoint is missing while legacy entries remain` |
| 事件时间戳 10ms 后退到 9ms | `E_ORDER` | 卷 `root`，事件 `seq=2`，`field_path=$[2].timestamp_ms` |

三种注入都在临时目录副本上执行，未修改交付样本。日志以 JSONL 写入，含 `timestamp_utc`、`code`、`operation`、`volume_id`、`field_path` 和 `details`。

### 版本整体替换

测试把样本修订从 3 编辑到 4，并修改 `child.capacity_bytes` 与 `mirror.ops_per_sec`：

- 修订 3 的 10 条历史判定全部置 `stale=1`。
- 当前判定、当前用量和进度游标被清空，不发生增量合并。
- `config_replacements` 稳定记录受影响卷 `["child","mirror"]` 及字段变化。
- 上游卷容量变化时，下游卷的 `effective_capacity_bytes`、`arbitration` 变化也会被列为受影响字段。
- 修订 4 重新重放后 10 条当前判定可复现。

## 未执行项

- 当前 Linux 环境无 `tkinter` 和图形显示，未实际打开 Tkinter 窗口、拖动曲线或点击表单。
- GUI 相关核心（快速预览、权威线程所用服务、延迟/差异日志、稳定排序数据查询）已通过无界面测试；Windows 交互窗口需在目标机按 `README.md` 的命令验收。
