# 实际验证记录

验证日期：2026-11-22  
工作目录：`/home/zsl/feedback/uidfc5909d3-default-80`

## 环境

```text
Python 3.11.0rc1
NumPy 1.26.4
pytest 8.3.3
执行平台：Linux/WSL 兼容环境
Tkinter：当前解释器未安装（No module named 'tkinter'）
```

Tkinter UI 代码已编译，纯视图模型已测试。当前容器无 Tk 模块，无法在此环境启动
桌面窗口；Windows 官方 Python 启用 `tcl/tk and IDLE` 后使用 `qos-fixed-diag ui`。
缺少 Tk 时 UI 启动会给出明确环境提示，不影响包 API、CLI、UDP、账本和重放。

## 自动化测试

命令：

```text
PYTHONPATH=src pytest -o addopts='' -q
```

结果：

```text
...........................                                              [100%]
28 passed in 5.31s
```

覆盖的验收点包括：

- 非法策略以单个 `E_POLICY` 聚合拒绝；
- 环、多上游、重复 ID、越界端口/速率、错误类型；
- scale 0..6、tick、令牌饱和、向上取整与误差推导；
- 三级严格 FIFO 排队（新包不得绕过队首）、队列上限、丢弃；
- `127.0.0.1` 真实 UDP 逐级转发及最终 sink 字节数一致；
- 误差累计、贡献占比、主导级、规范化不可区分判定；
- SQLite 批次事务、强杀 marker、unknown、恢复重跑；
- 恢复路径与不中断路径的决策序列、汇总和账本摘要完全一致；
- 快照摘要校验、整体替换、只读备份、快照账本原子物化；
- 重放成功、首个字段差异定位、不可复现分类；
- UI 视图稳定排序与三场景重复运行一致；
- 子进程 `os._exit(137)` 后重启。

## 三场景演示

命令：

```text
PYTHONPATH=src python -m qosfdiag demo --output artifacts/demo
```

实际输出：

```text
QoS fixed-point diagnostic demo
SCENARIO single_stage
  policy_version=single-stage-v1
  generated=24 released=24 released_bytes=11860
  bytes_error_bound=24000 avg_delay_ns=0 delay_error_bound_ns=0
  dominant_stage=access
SCENARIO three_stage
  policy_version=three-stage-v1
  generated=40 released=25 released_bytes=21771
  bytes_error_bound=30700 avg_delay_ns=31594936 delay_error_bound_ns=6000000
  dominant_stage=edge
  indistinguishable=true bytes_diff=0 combined_bound=61424
SCENARIO snapshot_restore_replay
  snapshot=three-stage.json
  applied=true replaced_fields=cascade_graph,error_conclusions,ledger,policy_set,scale,shaper_set
  backup_read_only=true
  restored_ledger_cursor=40 decisions=184
  replay_ok=true events_compared=184 classification=REPRODUCIBLE
SNAPSHOTS single-stage.json,three-stage.json
DONE
```

生成文件：

```text
artifacts/demo/summary.json
artifacts/demo/single_stage.sqlite
artifacts/demo/three_stage.sqlite
artifacts/demo/restored_three_stage.sqlite
artifacts/demo/snapshots/single-stage.json
artifacts/demo/snapshots/three-stage.json
artifacts/demo/snapshots/backups/backup-single-stage-v1-*.json
```

两次独立运行的三场景文本摘要完全一致，快照列表稳定排序。

## 真实强杀与恢复

命令一（批次 2 marker 提交后立即 `os._exit(137)`）：

```text
PYTHONPATH=src python -m qosfdiag run \
  --policy examples/data/three_stage_policy.json \
  --generator examples/data/three_stage_generator.json \
  --scale 3 --ledger /tmp/crash/ledger.sqlite \
  --in-memory --crash-after-marker-batch 2
```

实际结果：

```text
exit code = 137
stdout bytes = 0
stderr bytes = 0
没有生成退出报告
```

命令二（同一账本重启）：

```text
PYTHONPATH=src python -m qosfdiag run \
  --policy examples/data/three_stage_policy.json \
  --generator examples/data/three_stage_generator.json \
  --scale 3 --ledger /tmp/crash/ledger.sqlite --in-memory
```

实际结果：

```text
generated=40 released=25 bytes=21771
bytes_error_bound=30700 avg_delay_ns=31594936 delay_error_bound_ns=6000000
dominant_stage=edge
ledger_cursor=40 batches=10 digest=81429c9dcebbac91f0e76ec5452914243771093510db1f0d68bed464de5dadda
recovered_unknown=p00000008,p00000009,p00000010,p00000011
```

恢复后 SQLite 状态：

```text
decisions=184
unresolved_unknown=()
cursor=40
committed_batches=10
unknown_history:
  p00000008 resolved=1 resolved_batch=2
  p00000009 resolved=1 resolved_batch=2
  p00000010 resolved=1 resolved_batch=2
  p00000011 resolved=1 resolved_batch=2
```

另以自动化测试比较“不中断”与“强杀恢复”两条路径：

```text
same_content_digest=true
same_summary=true
same_events=true
```

## 真实回环 UDP

三级策略在 `127.0.0.1` 高位端口上执行，12 个包全部经三级实际
`sendto/recvfrom` 后到达最终 sink：

```text
ports: core=56300 edge=56301 tail=56302
final sink datagrams=12
final sink bytes=10223
transport datagrams sent=48
transport datagrams received=48
```

计数关系符合线性三阶级联：

```text
12 个根入包 + 12 次 edge->core + 12 次 core->tail + 12 次 tail->sink = 48 次发送
```

最终 sink 收包数与放行包数、字节数与最终放行字节数完全一致。

## Wheel 构建

命令：

```text
python -m pip wheel . --no-deps -w /tmp/qos-wheel
```

结果：

```text
Successfully built qos-fixed-diag
qos_fixed_diag-1.0.0-py3-none-any.whl
size=57337 bytes
sha256=103dd9a578f0394708a1993ecf64f5e1032a95f05466278469b71eb1819a0cd3
```

另将该 wheel 安装到临时 `--target` 目录并验证 `qosfdiag.__version__`、
`load_policy` 和三级拓扑可正常导入。

## 重放差异定位

自动化测试篡改快照中首个事件的 `tokens_after` 后重新计算快照内容摘要。重放按
稳定键定位并报告：

```text
code=E_REPLAY
classification=LEDGER_DEFECT
packet_id=p00000000
shaper_id=edge
field=tokens_after
```

策略版本不一致测试报告：

```text
reason=policy_version_mismatch
classification=UNREPRODUCIBLE
first_mismatch absent
```

正常快照重放结果：

```text
events_compared=184
classification=REPRODUCIBLE
```

## Tkinter 限制

当前解释器执行 `import tkinter` 得到 `ModuleNotFoundError`，因此不能在本机完成
窗口级截图验证。已完成的验证是：

- `src/qosfdiag/ui.py` 通过 `compileall`；
- UI 不依赖 Tk 的 `build_view` 视图模型被测试，拓扑、统计、时间线、误差热力、
  不可区分结果和快照列表顺序稳定；
- `launch_ui` 在未安装 Tk 时抛出包含安装建议的 `RuntimeError`。

在 Windows 官方 Python 安装器勾选 `tcl/tk and IDLE` 即可启动
`qos-fixed-diag ui`。
