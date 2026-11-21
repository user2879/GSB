# 实际验证记录

验证日期：2026-11-21。以下结果由最终版本代码实际执行产生。

## 环境

- Python 3.11.0rc1；工作区 `.venv` 由 local virtualenv 创建，`--system-site-packages`。
- numpy 1.26.4、PyYAML 6.0.2、pytest 8.3.3。
- 未使用 Docker/容器、未联网下载模型或数据。
- `.venv/bin/python -m pip install -e . --no-build-isolation --no-deps` 已完成本地可编辑安装。

## 自动化测试

```text
$ .venv/bin/python -m pytest -q
...........                                                              [100%]
13 passed in 5.36s
```

覆盖：加载校验错误码与继续处理、模型维度错误的报告化处理、参数形状、定点逐位确定性、真实哈希冲突、维特比并列、路径复算、BIESO 修复与类型并列、预算配额、LRU 淘汰/重算、并发一致性、四类注入、优雅停止续跑、原子快照与陈旧拒绝。

## 样本与完整运行

生成样本：32 个、741 字符、最长 30 字符、全局预算 489、缓存 5 块、字块 4。

```text
$ .venv/bin/ner-workbench run --samples samples --out runs/final-workers1 --workers 1
$ .venv/bin/ner-workbench run --samples samples --out runs/final-workers4 --workers 4
$ .venv/bin/ner-workbench compare \
    runs/final-workers1/report.json runs/final-workers4/report.json \
    --out runs/final-comparison-workers.json
```

结果：

- 两次确定性摘要均为 `c5c3ad595c99304511dfa4247045d1a5866dab88e54675636035e352ec66c9fa`，比较 `equal=true`、差异为空。
- 17 个成功样本、15 个终态失败、148 个实体、35 条修复记录。
- 消耗 489/489 字符预算，触发全局预算耗尽。
- 缓存 216 次淘汰、103 次淘汰后重算，重算校验无非一致。
- 327 次实际哈希冲突。
- 4 条注入全部命中：`E_NONFINITE_SCORE`、`E_LABEL_LENGTH_MISMATCH`、`E_PARAM_SHAPE_MISMATCH`、`E_SAMPLE_QUOTA_EXCEEDED`。
- 自包含面板：`runs/final-workers1/dashboard.html`、`runs/final-workers4/dashboard.html`。

## 优雅停止与续跑

```text
$ .venv/bin/ner-workbench run --samples samples --out runs/final-resume --workers 1 --stop-after 12
$ .venv/bin/ner-workbench run --samples samples --out runs/final-resume --workers 1 --resume
$ .venv/bin/ner-workbench compare runs/final-workers1/report.json runs/final-resume/report.json
```

停止时快照完成前缀 12 个样本，恢复点 `SMP-0012`，已消耗 286 字符；续跑后确定性投影与完整单线程运行完全一致。

## SIGTERM

```text
$ .venv/bin/ner-workbench run --samples samples --out runs/final-signal \
    --workers 2 --delay-ms 60 &
$ kill -TERM <pid>
```

进程退出码 0。快照前缀 9 个样本，恢复点 `SMP-0009`，已消耗 213 字符；续跑后与完整运行完全一致。

## SIGKILL 与强制恢复

```text
$ .venv/bin/ner-workbench run --samples samples --out runs/final-kill \
    --workers 2 --delay-ms 60 &
$ kill -9 <pid>
```

进程退出码 137。`CURRENT` 仍为完整符号链接，指向 `versions/v00000004`；该版本状态为 `running`，完成前缀 12 个样本，恢复点 `SMP-0012`，已消耗 286 字符。执行 `--resume` 后与完整运行零差异。未提交的半写临时/新版本目录不可由 `CURRENT` 到达。

## 陈旧快照拒绝

对并发 1 生成的快照以并发 2 续跑：

```text
$ .venv/bin/ner-workbench run --samples samples --out runs/final-resume --workers 2 --resume
exit code: 2
code: E_SNAPSHOT_STALE
message: config hash mismatch
```

机器可读汇总见 `artifacts/validation_summary.json`。
