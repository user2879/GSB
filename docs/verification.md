# 实际验证记录

验证日期：2026-11-27（Asia/Shanghai）

## 环境

```text
Python 3.11.0rc1
pip 20.3.4
numpy 1.26.4
Tkinter: ImportError: libtk8.6.so 缺失；当前用户 uid=1000，未安装 python3-tk
```

核心验收路径为无头进程/套接字/SQLite，不需要 Tkinter。`gsb gui` 是可选面板，缺依赖时返回 `E_UI`。

## 自动化测试

命令：

```bash
python3 -m unittest discover -s tests -v
```

结果：

```text
Ran 30 tests in 5.846s
OK
```

覆盖点：

- JSON 语义错误的首个非法字段/来源行号、DAG 环、未知字段、分文件加载。
- 参数摘要稳定、样本顺序无关、32 个并发任务逐位一致。
- 两次回滚的世代号推进、批次记录顺序不影响检查点规范字节。
- 租约乐观版本冲突不延长到期、真实单调时钟过期回收额度。
- 全局/分片/租约三重配额高水位、真实 `gsb-batch-*` 线程、误差决胜标记。
- 真实 TCP 回执、端口占用 `E_PORT`、握手超时 `E_TIMEOUT`、字段缺失 `E_FIELD`。
- SQLite 快照身份/内容摘要整体替换、SIGKILL 后 `unknown` 恢复、租约回收、重跑、`E_REPLAY` 定位。
- 配置变更后的失效子集、未失效批次字节一致性、账本与空白全量重建一致。

## 样本端到端运行

命令：

```bash
python3 -m gsb.cli run \
  --config samples/session.json \
  --ledger <temporary>/ledger.sqlite
python3 -m gsb.cli rebuild-check \
  --config samples/session.json \
  --ledger <temporary>/ledger.sqlite
```

关键结果：

```text
run ok: true
batches committed: 5
checked batches: 5
rebuild-check consistent: true
checkpoint_count: 3
current_versions: {"shard-a": 2, "shard-b": 7}
generations: {"shard-a": 2, "shard-b": 0}
quota high watermarks: global 3 / budget 4,
  shard-a active 1 / limit 2,
  shard-b active 1 / limit 1
events: commit=true, rollback=true, commit=true,
  stale renew=false (E_LEASE), cancel=true
```

事件流执行两次回滚：版本 2 → 1 → 2。三次 `shard-a` 批次分别绑定版本 2、1、2，最终世代号为 2。样本中的过期租约续约被拒绝，用于验证 `E_LEASE` 路径。

## 失效传播

命令：

```bash
python3 -m gsb.cli invalidate \
  --old samples/session.json \
  --new samples/session_modified.json
```

关键结果：

```text
shard-a change: layer:a-embed:scale
invalid batches: a-before-rollback, a-after-rollback, a-after-second-rollback
invalid checkpoints: cp-a-base, cp-a-v2
unaffected batches: b-before-rollback, b-after-rollback
unaffected checkpoint: cp-b-base
unaffected byte_identical: true
report_digest: 8c22260c9d511ef90241808d344598821f8bffedfc0f1d9e28906a4139d1c689
```

`a-after-rollback` 通过依赖链保持稳定排序；`shard-b` 未修改，其损失摘要和检查点声明字节不变。

## 已知环境限制

- 当前环境未安装 Tk 共享库，因此未执行 GUI 手工点击验收。
- 样本端口固定为 39101/39102；若被占用，按设计返回 `E_PORT`，测试使用动态空闲端口。
- 本实现只在回环地址工作，不访问外部目标。
