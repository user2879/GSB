# 实际验证记录

验证时间：2026-11-27（Asia/Shanghai）。

## 环境

系统缺少 `python3-venv` 的 `ensurepip` 组件且无免密 sudo，因此使用标准库先创建无 pip 虚拟环境，再用 `get-pip.py` 引导：

```bash
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install numpy pytest
```

实际版本：

```text
Python 3.11.0rc1
numpy 2.4.6
pytest 9.2.0
```

Python 3.11 满足“Python 3.10 及以上”。

## 自动化测试

命令：

```bash
.venv/bin/python -m pytest -q
```

实际结果：

```text
...........................                                              [100%]
28 passed in 5.83s
```

测试覆盖：

- `E_SPEC`：重复节点/制品、非 `127.0.0.1`、重复端口、非法正整数、事件序号、分片连续区间和越界。
- 确定性字节/摘要绑定与跨节点归属排序。
- 状态机、`E_GEN` 跳代/回退和代数保留。
- 全局 2、单节点 1 的真实线程池在途上限。
- `E_TIMEOUT`、`E_FIELD`、`E_PORT` 和真实节点 `E_STALE`。
- 任意事件前缀/分块重放逐字节稳定。
- 配额/节点移除演进的影响子集和 `E_DIVERGE` 定位。
- 真实 TCP 端到端运行。
- `SIGKILL` 强杀、未知在途请求回收、账本恢复和重复重放逐字节一致。
- 篡改快照摘要后只读重放返回 `E_VERIFY`，并定位到 `digest` 字段。

## 一键验证

```bash
./scripts/verify.sh
```

最近一次完整输出目录：

```text
artifacts/validation/20261127-020259/
```

该脚本退出码为 `0`，其中再次记录 `28 passed in 5.83s`。

## 固定样本结果

命令：

```bash
.venv/bin/python -m cacheflow validate
.venv/bin/python -m cacheflow run --work-dir artifacts/validation/fixed-sample/work
.venv/bin/python -m cacheflow replay \
  --ledger artifacts/validation/fixed-sample/work/cacheflow.sqlite3 \
  --out artifacts/validation/fixed-sample/replay.canonical.json
```

规格摘要：

```text
23db9f2a6007c82115fb4c025223aac0b1f56f21f5336f9eb01dec68af916fd7
```

运行峰值：

```json
{"global": 3, "nodes": {"node-a": 2, "node-b": 1, "node-c": 1}}
```

29 条等待记录均含单调时钟开始/结束值，最大等待约 `316 ms`。

规范结论文件与独立只读重放文件的 SHA-256 完全相同：

```text
b913269070ea860b06d37f2fc4f9039b8b19f9f67b048e2c995f86d41ee10a6d
  artifacts/validation/fixed-sample/work/conclusion.canonical.json
b913269070ea860b06d37f2fc4f9039b8b19f9f67b048e2c995f86d41ee10a6d
  artifacts/validation/fixed-sample/replay.canonical.json
```

结论中：

- `uniqueness.unique_holding_ok = true`
- `lease-beta` 为 `expired/generation=2`，回收类型 `expiry`
- `lease-gamma` 为 `released/generation=2`
- `lease-alpha` 配额换代后为 `expired/generation=4`，回收类型 `quota_invalidation`
- `lease-alpha2` 为新配额版本下 `granted/generation=1`
- 最终 `art-alpha` 唯一归属 `node-c`，3 个分片占用 slot `0,1,2`

## 强杀验收

命令：

```bash
.venv/bin/python -m cacheflow accept --work-dir artifacts/validation/20261127-020259/acceptance
```

实际关键结果：

```json
{
  "bytewise_repeat": true,
  "unique_holding_ok": true,
  "steps": [
    {"step": "normal_pipeline", "wait_record_count": 29},
    {
      "step": "force_kill_inflight",
      "node_id": "node-a",
      "forced_returncode": -9,
      "blocked_request_error": {"code": "E_EOF"},
      "node_exit_report_promised": false
    },
    {
      "step": "unknown_marked_and_quota_reclaimed",
      "status": "unknown_rerun_required",
      "hold_ids": ["hold:lease-crash-rerun:0:1"]
    }
  ]
}
```

强杀后重启的 node-a 从账本恢复到已提交状态，例如：

```text
lease-beta  state=expired generation=2 recovered=true
lease-delta state=expired generation=2 recovered=true
```

未提交请求先被标记为 `unknown_rerun_required` 并回收 provisional slot；重跑授予提交后，最终容量唯一性仍为 `true`。

强杀后只读重放报告的内部规范摘要为：

```text
188bd9c7a9ae9a9474ff76c3c34e4fcabc65d360c66a06a2e250c46797fedf82
```

其完整规范字节文件的 SHA-256 为：

```text
932b132c78bb6d34214b3e934322a5ea0ce47bf28d6da3cae87dbc5c94a61bd6
  post-kill-replay.canonical.json
```

连续两次只读重放得到相同规范字节，报告字段 `bytewise_repeat=true`。

## 演进校验

配额前缀演进：

```bash
.venv/bin/python -m cacheflow evolve --cutoff-seq 20
```

结果：4 个活跃租约全部受影响，增量结果与空白全量重建一致，摘要：

```text
05ea8d2bda2eaa765a2348aa6196c88809871e995307c0b46abc05ed45c0487e
```

节点移除子集：

```bash
.venv/bin/python -m cacheflow evolve \
  --evolution examples/evolution-node-removal.json \
  --cutoff-seq 24
```

结果只影响 `lease-alpha` / `node-c`，`lease-beta`、`lease-delta`、`lease-gamma` 为未受影响租约且规范字节保持不变；增量与全量重建一致，摘要：

```text
19949910e957dffc46e27cf8a5bfe4ec317fa68a4c1d3b3b5b768adb1caefc5b
```

## 约束确认

- 未使用 Docker 或其他容器。
- 所有节点均绑定 `127.0.0.1`，端口由本机高位空闲端口或示例固定高位端口提供。
- 只访问本地回环和工作目录。
- 未依赖外部服务或账号。
- 验证后未发现遗留 `cacheflow.node` 子进程。
