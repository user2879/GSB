# 使用指南

## 准备环境

```bash
# 本机 python3.11 -m venv 缺 ensurepip，使用 virtualenv 等价创建
virtualenv -p python3.11 .venv
source .venv/bin/activate
pip install -r requirements.lock      # numpy==1.26.4 pytest==8.2.0
pip install -e .                      # 注册 dispatch-tokens 命令
```

## CLI 命令

```bash
# 1. 规格强校验（失败返回 E_SPEC，含首个违规 location/field）
python -m dispatch_tokens validate samples/demo_spec.json

# 2. 离线确定性调度（不起子进程）
python -m dispatch_tokens run samples/demo_spec.json \
  --ledger artifacts/demo.offline.db \
  --report artifacts/demo.offline.report.json --offline

# 3. 真实回环子进程下发
python -m dispatch_tokens run samples/demo_spec.json \
  --ledger artifacts/demo.live.db \
  --report artifacts/demo.live.report.json

# 4. 增量 vs 空白全量重建（E_DIFF）
python -m dispatch_tokens verify samples/demo_spec.json \
  --report artifacts/demo.live.report.json

# 5. 账本恢复重放（E_REPLAY）
python -m dispatch_tokens replay samples/demo_spec.json \
  --ledger artifacts/demo.live.db
```

## 一键演示

```bash
bash scripts/demo.sh
```

## 规格文件段落

- `nodes`：`node_id/host(127.0.0.1)/port(1-65535 唯一)/rate/burst/
  bucket_key/seed/param_version`，均为非负整数。
- `shards`：`shard_id/upstream/sample_rows/weight/priority`，依赖无环。
- `revisions`：`version` 单调递增、`parent` 引用存在且单根无环，
  `scale` 仅允许 1/10/100/1000。
- `events`：`dispatch/renew/cancel/revise/exhausted`，按 `seq` 到达序、
  `ts_ms` 单调非降；续约需带 `expected_version`。
