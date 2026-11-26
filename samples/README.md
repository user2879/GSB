# 本地样本

- `session.json`：两个训练分片、三个检查点、五个批次和七条事件。事件流包含两次版本回滚（2 → 1 → 2）、一次必定被拒绝的旧版本续约以及一次取消。
- `session_modified.json`：仅将 `shard-a` 的 `a-embed.scale` 从 1000 改为 2000，用于失效传播对比。

固定端口为 39101/39102。运行前可用以下命令检查；若输出非空，说明端口已被占用，应复制样本并改成其他高位端口。

```bash
ss -ltn '( sport = :39101 or sport = :39102 )'
```

```bash
python3 -m gsb.cli validate --config samples/session.json
python3 -m gsb.cli run --config samples/session.json --ledger /tmp/gsb-ledger.sqlite
python3 -m gsb.cli rebuild-check --config samples/session.json --ledger /tmp/gsb-ledger.sqlite
python3 -m gsb.cli invalidate \
  --old samples/session.json \
  --new samples/session_modified.json
```

分文件模式使用严格的原始行号定位；仓库测试中已有最小分文件样本构造，用户可把顶层 `shards`、`checkpoints`、`batches`、`events` 分别保存为数组文件：

```bash
python3 -m gsb.cli validate \
  --shards shards.json \
  --checkpoints checkpoints.json \
  --batches batches.json \
  --events events.json \
  --global-budget 4
```

