# 实际验证结果

验证时间：2026-11-27（Asia/Shanghai）

环境：

- WSL Ubuntu 22.04.5 LTS
- Python `3.11.0rc1`
- NumPy `1.26.4`，安装在项目 `.venv`
- 未使用 Docker；节点均为本机 `127.0.0.1` 子进程

复现命令：

```bash
bash scripts/setup_venv.sh
bash scripts/acceptance.sh 2>&1 | tee artifacts/acceptance-output.txt
```

最近一次完整运行退出码为 `0`。完整终端输出保存在 `artifacts/acceptance-output.txt`，机器可读摘要保存在 `artifacts/acceptance-summary.json`。

## 自动化测试

```text
Ran 15 tests in 1.320s

OK
```

覆盖：

- 声明强校验、重复端口、环依赖和 `E_SPEC` 字段路径。
- 整数边界判断和候选误差向量。
- 容量版本失效推导。
- 分片资源需求变更、节点配额变更的直接/间接失效推导。
- 真实单调时钟过期与版本作废竞争只释放一次权益。
- 续约上限和 `E_LEASE`。
- SQLite 摘要不符整体替换。
- 未完成记录恢复为 unknown 并回收配额。
- 真实 TCP 租约 CAS 与节点容量版本推进。
- 双节点、真实子进程、线程池端到端调度。

## 网络异常注入

`artifacts/network-result.json` 实测：

- 回环端口 bind → close → rebind 成功。
- 占用端口启动节点得到 `E_PORT`，首个字段为 `nodes.net.port`。
- 500ms 延迟节点配合 100ms socket timeout 得到 `E_TIMEOUT`。

## 定点误差和边界裁决

样例 `Q=10`，全局预算 `6 RU = 60 q`。实测三级贡献：

```json
{
  "contributions_q": {"shard": 9, "node": 2, "global": 2},
  "share_basis_points_floor": {"shard": 6923, "node": 1538, "global": 1538},
  "dominant_level": "shard",
  "dominance_threshold_bp": 5000
}
```

关键裁决：

- `admit-3`：预算差 `G_q=0`，合成界 `E_q=9`，预算边界不可区分。
- 同次候选水位：`|10 - 0| q = 10 q`，配对误差界 `14 q`，水位不可区分；按 `(topological_rank, shard_id)` 放行 `s3`。
- `admit-5`：`s4` 预算差 `0 q`、误差界 `11 q`，同样标注受误差影响。
- 精确整数账本中的峰值在途需求为 `60/60 q`，`budget_never_exceeded_exact_integer=true`。

完整两候选误差向量、残差分子/分母和节点投影见 `artifacts/run-result.json` 的 `decisions[].watermark_pair`。

## 重建一致性

`artifacts/rebuild-check.json` 实测 `ok=true`，6 个分片的以下字段全部一致：

- `aggregate_summary`
- `watermark`
- `lease_ownership`

账本 `content_digest` 校验为 `ok=true`，且存储摘要与重建摘要一致。

## 容量版本替换

将 `n1.capacity_version` 从 `1` 增至 `2` 后，实测：

```json
{
  "direct_shard_ids": ["s1", "s3", "s5"],
  "invalid_shard_ids": ["s1", "s3", "s5", "s6"],
  "unchanged_shard_ids": ["s2", "s4"]
}
```

传播链包含 `s3 -> s5 -> s6` 的依赖边。`artifacts/evolve-result.json` 中：

- `evolution_ok=true`
- `byte_preservation_after_run.ok=true`
- `rebuild_consistency.ok=true`
- `s2`、`s4` 的 `summary_json`、`lease_ownership_json`、`watermark_json` 原始字节全部 `byte_identical=true`。

租约 CAS、旧版本租约撤销以及到期/作废竞争由 unittest 和 `artifacts/unittest-results.txt` 验证。

## 强制终止和恢复

崩溃声明在首个结局事务提交后向主进程发送 `SIGKILL`，验收脚本实测：

```text
force_kill_exit_code=137 (137 expected)
```

强杀后 SQLite 中存在：

- `s2=completed`
- `s1=running`
- `s3=running`

重启带 `--resume` 后，`s1`、`s3` 被标记为 `unknown_uncommitted_outcome`，先写配额回收记录，再重跑。恢复结果：

```json
{
  "recovery_ok": true,
  "reclaimed": ["s1", "s3"],
  "recovery_rebuild_ok": true
}
```

恢复后 6 个分片全部 completed，预算精确约束保持，重建校验通过。强杀进程未生成 `artifacts/crash-result.json`；恢复记录保留警告 `no force-termination exit report can be reconstructed`。
