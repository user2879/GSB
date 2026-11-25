# 本地样本

- `../configs/spec.json`：18 个分钟点的基础样本，适合演示滚动追加。
- `spec.full.json`：24 个分钟点的完整序列。
- `request_batches/batch-001.json`：追加第 18、19、20 分钟。
- `request_batches/batch-002.json`：追加第 21、22、23 分钟。
- `../configs/invalid-spec.json`：用于验证 `E_SPEC`，首个非法字段是 `lease_policy.renew_interval_ms`。

重新生成全部确定性样本：

```bash
python -m scale_review.cli init-sample
```
