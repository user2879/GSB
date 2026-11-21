# 倾斜注入参数说明

样本由 `hashlot/samplegen.py` 使用 NumPy `default_rng(seed)` 本地确定性生成，
不访问网络或外部账号。默认参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `seed` | 20261122 | NumPy 随机数生成器种子，固定后样本逐次一致 |
| `n_events` | 480 | 事实表 `events` 行数 |
| `n_sessions` | 24 | 维度表 `sessions` 行数，也是分组键基数 |
| `scale` | 1000 | 会话定点缩放因子，浮点向零截断为 int64 |
| `hot_sessions` | 0, 1, 2 | 三个倾斜分组键 |
| `hot_weights` | 0.28 / 0.18 / 0.12 | 三个热点键的抽样权重（合计 0.58） |

## 倾斜如何触发引擎关键路径

- 热点会话 0/1/2 承载约 58% 的事件行，经 FNV-1a 哈希后集中落入少数分区。
- 分区内行数达到 `hotspot.row_threshold`（默认 90）即判定为热点分区，
  其行按二级哈希再分发到 `subpartition_count`（默认 4）个子分区。
- 同一分组键经固定规则恒映射到唯一分区与唯一子分区，子分区分组键集合互不相交。
- 分区 0、1 独占连接数为 0，必须分别向分区 6、7 借用连接，
  从而真实建立借用账、归还截止与水位约束。
- `global_max_connections=4` 制造全局并发瓶颈，触发排队等待与单调时钟计时。

## 重新生成

```bash
python -m hashlot.samplegen --out samples/skew_sample.json --seed 20261122 --rows 480
```
