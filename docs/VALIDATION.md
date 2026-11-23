# 实际验证记录

验证环境：Linux 工作容器，Python 3.11.0rc1，NumPy 2.2.6；实现无平台专用依赖，Windows 10/11 使用同一 Python 代码路径。所有音频均由固定种子生成。

## 自动化测试

命令：

```text
python -m unittest discover -s tests -v
```

结果：`Ran 13 tests in 72.494s`，`OK`（最终一次全套运行；不同负载下耗时会有波动）。

覆盖点：

- 合法 WAV 往返；RIFF 截断、错误采样率返回格式错误。
- 扫描遇到两个坏 WAV 时保留两个好文件，失败清单按文件名稳定排序。
- 25 ms 短间隙合并为一个发音段，220 ms 长间隙切为两段，40 ms 短段被丢弃。
- F0、F1/F2/F3、语速与有效帧非空；帧序号升序；重复全量重建摘要一致。
- 4-bit 与 8-bit 强制档位产生不同整数码与不同描述向量。
- 阈值为 0 时强制合并导致两条 `E_ASSIGNMENT_CONFLICT`，但保留向量预测；聚合权重等于有效帧之和。
- 单段说话人输出 `aggregation_degenerate=true`。
- 相似度恰等于阈值时执行说话人标识字典序决断（含冲突与非冲突分支）。
- 先建立初始缓存再切分时，除新生成的两个半段外特征复用；随后追加 `set_assignment_metric=normalized_l1` 与 `noop`，最终 37 个当前产物全部复用、0 重算。
- 3 线程上限得到 `peak_workers_observed <= 3`。
- 预算 0 ms 时所有首响为 `E_FIRST_RESPONSE_PENDING`，摘要字段完整，第二轮摘要包含上一稳定段数。
- 3 条命令后快照恢复并续跑剩余 4 条，最终 canonical digest 与从空白强制全量一致。
- 增量/全量比较序列同时包含切分、单段音频替换、量化档位切换、阈值修改、合并、边界调整、归属规则修改与 noop。
- `--expected-version 999` 与修改基础配置均触发 `E_SNAPSHOT_STALE`。
- CLI `demo -> run -> compare` 全部退出 0，HTML 生成且 compare `equivalent=true`。

## Demo 命令流

固定样本：4 个 16 kHz 单声道 16-bit WAV，覆盖 alpha/beta/gamma 与混合文件。命令：

1. 切分 alpha 首段。
2. 替换切分后半段的音频。
3. 将 `f0_mean` 强制为 4-bit 档位。
4. 相似度阈值改为 0.95。
5. 合并前两个 alpha 段。
6. 调整 beta 第二段结束边界。
7. `noop`。

最终稳定结果：

- 12 个发音段。
- 5 个判定说话人（包括重参数化导致的 alpha 分裂组）。
- 6 个描述维度。
- 1 条标注一致性冲突记录，按 `keep_prediction_record_conflict` 保留预测。
- 4 个有效音频文件，0 个格式失败。
- 第 7 条 `noop`：37 个当前产物全部复用，0 个重算。

最终 demo 文件：

- `examples/demo/report.json`
- `examples/demo/report.html`
- `examples/demo/responses.json`
- `examples/demo/compare.json`

## 增量/全量证据

```text
python -m timbre_workbench compare \
  --input examples/demo/audio \
  --config examples/demo/config.json \
  --labels examples/demo/labels.json \
  --commands examples/demo/commands.json \
  --output examples/demo/compare.json
```

`compare.json` 的 `equivalent` 为 `true`，`differences` 为空；增量与全量的 `canonical_digest` 相同。比较包含：

本次实测：

```text
incremental_digest = 1dd38129044c26d93bc7b7fd610794be4a80bf25b57895c33accc1380ec4fb64
full_digest        = 1dd38129044c26d93bc7b7fd610794be4a80bf25b57895c33accc1380ec4fb64
incremental        = 37 reused / 0 recomputed
full               = 0 reused / 37 recomputed
```

7 条命令在 1 ms 首响预算下均返回 `E_FIRST_RESPONSE_PENDING`，最终实测首响延迟为 1.1–3.4 ms，配置的 20 ms 操作系统调度容差用于避免线程调度抖动误报；真实 `latency_ms` 始终单独记录。稳定后仍落盘最终结果。`report.json` 的 `diagnostics.first_response` 保留最后一条命令的首响记录，`responses.json` 保留全序列。

比较包含：

- 逐段特征、描述向量、归属与相似度。
- 说话人聚合向量、权重与参与者。
- 冲突、失败清单及其稳定顺序。
- 量化计划、依赖清单、当前缓存键与路径无关的失效/复用资格清单。

`diagnostics.cache_events` 只表达本次执行历史：增量运行显示命中，空白全量运行显示全部重算，这是预期差异，不属于语义产物。

## 快照证据

测试先执行 3 条命令并提交快照，随后恢复快照执行剩余命令。恢复后：

- 命令游标从 3 继续，不重复前 3 条命令。
- 快照 manifest 包含 `global_version`、配置/标签哈希、量化计划、命令前缀、媒体 SHA 与 `product_versions`。
- 最终语义与空白全量一致。
- 错误版本、基础配置哈希变化均输出 `E_SNAPSHOT_STALE` 并从游标 0 重建。

CLI 级续跑实测：对包含 7 条命令完成快照的 demo 执行 `run --resume`，本次执行 0 条命令，命令游标保持 7，canonical digest 仍为 `1dd381...fb64`。在快照副本上执行 `--resume --expected-version 1` 得到 stale 记录 `expected=1, actual=7`，随后自动重放 7 条命令，最终 digest 仍为 `1dd381...fb64`，证明 stale 恢复没有污染语义产物。
