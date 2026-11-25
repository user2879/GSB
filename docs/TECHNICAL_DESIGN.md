# 特征新鲜度预算与回填幂等诊断组件 — 技术说明

## 1. 范围与运行环境
- 纯本地组件，无容器、无外部服务、无网络访问。
- 运行时：Python 3.11（Windows 官方安装包），依赖仅 `numpy==1.26.4`。
- 所有路径使用 `pathlib.Path`，原子替换使用 `os.replace`（NTFS 与 POSIX 均为原子语义），换行统一写入 `\n`，可直接在 Windows 本地运行。
- 固定整数种子 `20261125` 生成样本，不含真实账号、线上站点或外部数据集。

## 2. 目录结构
- `ffbudget/loader.py`：阶段 1 加载与校验，非法项映射业务码进入失败清单。
- `ffbudget/authority.py`：阶段 2 幂等裁决、权威修订、并列摘要决断、保留深度裁撤与回填点账本。
- `ffbudget/view.py`：阶段 3 生效时刻视图、陈旧/未来回退、陈旧上限与缺省向量。
- `ffbudget/scoring.py`：Q16.16 定点整数槽位打分。
- `ffbudget/scheduler.py`：阶段 4 三重约束调度、并发执行、预算账本与候选排序。
- `ffbudget/cache.py`：内容指纹缓存簿与槽位级精确失效。
- `ffbudget/snapshot.py`：临时目录写入 + 整目录 `os.replace` 提交、清单版本识别与恢复判定。
- `ffbudget/report.py`：逐请求陈旧分布、并列裁决、拒因与预算诊断报告。
- `tools/generate_samples.py`：确定性样本生成；`tools/run_demo.py`：验收演示与一致性比对。
- `tests/`：26 个单元/集成测试。

## 3. 关键业务语义

### 3.1 加载校验（业务码见 `ffbudget/errors.py`）
- 槽位键唯一，层级限定 `request/item/cross`，保留深度与维度为正整数。
- 单个 NDJSON 文件只承载一个 `batch_seq`；文件内同槽位 `rev_seq` 非递减。
- `effective_ms` 为非负整数毫秒，`points` 为正整数，向量维度与槽位声明一致。
- 跨文件的相同批序号（重投/乱序）不在加载层拒绝，而交由裁决层处理，保证与到达顺序无关。

### 3.2 权威修订裁决
- 同组回执排序键：`(-rev_seq, -effective_ms, -batch_seq, digest)`。
- 前三者完全并列时以内容摘要 SHA-256 字典序决断，并置 `tie_resolved=true`，拒因 `E_TIE_DIGEST`。
- 落败回执按首个差异字段记 `E_REV_SEQ_LOWER / E_EFFECTIVE_EARLIER / E_BATCH_SEQ_LOWER`，并记录冲突权威摘要。
- 完全相同的重复投递（同 `batch_seq + digest`）记 `E_DUPLICATE_BATCH_DELIVERY`，`counted=false`，不重复进入回填点账本。
- 超出槽位保留深度的旧修订仅保留最近 N 条，其余记 `E_REVISION_PRUNED` 且不计回填点。

### 3.3 可用视图与陈旧回退
- `effective_ms <= rank_ms` 的最高修订为当前可用修订，陈旧度 `rank_ms - effective_ms`。
- 无已生效修订时回退到保留深度内最近修订，状态 `future_fallback`，陈旧度为负整数（显式表达“尚未生效”）。
- 陈旧度超过 `staleness.max_staleness_ms` 或无任何历史修订时状态为 `unavailable`，使用槽位固定缺省向量，产物记录 `vector_source=default:<配置来源>`。

### 3.4 定点打分与预算
- 位宽档位 `i64q16`（16 位小数），全程 `numpy.int64`，溢出按二进制补码回绕，Windows/Linux 位级一致。
- 槽位贡献 = 定点权重 × 陈旧惩罚系数 × 向量整数和；候选总分按 `total_score desc, item_id asc, request_id asc` 决断。
- 每处理一个槽位按 `dim` 计点；预算在任何线程启动前按 `request_id → slot_key → item_id` 稳定顺序领取。
- 全局预算不足：请求已领单元全额退款，请求进入待办并记 `E_SCORE_BUDGET_PENDING`。
- 单槽位超配额：槽位中止，后续单元记 `E_SLOT_QUOTA_EXCEEDED`。
- 被幂等拒绝的回填批不产生打分点；预算账本事件序列（claim/insufficient/refund）随产物落盘可复核。

### 3.5 指纹缓存与精确失效
- 视图键、打分键、排序键均由槽位内容指纹、权威修订指纹、陈旧参数指纹与位宽档位派生。
- 新增批次只失效其覆盖的槽位组；排序键仅引用该请求实际使用的槽位打分键集合，未受影响请求直接命中。
- 缓存命中/ miss 计数只出现在运行期报告，不进入阶段快照，保证全量与增量的规范化产物逐字段一致。

### 3.6 并发与确定性
- 预算领取、待办判定、失败清单与结果组装全部在主线程按稳定顺序完成；线程池只对不可变单元做纯计算。
- 因此 `max_workers=1` 与任意并发度、任意批文件到达顺序，业务产物完全一致（由 `tools/run_demo.py` 与测试验证）。

### 3.7 快照与恢复
- 四个阶段各自先写入临时目录再 `os.replace` 整目录提交；清单含全局版本、格式版本、配置哈希、批游标、已消耗点数、阶段产物版本与输入指纹。
- 启动时依据格式版本 + 配置哈希判定 `incremental/full_rebuild`；输入指纹不变时全局版本保持稳定，输入变化或清单损坏时全量重建。

## 4. Windows 本地运行
```powershell
py -3.11 --version
py -3.11 -m pip install numpy==1.26.4
py -3.11 tools\generate_samples.py
py -3.11 -m ffbudget.cli run --config samples\config.json `
  --batches samples\batches --requests samples\requests.json `
  --state state\run --mode full --out state\result.json
py -3.11 tools\run_demo.py
py -3.11 -m unittest discover -s tests -v
```

## 5. 验证结果
最新一次本地验证（见 `state/demo/acceptance_report.json`）：
- 全量重放、增量等价、并发度等价、乱序到达等价、快照恢复、幂等重投 6 项全部 PASS。
- 单元/集成测试 26 项全部通过。
