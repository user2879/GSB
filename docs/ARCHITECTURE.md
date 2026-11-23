# 架构与算法说明

## 模块划分（`src/arcore/`）
- `spec.py` / `model.py`：JSON 声明加载、强校验（`E_SPEC`）、DAG 拓扑构建与无环检测、最长依赖路径分层。
- `prng.py`：纯 Python 的 SplitMix64 确定性随机源，跨平台逐位一致，无第三方随机状态依赖。
- `stream.py`：按种子生成告警，源内时间戳单调不减、指纹唯一；规范化排序键为
  `(timestamp_ms, service_id, fingerprint, alert_id)`，不依赖 dict/hash 遍历顺序；
  源级与全局摘要均基于规范化 JSON 的 SHA-256。
- `lease.py`：租约签发、持有者+版本号乐观续约、基于单调时钟的真实过期回收与配额归还。
- `scheduler.py`：真实 `ThreadPoolExecutor` 投射；全局信号量 + 每服务信号量 + 租约三重门控，
  排队开始/结束均以单调时钟实测；`work_hook` 用于注入真实阻塞工作与续约。
- `correlation.py`：精确两两路径（滑窗 O(n·w)）与指纹分桶近似路径（仅桶内执行同一谓词），
  自适应选择、`E_DIVERGE` 一致性判定与误差界。
- `suppression.py` / `rootcause.py`：关闭告警静默抑制；基于 DAG 层级权重的根因打分与稳定排序。
- `fixedpoint.py`：numpy 向量化定点量化，向零截断、`±saturation_max` 饱和与误差界。
- `ledger.py`：SQLite 账本，整数格式版本 + 内容摘要、整体替换、未知轮次标记与重放逐项比对。
- `engine.py`：逐轮编排，输出可持久化的确定性 payload 与定点统计。

## 确定性边界
逐字节可复现的内容包括：告警序列、源级/全局摘要、两条关联路径的关联对集合、
抑制聚合、根因候选与排序、批次规格（`batch_specs`）、定点统计、内容摘要。
天然不可跨进程复现、因此仅作审计而不参与重放比对的内容：单调时钟时刻、等待时长、
租约标识/事件顺序/占用峰值（`lease_occupancy`、`batch_observations`）。

## 关联谓词与误差界
候选对需同时满足 `|ts_a-ts_b| <= correlation_window_ms` 且服务相同或存在直接依赖边。
近似路径是精确谓词的“子集限制”：同桶对运行完全相同的判定，因此
误关联（false positive）结构性上界为 0，任何非零实测值都会触发 `E_DIVERGE`；
漏关联（miss）来自跨指纹桶被跳过的对，上界依次给出
`实测漏关联数 <= min(精确对数, 跨桶候选对数)` 三档可复核量化依据。

## 定点规则
缩放因子为 10 的整数幂；量化向零截断，单值绝对误差 `< 1/scale`；
两比例独立量化后的比较判定误差界为 `2/scale`；超出 `±saturation_max` 饱和而非溢出。
