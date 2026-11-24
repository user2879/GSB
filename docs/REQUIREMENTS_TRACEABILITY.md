# 需求追溯矩阵

| # | 需求要点 | 实现位置 | 验证位置 |
| --- | --- | --- | --- |
| 1 | 声明/样本/候选加载与强校验，首个违规 `E_SOURCE` | `config.py`、`models.py` | `tests/test_validation_codec.py`、`demos/demo_error_paths.py` |
| 2 | 三类来源多版本编解码、长度自洽、字段补齐、`E_FORMAT` 行号 | `codec.py`、`aggregation.py` | `tests/test_validation_codec.py::CodecNormalizationTests` |
| 3 | 偏置/版本/标识全序仲裁、否决理由、顺序无关逐字节一致 | `arbitration.py`、`models.canonical_json` | `tests/test_arbitration_quantization.py::ArbitrationTests` |
| 4 | 10^k(k=0..3) 定点、整数乘除、截断/饱和、三段误差界与主导段 | `quantization.py` | `tests/test_arbitration_quantization.py::QuantizationTests`、`docs/FIXED_POINT_ERROR_BOUNDS.md` |
| 5 | 不可区分稳定决胜、三类成因与占比、与时序无关 | `diagnosis.py` | `tests/test_arbitration_quantization.py::DiagnosisTests`、`demos/demo_full_pipeline.py` |
| 6 | 全局/每来源配额、真实套接字超时、子进程回环服务、E_PORT/E_TIMEOUT/E_ACK | `replay_server.py`、`scheduler.py` | `tests/test_replay_scheduler.py` |
| 7 | 方案版本递增、引用传播、陈旧标记、视图整体替换与增删改清单 | `invalidation.py`、`engine.py` | `tests/test_engine_invalidation.py` |
| 8 | SQLite 账本版本/摘要、事务写入、硬杀恢复、unknown 重跑、`E_REPLAY` 定位 | `ledger.py`、`engine.py` | `tests/test_ledger_recovery.py`、`demos/demo_crash_recovery.py` |

## 错误码对照

| 码 | 触发条件 | 携带定位 |
| --- | --- | --- |
| E_SOURCE | 来源/候选标识重复、非负整数违规、类型/版本不支持、引用缺失、帧文件缺失 | 来源/候选标识 + 字段名 |
| E_FORMAT | 魔数/帧头版本未知、记录版本未知、frame_len 不自洽、字段缺失、区间反转 | 来源标识 + 首个违规行号 |
| E_PORT | 端口绑定失败/无监听/子进程未就绪 | host:port + 阶段 |
| E_TIMEOUT | 真实 socket 连接或回执超时 | 候选标识 + 阶段 |
| E_ACK | 回执缺必需字段或 source_id 不符 | 候选标识 + 字段名 |
| E_REPLAY | 恢复后重放与账本首个字段不一致 | 候选(查询) + 字段 |
| E_LEDGER | 账本不可识别（内部兜底） | 原因 |
