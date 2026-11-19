# 需求追踪

| 业务要求 | 实现位置 | 自动验证 |
|---|---|---|
| 1. 确定性切分、连续序号、摘要登记、覆盖语义、`E_CHUNK` | `chunkdeps/util.py`, `chunkdeps/engine.py::create_session/upload_chunk` | `DeterministicChunkTests` |
| 2. 五态状态机、迁移事件、完成终态、位图断点续传 | `sessions`, `session_events`, `engine.py::resume_session/abort_session`, `client.py::upload_session` | `StateMachineTests`, `test_client_resume_skips_registered_chunks_without_rewriting_them` |
| 3. 逐片校验、`E_INTEGRITY`、无半对象、原子可见性 | `engine.py::_verify_chunks/_write_object_temp/commit_session` | `IntegrityTests`, 强制终止演示 |
| 4. 派生元数据/清单/前缀视图、DAG 校验、稳定增量失效 | `chunkdeps/graph.py`, `engine.py` 依赖节点构造 | `DependencyTests`, 演示中的稳定控制节点检查 |
| 5. 全局/会话并发配额、排队、重复片收敛、顺序稳定 | `chunkdeps/server.py::UploadThrottle`, `scheduler_events` | `NetworkEndToEndTests::test_concurrent_upload_is_throttled_and_duplicate_chunk_converges` |
| 6. 版本单调、整体替换、旧版本过期、字段/摘要/视图差异 | `engine.py::commit_session`, `object_versions` | `VersionAuditSnapshotTests` |
| 7. 一致审计、磁盘分片集合、快照版本/图/审计、整体恢复 | `engine.py::audit/snapshot_save/snapshot_restore` | `VersionAuditSnapshotTests`, 演示 snapshot 检查 |
| 8. 强杀恢复、未知合并重跑、三类故障码和结构化日志 | `engine.py::_recover_after_restart`, `chunkdeps/logging.py`, 注入分支 | `FaultAndRecoveryTests`, 演示 forced termination 与业务码计数 |

## 关键不变量

- 可见对象 = 已提交会话 + 完整分片文件 + 完整对象文件 + 已提交 SQLite 元数据。
- 活动版本最多一个；版本号只增不减；历史版本只读。
- 一个 `(session_id, chunk_index)` 只有一个登记行。
- 失效只沿有向边向后代传播；未受影响节点的版本和摘要不写库。
- 依赖图每次修改前拒绝环；启动和查询时验证无悬空引用。
- 服务端并发配额是最终硬限制，客户端并发数不能绕过。
- 快照恢复是整体替换，不是行级增量合并。
- 强制终止时不推断合并结果，不生成退出报告。

