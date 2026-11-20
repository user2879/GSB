# 测试套件

运行全部测试：

    python -m pytest -q

覆盖范围：

- test_loader_validation.py：E_SPEC 强校验、E_ORDER 全局到达倒退、E_PRIORITY 缺失租户、首个出错对象定位
- test_engine_verdicts.py：授予/排队/拒绝三态、拒绝三分类、突发授予、并发排队与稳定唤醒、内部顺序无关性
- test_paths_consistency.py：三路径 NumPy 逐字节一致、归并批次统计与合并次序确定性、选择器阈值决胜
- test_ledger_snapshot.py：批次原子事务与摘要链、强制终止恢复与未知标注、E_LEDGER 篡改检测、
  快照写入/恢复/整体替换/只读备份/E_REPLAY、注入只改数据
