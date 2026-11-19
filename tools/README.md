# 本地离线编码工具目录

本工作台的“有损压缩样本”不依赖任何外部可执行编码器或在线服务，
有损编解码器 LRX1 完全由 Python/NumPy/SciPy 实现，位于
`voice_repair_workbench/lossy_codec.py`，容器扩展名为 `.lrx`。

如在真实 Windows 环境希望改用外部离线编码器（例如 ffmpeg.exe），
可将可执行文件放置在本目录，并在调用层封装为同样的
`encode_file(signal, sample_rate, path)` 接口；默认演示不需要任何二进制。
