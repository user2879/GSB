"""mvrb - 多视图三维几何重建工作台（确定性本地样本）。

模块划分：
- errors     业务错误码与异常/失败记录
- utils      稳定哈希、定点换算、JSON I/O
- config     重建配置与配置指纹
- camera     相机内参/外参与相机参数指纹
- ir         统一中间表示与校验
- exchange_v1/v2、convert  双版本交换文件解析、序列化与双向转换
- features/match/triangulate/depth/voxel  重建管线各阶段
- cache/scheduler/pipeline/snapshot       缓存、预算调度、编排与快照
- inject/diagnostics                      异常注入与跨版本诊断
"""

__version__ = "1.0.0"
