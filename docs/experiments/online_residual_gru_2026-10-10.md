# 复用已有GRU：在线一步预测接口

## 现有工作核查
PredictiveGRUStateInterpreter为单层GRU+线性头预测下一步embedding。原direct GRU的30组比较均弱于Persistence（gru_persistence_comparison_result_2026-09-16.md）。ResidualGRUStateInterpreter已实现z_next=z_current+delta，零初始化预测头使初始模型等于Persistence；旧结果B1_4平均Skill约2.20%，B1_5约22.28%，但后者只有27个目标且非盲测。不能重复搭建并把它当新方案，亦不能沿用这些结果称跨工况成功。

## 本次新增
OnlineResidualGRU复用ResidualGRU权重，每次update只输入一个当前embedding并保存hidden state。到当前z_t时，评分的是上一时刻对z_t的预测，同时比较Persistence的z_(t-1)。输出另含下一步预测。第一点无目标误差，返回None；reset清除hidden与上一预测。

输入必须经过原checkpoint一致的中心化/标准化，匹配dtype/device、坐标与顺序；接口不自动归一化、不拟合阈值，不称预测误差为损伤。输入是8维时可用于现有AE，但embedding_dim由权重决定。

推理模型为深拷贝eval，避免改变调用者训练状态。只允许Residual模型，新增文件不修改旧训练脚本、checkpoint或结果。

## 验证
六项新增测试连同原Residual六项，共12项通过：非零头逐步输出与整序列在atol1e-6/rtol1e-5内一致；目标对齐、Persistence误差、首次无误差、reset、零头恒等及非法输入检查通过。使用合成输入，没有加载真实checkpoint、读取原始/保护数据或训练。因此这是工程接口验证，不是GRU性能提升。

## 接下来的一个步骤
使用一个已有真实Residual checkpoint及其对应已归档、原统计量标准化的embedding，重放逐步预测并与旧批量误差逐项核对。只做集成，不重训、不调参、不新增保护数据。在加载前必须检查checkpoint架构、参考中心/标准化与episode重置口径，不能直接把其他AE模型的z输入旧GRU。
