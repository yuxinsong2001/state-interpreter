# Residual GRU 终止性实验方案（2026-09-17）

## 研究问题

将一步预测显式改写为`z_hat(t+1)=z(t)+delta_GRU(t)`，并以零初始化增量头从persistence开始，是否能够稳定取得正预测Skill？

## 唯一实验改动

- direct GRU：读取已冻结的control checkpoint，不重新训练；
- residual GRU：相同单层GRU、hidden dim 8、相同数据和优化设置；
- residual输出层初始权重与偏置均为0，未训练时必须逐值等于persistence。

## 固定设计

- 训练：Bearing1_1–Bearing1_3；
- checkpoint选择：Bearing1_4全序列next-step MSE；
- 旧holdout描述性评价：Bearing1_5；
- seed：20260916、20260917、20260918；
- 最多300轮、patience 40、Adam学习率0.001；
- 主评价目标：target step ≥ 25；
- 比较：direct GRU、residual GRU、persistence。

## 预设停止条件

Residual GRU只有在Bearing1_4和Bearing1_5上三个seed全部取得正Skill时，才支持继续将一步预测作为核心Health Level研究路线。否则停止该主线，转向健康感知Encoder或其他明确状态目标。

## 解释边界

所有bearing均已查看；B1_4参与checkpoint选择，B1_5也已多次检查。本实验不是盲测，也不进行显著性推断。三seed只描述初始化随机性。正预测Skill也不自动等于健康语义。
