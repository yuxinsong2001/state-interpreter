# 冻结GRU与持久性预测基线：执行前协议

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run
- Origin Date：2026-09-16
- Verification Status：执行前固定协议
- Version Label：gru_persistence_comparison_v1

## 问题

GRU的Level有时间结构，但这不直接证明它学到了有用的动态预测。本轮检验既有六个GRU（原预测目标/排序目标×三个seed）的单步预测误差，是否低于无需训练的持久性预测`z_hat_(t+1)=z_t`。

## 固定条件与对齐

读取2026-09-16排序实验的六个冻结checkpoint，按原来的前15点bearing中心化和checkpoint保存的训练统计量标准化。模型在时刻t的输出预测t+1；丢弃最后一个无观测目标的预测。主评分目标为零起算step 25至序列末尾。两种方法使用完全相同的目标行、八个维度和数值尺度，不平滑误差。

前15点校准完成之后才解释在线能力。全序列step 1起算MSE只用于历史复现和区间敏感性检查，不作为早期在线能力证据。normalized lifetime不参与预测或误差计算。

## 指标与判读

- 标准化embedding空间的MSE、RMSE及逐测量误差中位数；
- `ratio = MSE_GRU / MSE_persistence`；
- `Skill = 1 - ratio`，正数为优于持久性，0为持平，负数为更差；
- GRU逐测量误差低于基线的比例，辅助区分多数时刻表现与少数大误差；
- 每个bearing、每个训练臂跨seed的均值、样本标准差和最差值；
- 所有seed共享同一持久性基线，不将它重复计算为独立样本。

若验证bearing和旧留出bearing不能在每个seed都达到正Skill，则当前配置没有一致的额外单步预测收益。达到该条件也只能证明已见数据上的相对预测收益，不能证明健康语义或新设备泛化。不挑选最佳seed或训练臂，不在本轮重新训练。

## 核验

先验证输入与checkpoint哈希对上归档报告；全序列float32 MSE应逐项精确复现旧表；主评分用float64聚合保存的float32平方误差。检查前缀推理与完整推理的相同前缀是否一致（容差1e-6），验证持久性误差等于相邻标准化embedding差的平方。三seed和两臂的归一化统计量必须完全相同。所有受保护文件在运行前后校验哈希。

## 解释边界

持久性强是相邻测量接近时的正常现象。GRU不如它，只能说明本模型在本预测目标、数据与训练预算下没有建立优势，不能推导GRU普遍无效，亦不能单凭预测失败否定Level的全部用途。若GRU优于它，也不能代替故障事件、物理损伤或维护决策评价。Bearing1_4参与checkpoint选择，Bearing1_5已被多次查看；这是事后诊断，不是盲测。

## 运行

使用已有Python 3.12环境执行`python -B scripts/compare_gru_persistence.py --config configs/xjtu_gru_persistence_comparison_v1.json`。新目录保存误差明细、seed指标、bearing汇总、复现检查、图与报告；目录存在时拒绝覆盖。先检查绘图依赖再开始计算，避免再次到收尾时才发现缺少依赖。
