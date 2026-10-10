# PCGrad共享GRU正式开发实验结果

## 完成范围

三个seed各600轮，最佳checkpoint均epoch600；未触发提前停止，不能称已收敛。仅更改共享梯度更新，固定模型、训练数据、损失和600轮预算，与已归档普通GRU配对。协议见gru_pcgrad_formal_protocol_2026-10-10.md。

## B1_4验证结果

| seed | Level RMSE普通 | PCGrad | 相对增加 | 预测MSE相对增加 |
|---|---:|---:|---:|---:|
| 20260916 | 0.624075 | 0.650149 | 4.18% | 2.08% |
| 20260917 | 0.624336 | 0.633295 | 1.43% | 0.08% |
| 20260918 | 0.643786 | 0.685427 | 6.47% | 3.56% |

每seed两任务均变差，未观察到收益。早段正偏差分别由0.566/0.595/0.690增加为0.608/0.607/0.751；末段负偏差也没有改善，完整等数量三段统计见segment_bias_comparison.csv。Level只学习相对径向代理，不是健康损伤真值。B1_4反复用于checkpoint选择，均为开发结果，不能外推总体泛化。

## 梯度与实现证据

投影次数593、75、568（总600轮）；首轮各一次零范数跳过。其余599轮原始负cosine占98.998%、12.521%、94.825%。差异显示训练轨迹对seed敏感；原始负cosine仍存在不代表投影实现失败，因为诊断是在投影前记录，更新后的下一轮梯度仍可能冲突。

24项PCGrad与读出单元测试通过。正式三seed的权重重载输出完全一致；四bearing两头因果前缀、损失时间对齐、验证扰动不改变训练统计量及输入/config哈希核验通过。没有raw或保护目录读取；共享CSV扫描排除行但排除embedding未实例化。

## 决策与研究边界

不采用PCGrad作为默认，继续保留普通600轮GRU。该结论仅针对本双任务、当前输入/目标/优化设定，不是否定PCGrad论文，也不是否定GRU。

量级平衡和冲突投影均未改善现有偏差，因此不继续扫描冲突角度、权重或增加轮数。负冲突证据不是偏差根因的充分证明。下一项建议是同GRU的Level-only目标消融：仍保持架构、数据及读出，只将共享训练目标限定Level，预测头保留但不参与损失，专门判断联合预测任务是否导致偏差。该建议尚未实现，必须另立协议，不能事后称为本轮结果。

## 工件

- scripts/train_gru_state_v1_pcgrad_formal.py：独立正式runner，增加1800秒超时保护及状态输出；未覆盖已运行smoke runner。
- scripts/compare_gru_pcgrad_readouts.py：同seed配对及分段核查。
- configs/gru_state_v1_pcgrad_formal_2026-10-10.json：只读正式配置。
- results/2026-10-10_gru_state_v1_pcgrad_formal：三个checkpoint、1800行训练历史、12行指标、1392行状态、report.json、optimization_comparison.csv、segment_bias_comparison.csv、gradient_summary.csv。

配置SHA256：4499a930e3654b0d21a8262f807db7d7e256de7e8e04f6c3e260f80ec32023f3。runner SHA256：6b14baa958b685acd846919cfd1b9f758d594156bf43ad20d0eeaf904f749441。源z SHA256：1fbff8151767b124b68528ea5228f6e76cc391cc9b1ead333080d68e01fb9ec0。
