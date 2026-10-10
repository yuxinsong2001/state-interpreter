# GRU共享梯度量级平衡：三seed结果

## 实验完成
同一双头GRU、同数据、代理目标、三个seed、600轮预算和patience40，仅改变共享参数梯度更新。三seed均完成600轮，最佳仍在epoch600。保存三权重及1800行含梯度诊断的历史；权重重载、两头前缀、损失对齐和训练统计量隔离核验通过，相关18项单元测试通过。

## B1_4验证结果（普通更新→平衡更新）
| seed | Level RMSE普通 | 平衡 | 相对增加 | 预测MSE相对增加 |
|---|---:|---:|---:|---:|
| 20260916 | 0.624075 | 0.713142 | 14.27% | 3.51% |
| 20260917 | 0.624336 | 0.830061 | 32.95% | 0.05% |
| 20260918 | 0.643786 | 0.896971 | 39.33% | 2.35% |

各seed前、中、末段代理Level误差总体也变差，具体有符号误差/MAE见segment_bias_comparison.csv。验证参与权重选择，全部为开发结果，不能外推设备总体或称独立确认。

## 逐轮梯度证据
首轮预测共享梯度为零，各seed仅1次fallback；其余599轮负cosine比例86.98%、77.96%、89.98%。说明共享任务方向竞争在本轮训练中常见，但仍不能唯一证明验证偏差的因果来源。

prediction权重最大1858、3565、4009，Level权重通常在0.5至1之间。权重很大来自预测梯度范数很小；等范数公式使加权后的贡献有限，不代表直接梯度爆炸。但它可能过度放大小范数方向，机制尚未独立验证。本规则没有投影负冲突，两个相反贡献可抵消；不能用loss数量级相近替代此检查。

## 决定
**不升级该平衡策略为默认。保留普通600轮GRU训练权重和结果。** 这是GRU内部优化规则失败，不是GRU模型选型失败。不按当前验证集再扫描权重、不继续加轮数、不隐藏负面结果。

下一项仍限定GRU内部：先核查已有成熟的多任务梯度冲突处理方法及其适配条件，形成一个有来源的单变量更新方案；不要凭本结果继续创造任意权重公式。若执行，另立协议/config/结果目录，保持现有模型与数据边界。该后续方案尚未实施。

## 数据边界与工件
只实例化B1_1–1_4归档z，共享CSV扫描排除行但B1_5 z不实例化；无raw或保护目录读取。输入配置hash不变。scripts/train_gru_state_v1_balanced.py依赖train_gru_state_v1.py；scripts/compare_gru_balanced_readouts.py；configs/gru_state_v1_balanced_2026-10-10.json；results/2026-10-10_gru_state_v1_balanced/{metrics.csv,training_history.csv,state_outputs.csv,checkpoint_seed_*.pt,report.json,optimization_comparison.csv,segment_bias_comparison.csv,gradient_summary.csv}。
