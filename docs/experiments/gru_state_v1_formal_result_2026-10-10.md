# GRU v1正式三seed开发训练结果

## 范围
训练B1_1–1_3，验证B1_4，冻结同一归档Encoder表示。仅结构扫描共享CSV，不实例化B1_5的z；无原始/保护目录读取。三个seed20260916/17/18，固定单层hidden8、联合预测/代理Level损失，最多300轮、patience40；未重新选其他模型。

新正式runner复用train_gru_state_v1.py中的prepare/loss，固定episode顺序、训练统计量及尺度。该依赖脚本必须随runner保留以复现。score_start25、window10、gradient clipping1和hidden8在正式runner实现中固定；初版配置不支持这些参数自由扫描。

## 验证集真实结果
| seed | 最佳epoch | 验证联合loss | Level NRMSE | Level RMSE（latent距离单位） | 下一embedding MSE（标准化坐标） |
|---|---:|---:|---:|---:|---:|
| 20260916 | 300 | 0.049421 | 0.092950 | 0.722367 | 0.040781 |
| 20260917 | 300 | 0.049611 | 0.093518 | 0.726782 | 0.040865 |
| 20260918 | 300 | 0.052520 | 0.108088 | 0.840013 | 0.040837 |

NRMSE分母为训练代理目标RMS尺度7.771540，而不是验证范围、健康百分比或相对误差百分比。目标是因果径向偏离代理，Level读出拟合误差较小并不证明物理健康识别。

所有seed最佳都在300轮上限，不能称已收敛或已找到最优模型；未自动延长。联合loss主要由Level头拟合改进支配，不应将联合loss下降解释为预测头同等提升。B1_4参与最佳权重选择，结果仅为开发验证，不是独立测试。

## 完整性核验
- 三权重保存后重新加载，全部四episode的预测和Level逐值一致。
- 两头均通过前35点因果前缀检查（atol1e-6/rtol1e-5）。
- 预测target>=25与prediction[24:-1]对齐；两项损失与显式误差计算核对通过。
- 验证数据改变不影响训练mean/std及Level目标尺度。
- 输入和只读配置hash运行前后一致。
- 保存12组模型/episode指标、900行训练历史、完整状态轨迹、三checkpoint和report。

尚未完成跨seed轨迹一致性与Trend/Movement语义核查，也未比较独立bearing误差。不能从三个相近指标推出每个时间点一致。初始化与前缀等已有单元测试仍适用，本轮不是所有测试的全量重跑。

## 当前决定及下一项
GRU唯一主线继续。先在这三个冻结checkpoint的已有四条轨迹上检查Level轨迹一致性、Trend方向和Movement峰值，配图展示三seed状态及代理目标。只做已训练模型解释，不重训、不加Attention、不读取B1_5。不因为epoch300为最佳就自动增加预算。

## 工件
scripts/train_gru_state_v1_formal.py；依赖scripts/train_gru_state_v1.py；configs/gru_state_v1_formal_2026-10-10.json；results/2026-10-10_gru_state_v1_formal/{checkpoint_seed_*.pt,metrics.csv,training_history.csv,state_outputs.csv,report.json}。
