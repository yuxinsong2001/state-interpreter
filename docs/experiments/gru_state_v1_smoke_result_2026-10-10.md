# GRU v1联合训练：首次短训练

## 完成范围
新增train_gru_state_v1.py，复用冻结归档z8，无Encoder训练。仅实例化B1_1/B1_2/B1_3训练、B1_4验证embedding，分别123/161/158/122点；不实例化B1_5。共享CSV被逐行结构扫描以筛选，不能说物理文件完全不包含B1_5；未读取原始振动或保护目录。

每episode前15点中心化；训练episode校准后的数据拟合逐维mean/std，最小std1e-6；未标准化坐标径向距离的因果10点平均为Level代理。目标尺度为训练episode step>=25的平方均值按episode等权平均再开根号，7.771540。训练loss为下一embedding标准化MSE与缩放Level MSE之和，两项各按episode等权。

预测评分target step>=25，对齐prediction[24:-1]与z[25:]；读出评分step>=25。每episode独立forward重置hidden，不串设备。模型读出回乘训练尺度，Trend为最近10步读出的斜率，Movement为当前标准化embedding与上一步预测的RMS残差。不同旧Movement定义不可混用。

## 真实短训练结果
seed20260916，Adam lr0.001，hidden8，单层，5轮集成检查。没有参数扫描。

| 指标 | epoch1 | epoch5 |
|---|---:|---:|
| 训练联合loss | 0.599929 | 0.576860 |
| 训练预测MSE | 0.007523 | 0.007208 |
| 训练缩放Level MSE | 0.592406 | 0.569652 |
| 验证联合loss | 0.337396 | 0.312505 |
| 验证预测MSE | 0.041511 | 0.040986 |
| 验证缩放Level MSE | 0.295886 | 0.271519 |

5轮内最佳为epoch5，保存checkpoint及完整曲线CSV。这是短训练而非正式性能结论，不把loss下降称为健康诊断改善。

## 检查与限制
将验证embedding乘100重新准备统计量，训练mean/std/目标尺度逐值不变；各episode前35点的Level预测与全序列前缀在atol1e-6/rtol1e-5内一致；输入和配置hash执行前后不变。尚未重新加载checkpoint复核完整误差，也未执行正式三个seed、NRMSE汇总或跨bearing泛化验证。

运行出现将带梯度Tensor转float的日志警告，不影响已经完成的反向训练；当前脚本保留执行版本。正式runner如修复日志或增加完整性验证，应明确记录新版本，不能倒改这次工件。现有数据准备隔离检查不是所有泄漏风险的全面证明。原Encoder provenance沿用既有归档，本次未重新提取或验证其原权重。

## 工件
configs/gru_state_v1_smoke_2026-10-10.json：只读短训练配置。

results/2026-10-10_gru_state_v1_smoke：checkpoint.pt、training_history.csv、state_outputs.csv、report.json。

## 下一项
先补完整checkpoint重载、预测前缀和损失手算测试，并固定episode迭代顺序；使用新正式配置按设计执行三个seed最多300轮开发训练。保持B1_5不读，不重新选其他方法，不回写短训练配置和结果。GRU仍为唯一主线，但必须完整报告失败或局限。
