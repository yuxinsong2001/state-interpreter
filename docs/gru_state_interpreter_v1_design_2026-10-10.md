# GRU State Interpreter v1：唯一主模型设计

## 状态与方向
2026-10-10；设计已确定，模型扩展与训练尚未完成。依据用户决定，GRU为唯一核心时序模型，不再进行HMM/其他解释器选型。本方案是项目内工程设计，不声称为某篇论文的官方方法。

## 为什么不是重复旧任务
旧hidden-distance、predicted-z-distance、预测残差读出已在9月16日核查；时间排序损失0.1也已试过。本版不重复这些试验，不将时间先后强制解释为健康单调性。新增的是**显式监督的相对偏离读出头**，它与下一embedding预测共享GRU。

## 固定输入与结构
固定原Condition1 z8表示，复用runs/latent_analysis_z8_20260801/latent_trajectories.csv，必须核对其来源checkpoint和hash。每episode前15个z均值作为中心，仅校准完成后正式读出；前15点不作在线状态声明。GRU输入中心化后、仅训练轴承拟合的逐维标准化序列，维度8。

单层单向GRU hidden_dim8，初版不加入Attention、不增层：

```text
z序列（冻结Encoder） → GRU hidden h_t
                         ├─ 零初始化delta线性头 → z_next_hat = z_t + delta_t
                         └─ Linear(8,1) + Softplus → relative Level_hat_t
Level_hat历史 → 10步因果窗口斜率 → Trend_hat_t
z_t与上一步z_hat_t → RMS预测残差 → Movement_t
```

Level_hat不是hidden坐标的距离；它是有明确目标的标量读出。Movement是预测残差，不能与旧相邻embedding距离混用。模型尚无证据表明此设计改善状态可靠性。

## 训练目标与语义
相对偏离监督目标d_t为**未标准化Encoder坐标中**||z_t−前15点中心||，再做过去最多10点的因果平均。训练目标只从校准及窗口形成完成后的step≥25生成，不包含寿命、故障阶段、健康标签或总序列长度。

读出目标除以训练轴承d_t的RMS尺度（不减均值），使非负目标适配Softplus；推理回乘该训练尺度。若尺度≤数值容差则停止并报告，不能人为调大。

损失固定：L = mean(next-embedding MSE) + mean(scaled relative-Level MSE)。两项权重均1，分别先在每个训练episode有效行内取均值，再按episode等权平均。下一步目标>=25；读出目标>=25。初版无排序、无单调性、无附加平滑损失。

**重要边界：该读出是在学习一个透明的相对距离代理。** 不是发现健康真值，训练误差下降不证明退化识别；即使复现目标，也不能说独立学出了新的健康含义。这个明确目标用来使GRU状态读出可训练且可检查，后续才能讨论增加有效状态依据。

## 第一轮数据与训练
训练B1_1–B1_3；验证B1_4；B1_5暂不读取。本轮只开发，不叫盲测。沿用旧Residual的Adam lr0.001、weight_decay1e-5、gradient_clip1、最多300轮、patience40；seed20260916/17/18。不因新结果回写旧配置。

验证集仅用同一联合损失选checkpoint；每seed保存最佳与完整曲线。每epoch独立episode重置hidden，禁止串接不同轴承。规范记录输入hash、训练统计量、目标尺度、实际有效行与checkpoint。

## 验证目的（不是替代模型选型）
- 单元验证：形状、非负读出、零delta初始化、预测目标对齐、因果前缀与流式/批量一致、episode重置。
- 完成训练：每seed产生有效checkpoint与无NaN曲线；读出训练/验证NRMSE及每轴承误差完整报告，不预设必然提升。
- 语义验证：Level代理误差与时间ρ分别报告，不相互替代；Trend方向一致性单独报告。
- 三seed状态轨迹的一致性与幅值差异记录，不择优seed，不把seed当独立设备。
- Persistence仅是预测头的恒等性/技能诊断，不是候选解释器；不是重新进行方法选型。

首轮无健康性能成功门槛，也不宣布部署；完成标准是可运行、可复现、有可解释目标且完整报告局限。性能结论由真实结果决定。若读出不能学到该代理，先定位训练/尺度/容量；不否定用户已确定的GRU路线，但不隐瞒失败。

## 下一实施任务
新增独立模型文件gru_state_readout.py和对应测试；复用Residual核心，增加标量读出和明确输出类型。先通过shape、梯度、因果及零头测试，再新增训练runner。不得覆盖旧GRU配置、脚本、结果或打开保护集。
