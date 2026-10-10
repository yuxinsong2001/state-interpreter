# GRU双任务梯度冲突：来源核查与下一步实施边界

日期：2026-10-10。状态：来源核查完成；PCGrad代码、训练与效果验证尚未完成。

## 1. 本项目为什么检查这一方法

依据已有 `gru_state_v1_balanced_result_2026-10-10.md`：等范数共享梯度更新使B1_4的Level RMSE比普通600轮更新增加14.27–39.33%；非首轮负cosine占77.96–89.98%。因此不采用该平衡规则，普通600轮GRU仍为当前保留版本。

负cosine说明两个训练目标的共享梯度方向竞争，不能唯一证明验证偏差原因；我们没有测量曲率，不能宣称已经验证论文的完整“tragic triad”。

## 2. 核实的原始来源

| 来源 | 核查材料 | 核实结论 | 边界 |
|---|---|---|---|
| Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020 | [原论文](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)，§2.2–2.4、Algorithm 1 | PCGrad投影负内积梯度；不冲突时不修改；可用于共享参数 | 非当前PHM/GRU实验的效果证据；理论有条件 |
| PCGrad作者仓库 | [仓库](https://github.com/tianheyu927/PCGrad)、[PCGrad_tf.py](https://raw.githubusercontent.com/tianheyu927/PCGrad/master/PCGrad_tf.py)，compute_gradients | 逐任务梯度、投影、求和、交给包装优化器 | 检查到的实现是TensorFlow，不是官方PyTorch实现 |
| Liu et al., Conflict-Averse Gradient Descent for Multi-task Learning, NeurIPS 2021 | [原论文](https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf)，[作者仓库](https://github.com/Cranial-XIX/CAGrad) | 另一种已有多任务优化方法；作者提供PyTorch代码 | 本轮不实现，不宣称不如PCGrad |

核查方式：关键词PCGrad、Gradient Surgery、CAGrad，限定原论文和作者仓库，交叉检查算法与代码。不引用第三方PyTorch移植为官方实现。这是定向方法核查，不是完整系统综述。原始会议论文和作者代码可核实方法定义，但不能证明本项目收益；未做完整作者利益冲突或撤稿数据库调查。本记录由AI辅助检索与整理。

## 3. 拟采用的GRU适配（项目设计，不是原论文实验复现）

保持双头GRU：预测下一embedding与读出相对Level。仅共享GRU参数使用投影，各独立头保留自身原始梯度。

设原始共享梯度为a、b，d=a·b。当d<0且两范数非零时：

```text
a_projected = a - d / ||b||² * b
b_projected = b - d / ||a||² * a
shared_gradient = a_projected + b_projected
```

两个投影必须分别使用原始a、b，不能把第一项投影结果代入第二项。没有冲突时保留a+b。两个任务只有一个其他任务，伙伴排列不影响本方案。投影后统一clip，再由原Adam更新；不叠加等范数加权。

零范数跳过投影，非有限梯度直接报错。完全反向时共享更新可能为零，必须记录，不能偷偷回退到普通更新。数值保护只防除零，不用验证集选择冲突角度阈值。

## 4. 下一次实施的固定条件

- 新增独立模块、测试、runner、配置与结果目录；不覆盖旧实验。
- 保持输入z8、hidden8、两任务损失及训练集统计量、15点参考、10点窗口、step25评分边界不变。
- 保持B1_1–1_3训练、B1_4验证；三个seed 20260916/17/18、Adam学习率0.001、weight decay 1e-5、clip1、600轮上限和patience40不变。
- 只实例化允许的归档z；不开放其他bearing或保护数据。
- 对照是已归档普通600轮GRU，不是更换模型，也不是使用失败的平衡版本当默认。
- 同时报告预测MSE、Level RMSE、分段有符号偏差、原始冲突比例、投影次数与共享更新范数；不只选改善的指标。
- 保留原验证联合损失用于checkpoint选择；重复使用B1_4意味着结果仍是开发证据，不是独立确认。

## 5. 实施前测试与决策

先做解析向量测试：非冲突原和不变；[1,0]与[-1,1]投影求和应为[0.5,1.5]；零梯度安全；完全反向为零；独立头梯度不变；非有限输入拒绝。随后做两头因果前缀、权重重载与损失时间对齐核验。

通过实现核验才训练。只有同时呈现两任务和三个seed的结果后才决定是否采用；一头改善但另一头恶化必须称权衡，不能称全面提升。若不改善，保留普通GRU，不立即开始验证集权重扫描。

## 6. 不作出的结论

PCGrad不保证在非凸GRU与Adam下每一步两项损失同时下降；不能修复代理Level缺乏健康真值，也不能自动修复Encoder域偏移。当前只确定它具有可核实来源并可适配，不代表问题已解决。
