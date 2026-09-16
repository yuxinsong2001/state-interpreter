# GRU 时间排序目标：三随机种子配对实验结果

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Origin Mode：run
- Origin Date：2026-09-16
- Verification Status：对照训练与冻结推理精确复现；排序训练未另做同seed重复
- Version Label：gru_temporal_ranking_result_v1

## 结论

这一次固定权重的时间排序约束没有解决当前 GRU Level 的跨 seed 稳定性问题。Bearing1_4 的ρ标准差略增，Bearing1_5 下降约16.82%，没有达到预设的20%改善门槛。两个bearing的平均ρ均略降；当前不据此升级默认State Interpreter。

这个结论仅限本次结构、权重、排序间隔和300轮训练预算，不代表GRU或所有排序训练方法不可行。

## 做了什么

保留504参数单层GRU和hidden-distance Level。三个相同seed分别训练两臂：

- control：next-step embedding prediction MSE；
- ranking：MSE + 0.1 × 时间排序损失。

排序损失比较同一训练bearing中相隔10次测量的10点平滑Level，惩罚较晚Level没有比早期Level高出0.05的情况。使用前15点参考和训练集专用标准化。训练Bearing1_1–1_3，Bearing1_4仅按预测MSE选择checkpoint，所有六个checkpoint锁定后统一评价。

排序标签来自先后顺序，这是一项弱监督退化假设。模型没有输入时间索引、总寿命或normalized lifetime；但其训练已受到时间顺序约束，因此高时间ρ不能单独证明健康语义。

## 结果

全部数字在相同的step 25起点评价。简单距离基线在本轮重新计算；请勿与旧step 15报告混用。

| Bearing | 简单距离ρ | 原预测GRU：ρ均值 ± seed标准差 | 排序GRU：ρ均值 ± seed标准差 |
|---|---:|---:|---:|
| Bearing1_1 | 0.9519 | 0.9518 ± 0.0006 | 0.9518 ± 0.0006 |
| Bearing1_2 | 0.9965 | 0.9996 ± 0.0001 | 0.9997 ± 0.0000 |
| Bearing1_3 | 0.9598 | 0.9634 ± 0.0021 | 0.9635 ± 0.0021 |
| Bearing1_4 | 0.8531 | 0.8284 ± 0.0632 | 0.8265 ± 0.0653 |
| Bearing1_5 | 0.9713 | 0.9489 ± 0.0320 | 0.9471 ± 0.0266 |

表中0.0000为四位小数舍入；Bearing1_2排序GRU的实际标准差为0.00004551，并非完全一致。

### Bearing1_4

- seed标准差：0.06321 → 0.06527，增加约3.25%；
- 最差seed ρ：0.75638 → 0.75259；
- 平均预测MSE：0.05522 → 0.05506；
- 回落步比例均值：0.4132 → 0.4167。

预测误差稍有改善，但健康排序和稳定性没有同步改善。图上原GRU与排序GRU几乎重叠，明显的后期回落仍存在。

### Bearing1_5

- seed标准差：0.03197 → 0.02659，下降约16.82%；
- 平均ρ：0.94892 → 0.94709；
- 最差seed ρ：0.92552 → 0.92552；
- 平均预测MSE：0.03210 → 0.03197。

存在有限的seed稳定性改善，但未达到协议门槛，且平均ρ仍低于简单距离基线。不能只挑这个局部改善宣布方法成功。

## 预设决策检查

| 条件 | Bearing1_4 | Bearing1_5 |
|---|---|---|
| seed标准差降低至少20% | 未达到 | 未达到 |
| 平均ρ下降不超过0.01 | 达到 | 达到 |
| 最差seed ρ不降低 | 未达到 | 达到 |
| 预测MSE增加不超过10% | 达到 | 达到 |
| 无状态塌缩 | 达到 | 达到 |
| 恒定输入动态范围检查 | 通过 | 通过 |
| 支持按当前配置升级验证 | 否 | 否 |

门槛是实验前固定的探索性决策规则，不是统计显著性检验。

## 是否仅学会了计时

保留前15点后，将输入固定为早期参考，排序GRU的Level变化范围最多约为真实轨迹的0.10%，低于预设10%标记阈值。这轮人工诊断未发现大幅度的恒定输入上升现象。

有些恒定输入曲线的ρ仍接近1，但幅度极小；这直接说明只看ρ会误导。ρ衡量顺序，与变化幅度无关。

将校准后的输入打乱顺序，Bearing1_4/1_5排序GRU的ρ变为约-0.30至-0.22。原来的正相关依赖于真实观测序列；不过这些人工输入检查不能证明模型具有物理健康语义，也不能完全排除其他捷径。

## 实现和验证

- 新增可微分因果Level与排序损失模块，模型参数数量不变；
- 83项测试通过，包括排序方向、恒定Level惩罚、梯度传播和完整窗口因果性；
- 三个control checkpoint与既有实验参数、最佳epoch、验证MSE精确一致；
- 两臂同seed初始化完全相同；
- 六次训练各300轮，最佳checkpoint均为第300轮，尚不能确认充分收敛；
- 最终重新加载六个checkpoint，逐点Level与预测MSE的最大复现差异均为0；
- 1,800行训练历史、3,696行逐点状态、35行指标、60行人工输入诊断；
- 原配置、既有训练结果和已保存数值工件未被修改。

## 环境问题与恢复

初始训练命令完成模型和指标保存后，在绘图阶段因缺少matplotlib退出（exit code 1）。经工具批准安装matplotlib，使用Python 3.12运行独立的finalize脚本补齐图表和报告；未重复训练，也未覆盖已有CSV或checkpoint。

`recovery_record.json`保留原执行记录、错误原因、恢复动作及数值工件不变检查。最终状态为`completed_after_plot_dependency_recovery`。普通沙箱中的`python`与批准运行时的`python`指向不同安装；本次最终报告明确保存了Python 3.12绝对路径及软件版本，后续建议使用显式解释器路径。

## 文件入口

- 固定配置：`configs/xjtu_gru_temporal_ranking_v1.json`
- 实验脚本：`scripts/run_gru_temporal_ranking.py`
- 无训练恢复脚本：`scripts/finalize_gru_temporal_ranking.py`
- 排序模块：`src/state_interpreter/temporal_ranking.py`
- 结果目录：`results/2026-09-16_gru_temporal_ranking/`
- 汇总：`summary_by_bearing.csv`、`decision_checks.csv`、`experiment_report.json`
- 图：`level_comparison.png`（各曲线仅为绘图单独缩放至0–1，不是在线归一化或损伤百分比）

## 限制与下一步

全部bearing都已被查看；Bearing1_4参与checkpoint选择，Bearing1_5多次用于事后分析，不能宣称盲测泛化。三个seed仅描述初始化波动，不代表三台独立设备。训练和评价共享时间顺序假设，缺少物理损伤标签的独立验证。

暂不升级默认Level，也不在本轮调整排序权重。继续研究前，应补一个成本低但关键的检查：冻结这些GRU，与`z_hat_(t+1)=z_t`的持久性预测基线比较，判断GRU到底有没有学到超出“保持当前状态”的预测信息；同时明确Level与变化检测的评价任务。该检查尚未执行。通过更有区分力的评价后，再决定是否值得扩大训练预算、换结构或加入Attention。

