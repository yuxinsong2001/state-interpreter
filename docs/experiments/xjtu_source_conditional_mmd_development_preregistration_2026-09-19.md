# Condition 3 source-source 条件 MMD 开发实验预注册 v1

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + implementation preflight
- Origin Date: 2026-09-19
- Verification Status: PREREGISTERED / NOT RUN
- Version Label: source_conditional_mmd_development_v1
- 配置：[xjtu_condition3_source_conditional_mmd_development_v1.json](../../configs/xjtu_condition3_source_conditional_mmd_development_v1.json)，SHA-256 `1d9e9cfbf8b5f04bc2beaccdd5fd18fa8abf1a57b1efbef1c17cc921b826114b`。
- 训练入口：[train_source_conditional_mmd_condition3_development.py](../../scripts/train_source_conditional_mmd_condition3_development.py)，SHA-256 `41256a2500206ffa6016f2a0df8dc3c7f95df60a3c7f2df9dc7769593d55cc8d`。
- 来源：[方法审计](../research/conditional_alignment_method_audit_2026-09-19.md)、[数值预检结果](xjtu_source_conditional_mmd_numerical_preflight_result_2026-09-19.md)。

## 研究问题

在严格 LOBO 中，仅用两个训练 bearing，按相同归一化**时间进度区间**对齐表示，是否比无对齐和全局 MMD 更能同时保持完整生命周期健康排序并抑制 bearing identity？该区间不是已验证的物理损伤阶段；本方法是借鉴条件/子域对齐思想的 source-source 改写，不是原版 CDAN 或 DSAN 复现。

## 三分支配对设计

1. `source_only`：健康MSE，无 MMD。
2. `global_mmd`：相同健康MSE + 两个训练 bearing 的整批 MMD²。
3. `conditional_mmd`：相同健康MSE + 五个对应时间区间 MMD² 的等权平均。

三分支每 fold×seed 从相同模型初始化开始，使用完全相同的训练 endpoint、epoch、batch 顺序、优化器和固定第30 epoch checkpoint。每批从两个 bearing 的五个区间各取6个endpoint，共60；每epoch十批。B3_3 某区间只有59个 endpoint，按固定 seed 区间内循环取样，一个重复位置随 epoch 洗牌轮换；此现象必须报告。因为采样器变了，历史 Source-DANN 的 source-only 不作为严格配对对照，必须重新训练本次 `source_only`。

模型保持65维 signed-log1p 工程特征、最多128步因果上下文、单向GRU(32)、16维 embedding 和线性健康回归头；早期15步自校准、每bearing 300个均匀endpoint。每折仅两个训练 bearing；第三条完整留出做开发评价。`Bearing3_4`验证与`Bearing3_5`盲测不读取。

## 固定的损失与权重

使用包含对角项的有偏 RBF MMD²。核尺度由每 fold×seed 的**两个训练 bearing**在同一初始化模型的首批 embedding 成对距离平方中位数确定，之后冻结；用其0.5、1、2倍计算三项后取平均。global与conditional分支共用该尺度和权重。

正则权重一次性定为 `λ=0.1`，不进行网格搜索。此前只用训练来源初始化首批的九组数值预检，MMD/健康MSE 的共享encoder梯度范数比约1.48–3.23；乘0.1后约0.15–0.32。此选择仅控制初始数量级，**不是已证实的最优权重**；训练中梯度可能变化。不能根据任何留出bearing表现更改 λ、bin 数、核尺度或训练轮数。

## 固定评价与停止规则

三折 `Bearing3_1`–`Bearing3_3` LOBO，各三个seed。健康评价为留出整条 bearing 的预测健康与 normalized lifetime 的 Spearman ρ；identity 为16维embedding的五连续时间块 nearest-centroid probe。三个分支均完整报告逐折、逐seed、均值和 identity；进入下一阶段的预先指定候选仅为 `conditional_mmd`。

候选必须同时满足：三个bearing平均ρ都大于0；至少两个平均ρ≥0.5；每个bearing至少两个seed的ρ为正；identity平均准确率≤0.80。若任一失败，不打开B3_4/B3_5，不挑选好seed，不根据结果重设λ或改脚本。若全部通过，先做独立代码/泄漏复核，再按既有阶段保护流程决定是否读取B3_4。B3_5保持最终盲测。

## 当前安全检查

- `--check-only` 已通过，仅核对配置和七个冻结工件哈希，不读取开发缓存。
- 合成数据完成三个分支的一次有限前向、反向和优化器更新；错误令牌在训练前被拒绝。
- 全量测试 `200 passed`；目标输出目录与 `.staging` 均不存在。
- 原 Source-DANN 脚本/配置/结果未改动。本次尚未正式训练，也没有新的健康或identity指标。

## 后续执行边界

下一步在明确执行本预注册开发实验时，必须使用配置中唯一的执行令牌与新结果目录，一次运行三分支，不允许先看其中一个分支的留出结果再改变另两个分支。训练程序禁止覆盖现有输出，并在运行结束保存逐seed、checkpoint、轨迹和门槛报告。若中途失败，保留 `.staging` 供审计，不把残缺文件视为完成；不自动改参重跑。

## AI 使用与限制

该方法、代码和协议由 AI 协助制定及检查；科学结论只能来自后续真实开发实验。normalized lifetime 是时间代理，同百分位不一定代表相同物理健康状态；三条开发 bearing 数量少，不能把重复endpoint当作独立实验单位。核/梯度预检及合成测试验证的是实现路径，不证明泛化能力。
