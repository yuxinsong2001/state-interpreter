# 相对状态交付演示

## 用途
可运行的接口演示，不是新的模型效果实验。只构造130点八维合成embedding：初始静止→远离→停留→返回→静止。并不经过振动预处理或Encoder，不能称完整振动端到端验证。

## 运行
在state-interpreter仓库根目录：

```powershell
python scripts/demo_relative_state_contract.py --output results/my_relative_state_demo
```

输出目录必须不存在，脚本拒绝覆盖。无需数据集或checkpoint，仅依赖项目Python/PyTorch环境。

## 如何阅读
- demo_states.csv每行记录step、输入第一轴、Level/Trend/Movement和两种资格下方向。
- unreviewed_direction：参考没有认可声明，校准后保持direction_uncalibrated。
- synthetic_accepted_direction：只对这条明确构造的合成数据给出声明，显示窗口形成、分辨界限内、远离和靠近。
- 前15点为calibrating，不输出连续状态。
- 比较两种资格，连续三量逐点完全一致；资格只限制方向解释，不修改原三量。
- 阈值0.05仅用于合成演示，绝不可替代真实数据q80或现场健康阈值。

## 本次验证
130点演示执行成功；两种资格的连续输出一致、未审参考方向被抑制、窗口/未分辨/远离/靠近标签均出现。verification.json记录验证结果。没有读取真实数据、训练或授予真实轴承资格。

这是第一阶段的演示入口。后续研究仍需明确目标和可用评价依据，演示成功不改变已有性能限制。
