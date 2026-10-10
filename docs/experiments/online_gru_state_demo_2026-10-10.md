# 双头GRU在线相对状态接口与演示

## 已完成

新增OnlineGRUStateInterpreter，使用保留的普通600轮双头GRU，不重新训练、不修改损失/窗口。输入是一条原Encoder坐标中的z8，不是振动波形；构造时加载checkpoint的训练mean/std及Level目标尺度。

生命周期：step0–13 CALIBRATING；step14收到第15点后固定episode参考中心，并回放缓存15点恢复与训练一致的隐藏状态；step14–24 WARMING，不公布正式三量；step25起READY。每个新bearing调用reset()清除参考、隐藏状态、历史预测和窗口。

READY输出：Level为读出乘训练目标尺度；Trend为最近10点Level的OLS斜率；Movement为当前标准化z与上一点预测的RMSE，必须等新测量到达才能计算。时间轴使用测量索引，不将Trend误称每秒变化。没有健康阈值、损伤百分比或维护触发。

## 验证

新增5项接口测试加已有11项双头测试，共16 passed。覆盖在线/离线三量一致、阶段、reset重放、非法输入不推进、无未来输入、统计量校验及不改变原模型training模式。

使用预先固定seed20260916的普通600轮权重，不挑选演示最优seed。四条既有B1_1–1_4归档轨迹，共564条输入、464条READY。与原state_outputs.csv配对最大绝对误差：Level 9.54e-7、Trend 1.21e-7、Movement 6.33e-8；逐点误差低于atol1e-5+rtol1e-5界限。每条reset重放完全一致。

这是可复现embedding在线回放与工程一致性成果，不是新泛化实验或实时采集部署。既有早期高估、末期低估容忍但保留为已知限制，不据此宣称真实健康可靠。

## 文件与使用

- src/state_interpreter/online_gru_state.py：独立在线包装，不改旧在线ResidualGRU接口。
- tests/test_online_gru_state.py：合成核验。
- scripts/demo_online_gru_state.py：固定开发回放演示。
- results/2026-10-10_online_gru_state_demo/online_states.csv：全部阶段与三量。
- 同目录report.json：权重/input哈希及一致性记录。
- 同目录online_state_timeline.png：B1_4在线三量展示，带单位/来源。

```powershell
python scripts/demo_online_gru_state.py --repo . --output results/new_online_demo
```

输出目录必须不存在，旧工件不覆盖。只实例化允许开发z，扫描共享CSV排除行但未实例化B1_5，无raw/保护读取。

下一步可在该固定接口上完成checkpoint加载入口和用户演示说明，随后按明确授权选择真实数据接入；无需先消除Level偏差，不自动开放保护集。
