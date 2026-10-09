# 相对事件状态接口（2026-10-09）

## 目的与实现
新增`RelativeEventStateInterpreter`，包装既有`OnlineEpisodicWeakEvent`。保持冻结v2计算不变，为使用方同时提供连续偏离、趋势与事件生命周期。模块：`src/state_interpreter/relative_event_state.py`。

| 字段 | 含义 |
|---|---|
| feature_deviation_z | 当前特征相对前15次参考的有符号标准化偏离，单位参考SD |
| feature_trend_z_per_measurement | 最近至多10次有效连续观察的斜率；不足2点为null |
| observation.active_event | true活动、false监测、null校准/无效参考/缺口未知 |
| observation.duration_since_confirm | 当前事件自确认以来已观察的活动测量数 |
| transition | none、entered、exited、interrupted |
| termination_now | 仅在退出或活动事件遇缺口时产生的结束元数据 |
| last_termination | 保留最近一次结束信息，后续事件仍可查看 |

连续偏离不是物理损伤百分比；趋势不是故障概率；active_event=false不代表健康。一个新超阈点可具有较大偏离但尚未达到5次确认，连续输出与离散状态因此同时保留。当前模块输入是单一RMS/峭度均值流，尚未连接latent或GRU Movement。

## 结束原因与时间口径
- `consecutive_non_exceedances`：连续5次回到阈值以内，确认规则结束；不声明物理恢复。
- `measurement_gap`：缺口中断观察，持续时间仅截至最后已观察活动点，right_censored=true；不把缺失时间补成活动时长。
- `record_end`：调用`finish_record()`显式结束记录；尚活动事件标记right_censored=true，不生成虚假的在线退出。

结束元数据保留episode编号、候选开始、确认步、最后活动观察、结束被观察到的步号和已观察活动数。正常退出步本身不是活动点。例如确认19、退出24，活动点19–23共5个。`reset()`清除当前bearing状态和历史结束信息；结束记录后继续update需要reset。

`schema_version=relative_event_state_v1`是接口第一版；底层算法仍为episodic v2。`to_dict()`提供JSON可序列化消息，示例已保存在结果目录。

## 验证结果
14项接口与原v2合成测试通过，包含重新进入、退出时长、缺口删失、记录末端、拒绝后状态不变、reset、无效参考、连续偏离与事件并列、因果前缀一致和底层逐步输出一致。

仅从已保存CSV重放10,810条观察；全部原v2字段逐项精确相同。接口产生19份事件结束元数据，其中14份规则退出、5份记录末端右删失，全部与原episode表的确认步、持续步数和删失状态匹配。没有重新训练或读取B3_4/B3_5；此验证只证明接口保留历史算法行为，不增加健康有效性证据。

## 计数更正
之前两份说明将事件总数写为20，这是整理错误。正确计算为2+3+12+0+1+1=19；原CSV及数值诊断正确。验证脚本最初因错误硬编码20而在写结果前停止，修正为依据归档记录数核对后成功；旧报告追加更正，不覆盖历史实验。

## 工件与下一步
`results/2026-10-09_relative_event_state_interface/`：states.csv（10,810条）、episode_terminations.csv（19条）、example_state_messages.json（B3_1 RMS进入/退出示例）、report.json（哈希与核对结果）。重放入口：`scripts/replay_relative_event_state_interface.py`；测试：`tests/test_relative_event_state.py`。

下一步可建立面向真实使用的状态时间线，将偏离、趋势、事件状态和结束原因一起展示，先用开发归档做演练；如要进入B3_4，应另建固定验证协议，并明确只评价操作性变化事件。
