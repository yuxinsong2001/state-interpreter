# GRU相对状态原型：使用说明与阶段验收

## 交付定位

第一版可用的embedding→GRU→相对状态原型。已完成模型训练、在线推理、统一checkpoint加载、一键回放和可复现验收；没有宣称健康诊断、维护策略或跨工况普遍可靠。当前普通600轮双头GRU保留，Level早高末低为接受的已知偏差。

## 一键运行

在state-interpreter仓库根目录运行，Python环境需要torch和matplotlib（项目analysis可选依赖）。无需新训练。

```powershell
python scripts/replay_gru_state.py --checkpoint results/2026-10-10_gru_state_v1_budget600/checkpoint_seed_20260916.pt --embeddings runs/latent_analysis_z8_20260801/latent_trajectories.csv --bearing Bearing1_4 --acknowledge-matching-encoder --output results/my_gru_replay
```

输出目录必须不存在。每次运行用新目录；重复--bearing可显式选择更多episode，不默认读取全部episode。不要把保护数据加入开发演示。

CSV要求bearing_id、step_id、z_0至z_7；每个所选episode至少26点，在文件中按step_id从0连续递增，无重复、缺失或非有限值。允许不同episode行交错，但各自顺序须正确。这里输入原Encoder输出z，不是已经标准化的z，更不是原始振动。

--acknowledge-matching-encoder表示使用者确认输入来自训练时同一Encoder坐标系；这不是自动验证，也不是安全认证。不同Encoder的8维向量不能直接混用。输出记录明确保存matching_encoder_automatically_verified=false。

## 输出与生命周期

- states.csv：step、phase、Level、Trend、Movement和bearing。未READY的三量留空，不替换为0。
- state_timeline.png：所选episode的三量图，带坐标单位、来源说明与限制。
- replay_record.json：input/checkpoint/runner SHA256、所选episode、行数及限制。记录不回写模型配置。

前15点建立当前episode参考；step14缓存回放，恢复hidden；step14–24形成窗口；step25起READY。新episode自动reset。Level是相对偏离代理；Trend是每次测量索引的斜率，不是每秒；Movement是上一预测对新标准化z的RMSE，不是相邻距离。

## Python加载入口

```python
from state_interpreter.gru_checkpoint import load_online_gru_checkpoint
state = load_online_gru_checkpoint(checkpoint_path)
result = state.update(raw_encoder_z_tensor)  # CPU float32, shape [8]
state.reset()  # new device / episode
```

加载使用CPU、weights_only=True及严格model_state校验，要求训练mean/std与Level尺度、固定15/10/25契约，拒绝缺失统计量、非有限模型、非正std和Level-only诊断模型。加载器按权重推断维度，不拟合新数据。当前交付只验收普通保留权重；勿据加载成功宣称其他实验权重效果合格。只加载可信来源checkpoint，weights_only不是完整安全沙箱。

## 验收记录

- 统一loader6项＋online5项＋双头模型11项：22 passed。
- CLI缺少坐标确认、已存在输出目录：均在加载数据前拒绝，退出码2。
- 新一键演示：固定普通seed20260916、B1_4，122输入/97READY，输出完整，图已检查可读。
- 上一步四开发episode在线离线一致性：464READY，三量最大误差均<1e-6，reset重放完全一致；参见online_gru_state_demo_2026-10-10.md。
- 未修改模型、旧配置或历史结果；无新训练或保护数据读取。共享CSV未选行会被结构扫描，但未实例化其embedding。

首次6项loader测试在系统pytest临时目录创建阶段因权限失败，16项非临时测试通过；改用全新工作区临时目录后22项全部通过，不是模型异常。没有删除用户临时文件。

## 阶段完成与后续

此交付完成“可运行相对State Interpreter”的第一阶段。当前仍为归档embedding回放，不包括实时采集、原始振动预处理或Encoder加载。后续可接入固定STFT/AE提取链，先用现有开发数据验证端到端接口，不以接受偏差为由绕过数据隔离或开放保护集。
