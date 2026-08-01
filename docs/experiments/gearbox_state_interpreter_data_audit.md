# Gearbox State Interpreter 数据审计

## 审计目的

确认当前 Gearbox 项目能否在同一个时间步关联以下信息：

- vibration
- damage
- operating conditions
- action
- reward
- next state
- episode ID

本审计只判断数据与接口是否存在，不把当前仿真器视为已经通过物理验证的正式实验数据源。

## 运行链路

当前环境中一次 `step(action)` 的实际顺序是：

```text
action
  → action2torque
  → gearbox.set（保存 torque，并计算 load）
  → nolc 增加
  → degradation.run
  → vibration.run
  → vibration 转 observation
  → stop criteria
  → reward interpreter
  → 返回 next observation, reward, done, info
```

这说明项目确实生成连续 episode，而不是互不相关的单次测量。一个 episode 内的数据可按
`episode + counter/nolc` 排序组成时间序列。

## 字段审计表

| 所需字段 | 当前来源 | 同步情况 | 当前是否对外返回 | 结论 |
|---|---|---:|---:|---|
| vibration | `env.vibrations` / `gearbox.ga_vibration` | 当前时间步 | observation 只返回其变换结果 | 可获得，但需显式记录原始值 |
| observation | `env.obs` | 当前时间步 | 是 | 可直接使用；当前 smoke 配置形状为 `(1024, 1)` |
| damage | `gearbox.ga_statei[-1]` | 当前 `nolc` | 否 | GearIn/GearOut 可获得；Bearing1–4 当前均为 `None` |
| operating condition: torque | `env.torque` / `gearbox.ga_torque[-1]` | 由当前 action 设置 | 否 | 可获得，但需显式记录 |
| operating condition: load | `gearbox.ga_loads[-1]` | 与当前 torque 对应 | 否 | 可获得，但需显式记录 |
| rotational frequency | `gearbox.Vibration.rotational_frequency_in/out` | 当前配置中固定 | 否 | 可作为 episode/static metadata |
| action | `env.action` | 当前 transition | 仅 verbose 时写入 `info` | 可获得，但默认不记录 |
| reward | `env.reward` | 当前 transition | step 返回；verbose 时也写入 `info` | 可直接获得 |
| next state / next observation | `step` 返回的 `obs` | action 后的下一时刻 | 是 | 外部 transition collector 可与前一 observation 配对 |
| episode ID | `env.episode` | reset 后递增 | 仅 verbose 时写入 `info` | 可获得，但默认不记录 |
| step ID | `env.counter` | episode 内递增 | 仅 verbose 时写入 `info` | 可获得，但默认不记录 |
| physical time index | `env.nolc` | load-cycle 时间轴 | 仅 verbose 时写入 `info` | 可获得；不是秒，应保留单位语义 |
| done | `env.done` | 当前 transition | 是 | 可直接获得 |

## 关键结论

### 1. 数据是连续序列

`reset()` 建立一个 episode，`step()` 每次令 `nolc += nolc_step`，并在同一 Gearbox
退化对象上继续推进。因此可以构造：

```text
(observation_t, action_t, reward_t, observation_t+1, done_t)
```

也可以把每个时间步的 vibration、damage、torque/load 一起保存。

### 2. 当前还没有标准 transition 数据接口

环境内部拥有多数目标字段，但默认 `verbose=0` 时 `info` 基本为空；即使打开 verbose，
`info` 也不包含 vibration、damage、torque/load 和 next observation。因此不能直接把
现有 `info` 当成 State Interpreter/RL 数据表。

### 3. Gear damage 有标签，Bearing damage 暂时没有

`gearbox_degradation.py` 当前对 `GearIn`、`GearOut` 计算退化状态，而
`Bearing1`–`Bearing4` 在初始化和每步运行时均设为 `None`。所以当前代码只能提供齿轮
damage/pitting ground truth，不能提供轴承健康状态标签。

### 4. 当前 reward 不是学习型 State Interpreter

现有 `interpreter` 实际用于计算 reward，可选规则包括 MSE、scaled、linear 和固定 step
reward。smoke 配置得到固定 `0.5`。它与计划中的
`VibFM z_health → interpretable RL state` 不是同一个模块，后续应避免名称混淆。

## 建议的标准 transition 记录

每个时间步至少保存：

```text
episode_id
step_id
nolc
observation_t 或 vibration_t
z_health_t
damage_t
operating_condition_t
action_t
reward_t
observation_t+1 或 z_health_t+1
damage_t+1
done_t
```

其中 `z_health` 要等 VibFM checkpoint 和适配器可用后生成；在此之前可以用 fake
embedding 测试接口，但不能用于科学结论。

## 对第一版 State Interpreter 的影响

第一版采用单窗口模型：

```text
z_health_t → MLP → health indicator + stage probabilities → compact RL state
```

接口暂时不强制要求 history、action 或工况。后续用 reward/transition prediction 和
history-gain test 判断是否升级为 GRU/TCN 或 Hybrid Interpreter。

## 已执行验证

命令：

```powershell
python .\env_smoke_test.py --steps 2
```

结果：

- reset observation shape：`(1024, 1)`
- step 1：`action=0`，`nolc=256`，`reward=0.5`
- step 2：`action=1`，`nolc=384`，`reward=0.5`
- 两步均未结束 episode

运行时出现 Gym 已停止维护的警告；本次 smoke test 仍成功完成。
