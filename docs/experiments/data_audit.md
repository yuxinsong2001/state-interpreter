# XJTU-SY 数据审计

> 审计日期：2026-08-01  
> 数据根目录：`D:\1\德国留学\斯图加特大学在校资料\hiwi工作\XJTU-SY_Bearing_Datasets`  
> 当前状态：原始数据已通过 7-Zip 完整解压，并完成结构完整性复核。

## 1. 本地文件状态

`Data` 目录包含六个连续分卷：

| 分卷 | 大小（字节） |
|---|---:|
| `part01.rar` | 744,488,960 |
| `part02.rar` | 744,488,960 |
| `part03.rar` | 744,488,960 |
| `part04.rar` | 744,488,960 |
| `part05.rar` | 744,488,960 |
| `part06.rar` | 722,155,640 |

压缩文件总大小约 4.44 GB。当前 D 盘可用空间约 115.5 GB，足够进行后续解压。

## 2. 归档内容完整性

通过只读列出六个分卷，共发现 9,216 个 CSV 条目、3 种工况和 15 个轴承。每个轴承的测量编号均从 1 连续到终点，没有缺号。

| 工况 | 轴承 | CSV 数量 | 编号范围 | 缺失编号 |
|---|---|---:|---:|---:|
| 35 Hz / 12 kN | Bearing1_1 | 123 | 1–123 | 0 |
| 35 Hz / 12 kN | Bearing1_2 | 161 | 1–161 | 0 |
| 35 Hz / 12 kN | Bearing1_3 | 158 | 1–158 | 0 |
| 35 Hz / 12 kN | Bearing1_4 | 122 | 1–122 | 0 |
| 35 Hz / 12 kN | Bearing1_5 | 52 | 1–52 | 0 |
| 37.5 Hz / 11 kN | Bearing2_1 | 491 | 1–491 | 0 |
| 37.5 Hz / 11 kN | Bearing2_2 | 161 | 1–161 | 0 |
| 37.5 Hz / 11 kN | Bearing2_3 | 533 | 1–533 | 0 |
| 37.5 Hz / 11 kN | Bearing2_4 | 42 | 1–42 | 0 |
| 37.5 Hz / 11 kN | Bearing2_5 | 339 | 1–339 | 0 |
| 40 Hz / 10 kN | Bearing3_1 | 2538 | 1–2538 | 0 |
| 40 Hz / 10 kN | Bearing3_2 | 2496 | 1–2496 | 0 |
| 40 Hz / 10 kN | Bearing3_3 | 371 | 1–371 | 0 |
| 40 Hz / 10 kN | Bearing3_4 | 1515 | 1–1515 | 0 |
| 40 Hz / 10 kN | Bearing3_5 | 114 | 1–114 | 0 |

## 3. CSV 结构抽查

抽查对象：

- `35Hz12kN/Bearing1_1/1.csv`；
- `35Hz12kN/Bearing1_1/123.csv`。

两者均满足：

- 32,768 行测量数据；
- 2 列；
- 列名为 `Horizontal_vibration_signals` 和 `Vertical_vibration_signals`；
- 数值可正常解析为浮点数。

抽查统计：

| 文件 | 水平 RMS | 垂直 RMS | 水平最小值 | 水平最大值 |
|---|---:|---:|---:|---:|
| `1.csv` | 0.563890 | 0.560471 | -2.529883 | 2.354336 |
| `123.csv` | 7.268278 | 5.783524 | -35.597789 | 29.754794 |

末次测量振动幅值和 RMS 明显高于首次测量。这只证明抽查样本存在明显信号变化，不代表已经完成完整退化趋势验证。

## 4. 可构造的数据契约

每个 CSV 可转换为：

```text
episode_id = bearing name
step_id = CSV numeric filename - 1
time_index_minutes = CSV numeric filename - 1
vibration = float tensor [2, 32768]
operating_conditions = {speed_hz, radial_load_kn}
metadata = {condition_id, source_path}
```

数据不包含逐时刻的 `damage_target`、`health_index_target`、`stage_target`、maintenance action 或 reward，这些字段必须保持为空，不能由最终故障类型伪造。

## 5. 推荐划分

正式实验先选择单一工况，并按 bearing 做 leave-one-bearing-out。禁止随机按 CSV 或由同一 CSV 切出的子窗口划分训练和测试。

管线调试可先使用 `35Hz12kN/Bearing1_1`，因为只有 123 个测量文件，能够较快检查完整生命周期流程。单轴承调试结果不作为科学泛化结论。

## 6. 审计结论

数据满足第一阶段 AutoEncoder 与 latent-space State Interpreter 的结构要求：设备身份、工况、时间顺序、双通道振动和完整 run 终点均可恢复。下一步可以完整解压数据，然后实现只读 XJTU-SY Dataset Adapter 和预处理 shape smoke test。

## 7. 解压后复核

- 完整解压目录：`Data/XJTU-SY_Bearing_Datasets`；
- CSV 总数：9,216；
- CSV 总大小：12,219,567,144 字节；
- 零字节 CSV：0；
- 额外文件：`Introduction_to_XJTU-SY_Bearing_Dataset.pdf` 1 个，因此 7-Zip 界面显示 9,217 个文件；
- 15 个 bearing 均存在，每个 bearing 的数字编号连续且无缺号；
- 跨分卷边界的 `Bearing2_3/358.csv` 与 `359.csv` 均存在；
- 对 15 个 bearing 的首个和末个 CSV 共 30 个文件进行抽查，全部为 32,769 行（1 行表头 + 32,768 行数据），表头均为两个规定振动通道。

7-Zip 报告“有效数据外包含额外数据”警告，但上述文件总数、bearing 结构、编号连续性、边界文件和抽查行数均通过，因此当前数据通过结构完整性验收。尚未对全部 9,216 个 CSV 逐文件执行数值有限性扫描。

## 8. 尚未完成

- 尚未逐一验证全部 9,216 个 CSV 的列名和有限值；
- 已实现 XJTU-SY Dataset Adapter，并通过 `Bearing1_1` 前两个测量的真实数据 smoke test；
- 尚未固定 STFT 参数；
- 尚未实现或训练 AutoEncoder；
- 尚未产生任何 latent-space 科学结果。

## 9. Dataset Adapter smoke test

真实输入：`35Hz12kN/Bearing1_1/1.csv` 和 `2.csv`。

验证结果：

- `episode_id = 35Hz12kN/Bearing1_1`；
- `step_id = [0, 1]`；
- `time_index = [0.0, 1.0]` 分钟；
- 两个 vibration tensor 均为 `[2, 32768]`；
- dtype 均为 `torch.float32`；
- 两个 tensor 均全部为有限值；
- 工况解析为 `rotational_frequency_hz=35.0`、`radial_load_kn=12.0`；
- source path 可追溯到原始 CSV。

该结果证明真实数据能够进入统一 `RawMeasurement` 契约，不证明 STFT、AutoEncoder 或健康状态解释已经有效。
