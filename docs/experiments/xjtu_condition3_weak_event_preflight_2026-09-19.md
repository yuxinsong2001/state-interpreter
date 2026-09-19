# Condition 3 弱事件开发实验：运行前检查

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: pre-execution validation
- Origin Date: 2026-09-19
- Verification Status: PREFLIGHT PASSED / NO NPZ PARSING / NO EVENT EXECUTION
- Protocol: `configs/xjtu_condition3_weak_event_development_v1.json`
- Config SHA256: `5e72b9646d4bcbfb11994b9ff346628ac61eb95145ba6c13ad103edff0b353b0`

本步将[弱事件草案](xjtu_condition3_weak_event_protocol_draft_2026-09-19.md)中的固定规则写成独立配置与失败即停止的预检门禁。输入仅为既有65维特征缓存中的B3_1、B3_2、B3_3。预检对manifest、三个NPZ文件和检测器代码计算SHA256；**会读取文件字节用于哈希，但不解析NPZ或接触特征数值**。B3_4/B3_5不在缓存清单和允许读取列表中。结果目录与执行记录路径分别固定，不向配置文件追加运行状态。

固定算法：RMS主规则`(h_rms+v_rms)/2`，峭度独立对照`(h_kurtosis+v_kurtosis)/2`；每条bearing前15次拟合总体均值/标准差，阈值`μ+2σ`，严格连续5次超阈后于第5次确认。两个分支都必须报告，不能事后择优；算法事件不是真实故障onset。检测器代码哈希：`e2d1f8e969301286c78b08ee7619e9ca78638e41464533d18aa115cd0de5125f`。

预检入口`python scripts/preflight_weak_event_development.py`输出`preflight_passed_no_npz_parsing_no_event_execution`，三个缓存哈希均匹配配置。合成门禁测试覆盖保护集请求、规则篡改、缓存篡改和结果覆盖拒绝；本轮全量测试**209 passed**。此检查不产生真实报警时刻或评价指标。

下一关：建立一个仅接受上述固定配置、先调用门禁再读取开发NPZ的执行入口，并在执行前明确输出格式和缺失/非有限特征处理。首次真实开发运行仅限B3_1–B3_3；B3_4/B3_5继续保护。开发结果只能用于候选事件可行性与失败模式描述，没有独立物理标签时不报告onset准确率。
