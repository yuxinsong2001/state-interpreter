# Bearing1_4证据核查（2026-09-16）

本目录是描述性证据核查，不是训练或盲测。

- aligned_evidence.csv：616次原始测量的RMS/峭度/峰值、存档latent与HMM发射证据逐点关联。
- audit_report.json：五个bearing摘要、上次概率域/log域差异的精确复核与输入哈希。
- raw_file_hashes.json：616份原始CSV的SHA-256。
- alignment_verification.json：冻结checkpoint重提取Bearing1_4的122个embedding，与存档的容差检查及末点敏感性。

详细解释见../../docs/experiments/bearing1_4_evidence_audit_2026-09-16.md。

关键结论：Bearing1_4同时呈现持续信号变化与末次强冲击，embedding保留了变化而HMM未充分表达。数据不足以把全部现象归结为随机波动，也不足以识别物理损伤原因。

更正：旧数值验证“最大差为0”是显示精度误读；实际最大差约1.33e-15，离散阶段一致。旧文件保留，以本次精确检查为准。
