# 参考与实现差异

- JUICE：https://arxiv.org/abs/2610.28801 — 连续表征空间、clean-state prediction、双向 Joint Flow。
- Dynin-Robotics: Omnimodal Unified Diffusion Vision-Language-Action Model：https://arxiv.org/abs/2609.13053 — 本次依据用户上传 PDF 核对。

已阅读与本版有关章节：
- §3.2.1 / Eq.(3)：Action 训练集稳健统计缩放、均匀量化、time-major 标量 token；本版只使用连续缩放，不量化。
- §3.2.2：Sensor 为 optional，报告的 Stage-1 与 Stage-2 policy decoding 不启用 sensor；本版完全省略 State/Sensor。
- §3.3 / Fig.4：PO、WM、GP、TU 的 visible/target 布局；本版保留任务方向，添加显式 objective embedding。
- §3.4 / Eq.(11)：Action/WM Joint 是 inference-only composite；本版 joint_* 监督训练为项目扩展。
- §4.2：任务调度和 loss 系数不是同一件事；本版用可配置随机任务混合，不声称复现该 schedule。

论文不支持“Dynin 原生用连续 Action Flow/可训练 Action MLP”“所有 visual state 都是 proprioception”“Dynin 已验证三目标 Language Plan–Vision–Action 联合训练”等结论。
