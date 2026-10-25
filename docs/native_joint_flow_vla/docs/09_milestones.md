# 里程碑与当前状态

已实现：无 State 三流模型；直接连续动作与可选 MLP；冻结缓存数据接口；独立当前/未来视觉和 Instruction/Plan；12 类目标路由；Action-only 默认配置；三流 Flow loss、Euler latent sampler 与小型 CPU 回归测试。

尚待实际机器人数据验证：原始轨迹转换/时间对齐工具；B1 Action Policy 训练和闭环评估；WM/Language 单目标训练；VA/LA/LVA Joint 实验；语言/视觉解码与完整控制服务。

论文四任务方向已核对；训练目标/噪声/损失采用 JUICE 连续 Flow，不能称为 Dynin 离散实现的复现。先完成可比较的 Action-only 基线，再评估辅助目标的实际收益。
