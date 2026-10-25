# 评估与消融

代码验证：原图文接口回归，12 类任务/两种 Adapter 的有限 loss 和反向传播；Action Policy 对真实未来视觉、目标视觉和 Plan 变化不敏感；采样不读取真实目标 Action；缺标签失败、归一化边界、任务配置与缓存读取校验。

机器人实验待做：normalized/physical Action error、闭环 success、延迟，Future Vision latent error、Language Plan 可解码性、视觉/计划/动作一致性。测试过代码不等于验证过这些指标。

对照：直接连续 Action vs MLP 边界；Action-only vs PO/WM/GP/TU 混合；单/双/三目标联合训练；相同 checkpoint 下 Action-only vs Joint sampling。控制数据量、计算预算、主干调用数和动作语义。

禁止用真实未来帧或真实动作作为在线 Action Policy 条件。所有统计和数据划分必须基于 train trajectories，不能随机划分相邻窗口。
