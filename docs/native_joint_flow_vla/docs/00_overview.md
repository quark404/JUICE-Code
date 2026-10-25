# 总体目标与边界

常规条件只有当前视觉 `I_t` 和任务指令 `L_inst`。没有 `S_t`、proprioception、force/torque 或 sensor 输入。

可选择的输出：未来视觉 `V_future`、语言规划 `L_plan`、动作块 `A_[t:t+H)`。可选离线条件为已知动作、目标视觉或已知计划。当前视觉/未来视觉、Instruction/Plan 必须分别编码并标明变量身份，不能用一组特征同时代替条件和标签。

建模目标为 `p(L_plan,V_future,A | I_t,L_inst)`；Baseline 只启用 `action_policy`。三条流保持模态专属投影、时间条件和输出头，在共享双向 Attention 内交互。

首版只实现冻结特征缓存训练接口。原始轨迹转换、机器人控制、像素/文本解码及真实闭环评估不属于已验证功能。
