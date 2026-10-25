# 任务矩阵与阶段

所有任务均不使用机器人 State；下表当前视觉是 `I_t`，Instruction 是 `L_inst`。

| 配置名称 | 干净条件（当前视觉外） | 目标 | 定位 |
|---|---|---|---|
| action_policy | instruction | action | 默认 PO Baseline |
| world_model | instruction, action | future_vision | Dynin WM 任务方向 |
| goal_prediction | instruction | goal_vision | Dynin GP；样本当前视觉应选初始帧 |
| task_understanding | 无指令条件 | instruction | Dynin TU；输入视觉特征由数据准备选择 |
| language_planning | instruction | language_plan | 项目扩展 |
| inverse_dynamics | instruction, future_vision | action | 离线辅助 |
| plan_conditioned_action | instruction, language_plan | action | 项目扩展 |
| goal_conditioned_action | instruction, goal_vision | action | 目标引导策略 |
| joint_va | instruction | future_vision, action | 联合训练扩展 |
| joint_la | instruction | language_plan, action | 联合训练扩展 |
| joint_lv | instruction | language_plan, future_vision | 联合训练扩展 |
| joint_lva | instruction | language_plan, future_vision, action | 联合训练扩展 |

12 项是可选接口，不表示 Dynin 训练了 12 项。Dynin 主要训练 PO/WM/GP/TU；§3.4 的 Action–WM 联合去噪是 inference-only composite，本项目的 joint_va 监督训练应作为单独实验。

阶段：Action-only → 单目标 WM/Language 基线 → 双目标 Joint → 三目标 Joint → Unified。用 vla_tasks 权重选择任务，不强制依序初始化。默认只 action_policy，不能把辅助目标提升假定为必然收益。
