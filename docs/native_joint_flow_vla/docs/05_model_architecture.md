# 共享三流主干

`MMRiTConfig.action_dim=0` 保持原图文模型及 checkpoint 键空间。设置正的 action_dim 启用 Action 专属 QKV、Norm、FFN、输入/输出接口和时间 embedding。三流共享一次双向 Attention。

`model(noisy_future, tau_V, noisy_plan, tau_L, action=noisy_action, t_action=tau_A, image_condition=current_vision, language_condition=instruction, ...masks, objective=id)` 返回 `(future_x0, plan_x0, action_x0)`。预测仅覆盖目标组，不包含输入条件。

视觉条件与目标各为完整 patch grid，保持各自二维位置。语言和动作使用一维位置。目标/条件角色与 objective embedding 提供变量和任务身份。

训练沿用已有优化器、梯度累积、DDP、双 EMA、日志和 checkpoint。原图文权重可严格检查后扩展初始化：只允许缺失新增 Action/角色/objective 参数。Resume 仍 strict load，错误尺寸不能静默忽略。

不包含 proprioceptive State 参数；不复制独立 Action Expert 或独立生成 Backbone。没有宣称与原预训练结果数值一致的机器人行为。
