# 变量组、Mask 与时间

输入组：current_vision / instruction / future_vision / language_plan（TU 用 instruction）/ action。goal_prediction 或 goal_conditioned_action 使用独立 goal_vision 代替 future_vision 插槽。

每个样本有 objective id，作为可学习 objective embedding 注入时间上下文。角色 embedding 区分当前/目标视觉与 Instruction/Plan。未使用组不作为 attention keys；条件保持干净 t=1。当前条件始终独立于目标标签。

目标各自独立采样 `tau_m = sigmoid(N(0,1))`，`z_m=tau_m*x_m+(1-tau_m)*epsilon_m`，m∈{V,L,A}。没有第四个 State 时间。目标之间通过同一次双向 Attention 交互。

Loss Mask、attention 可见性和数据可用性是三个不同概念。缺标签组不能被选为目标；生成组隐藏真实长度；条件组只暴露有效 token。Action-only 中真实 future_vision、goal_vision、language_plan 均不可见。

这是 JUICE 连续 Flow 的噪声机制，不是 Dynin 的随机/连续块离散 token masking。块掩码及 chunk-causal history 不属于本次实现。
