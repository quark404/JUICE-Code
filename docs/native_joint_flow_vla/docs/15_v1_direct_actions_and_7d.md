# V1：参考 OpenPI/π0.5 的直接连续动作；7D EEF 转换分析

## 本次第一版决定

默认训练配置 `configs/vla/action_baseline.yaml`：

```yaml
action_mode: padded
action_representation: continuous
action_dim: 32         # 模型动作宽度，不是机器人自由度
action_native_dim: 7   # 无标签推理记录的示例回退宽度，不是动作转换算法
action_adapter: continuous
vla_tasks: {action_policy: 1.0}
```

原生归一化动作 `[H,d_i]` 先在末尾补零为 `[H,32]`，然后直接加连续噪声、输入 Linear→Joint Attention→Linear、计算连续 Flow loss。没有 Action Encoder/Tokenizer、RVQ、码本或 learned latent 目标。

参考依据是 [OpenPI PadStatesAndActions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L328)、[默认宽度32](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)、[DROID输出截回8D](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py)、[ALOHA输出截回14D](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/aloha_policy.py)。

本版仅借鉴其动作接口与直接连续处理，不复制完整 π0.5 架构、State 输入、离散辅助训练或噪声时间约定；仍是无 State 的 JUICE 三流模型。论文与源码对照见 [参考文档](13_action_references.md)。

## Padding 与 Mask 的明确区别

为贴近所核对 OpenPI 的实现，padded 模式对**整个模型宽度**采样噪声并计算原有动作维度平均损失，padding 的 clean target 为零。它不启用 heterogeneous 的逐维 Mask Loss 或 layout embedding。预测尾部不保证严格为零，部署时按原生维度截掉，不能把尾部当成可执行通道。

已有 fixed 模式（严格 `[H,action_dim]`）保留；heterogeneous 的具名字段/Mask 分支与 frozen X-Tokenizer 分支都保留为额外实验，不是第一版主线。

## 数据与输出

- 每条有标签记录的 `action` 保持原生 `[H,d_i]`，`0 < d_i <= action_dim`。DataLoader 推断 d_i 并补齐；同一 batch 可以混合不同原生宽度。
- 可显式给出 `action_native_dim`，必须匹配原生 tensor 宽度。不要把已经补齐的32D tensor伪装成原生7D记录，补齐应只做一次。
- 无 action 标签的推理记录需要显式 `action_native_dim`，或使用配置的回退值。否则不能推断应该截回几维。
- `sample_vla` 的 `A` 是 `[B,H,32]`；额外返回 `A_native` 列表，每项为该样本 `[H,d_i]`。反归一化和控制器调用使用 A_native。
- 原生动作的关节/EEF语义、字段顺序、单位、旋转表示、absolute/delta、夹爪极性、控制周期与归一化统计必须在数据适配器中定义和记录。补零不提供这些转换。
- 若 native_dim>32，可显式选择更大的模型宽度，但涉及模型形状/权重迁移；不自动截断，也不把超宽字段丢弃。

改变 action_mode 或宽度不能直接 resume 旧配置。跨接口迁移使用 `init_from` 和 `init_backbone_only: true`，显式重置 Action 分支；相同 padded architecture 的 checkpoint 正常 resume。

## “统一7D末端控制”到底是什么意思

Dynin-Robotics §4.1 声明：来源专属变换将 native controls 变成 translation(3)+rotation(3)+gripper(1)；运动通道按数据源 p01/p99 归一化，夹爪保留 absolute，常量/unused运动通道置零。§3.2.1 在此之后才做均匀量化。

这里统一的是**物理动作接口**，不是任意原生向量取前7个数，也不是把7个关节角当成末端6自由度＋夹爪。论文没有完整给出所有来源的轴序、旋转表示、控制增量定义或双臂选择规则。因此下面是可实现的数据适配流程，不声称这些具体公式就是 Dynin 每个数据源的原实现。

### 1. 先声明目标接口

例如选择 `a_t = [delta_p_x,delta_p_y,delta_p_z,r_x,r_y,r_z,g_absolute]`。

必须先声明：位移所在坐标系、旋转三参数的表示（Euler增量/rotation vector等）、旋转左乘还是右乘、每步时间间隔和夹爪取值。三个旋转参数不等于X-Tokenizer的6D rotation表示。不要在同一个通道中混合不同表示。

### 2. 按来源类型转换运动通道

| 数据源原始控制 | 可行转换 | 必要前提 |
|---|---|---|
| 已有EEF位移＋旋转增量 | 重排轴/符号，转换单位与坐标系，再转换旋转表示 | 原始frame和旋转约定已知 |
| EEF绝对目标pose | 用目标pose相对当前pose计算delta | 当前pose与时间对齐 |
| EEF线/角速度 | 根据控制周期积分为位移/旋转增量 | 周期与速度frame已知 |
| 关节位置目标 | 对当前q、命令q_target分别做FK，再计算末端pose差 | 正确机器人URDF/标定、base frame与关节定义 |
| 关节增量 | q_target=q_current+delta_q，再做上述FK | 关节增量定义、当前q可用 |
| 关节速度 | 积分/Jacobian得到近似末端变化，或用实测pose转移构造标签 | 状态、周期、运动学模型；必须区分近似命令与实际运动 |
| 扭矩/力或接触控制 | 不存在只依赖动作向量的通用确定性7D转换 | 需要动力学/接触和状态，或独立实测运动标签 |

例如在明确定义为 world-frame 增量时，可采用：

`delta_p_world = p_target_world - p_current_world`

`R_delta_world = R_target_world @ R_current_world.T`

`r_world = Log(R_delta_world)^vee`

这是 rotation-vector 目标接口的一种选择。若控制器需要tool-frame增量，应使用相应的局部旋转组合，不能沿用上式后不作标注。

对于速度，小时间步可用 `delta_p≈v*dt`，角速度应按约定通过SO(3)积分。大的时间间隔或接触过程不能把一次线性近似视为精确执行结果。

### 3. 转换夹爪

明确原始值是开合宽度、二值开合、位置命令还是力命令；确定开/关极性并标定到同一数值约定。夹爪保持 absolute，不把“张开0.8”累加成增量。无法等价转换的力控制不能静默冒充位置命令。

### 4. 时间对齐和训练集统计

先统一或显式标注控制周期，确保动作、当前观测与目标帧对齐。完整chunk `[t,t+H)` 对应的未来帧是执行该chunk后的帧；单步frame与chunk-end frame不能混用。

在**物理语义转换之后**按训练集、数据源计算六个运动通道的lower/upper（Dynin用p01/p99），裁剪并缩放到[-1,1]。部署保留同一来源统计用于反归一化。归一化只处理尺度，不负责解决坐标系、旋转表示或控制模式差异。

### 5. 双臂及信息损失

两个独立末端＋夹爪自然是14D。规约为7D意味着选择一只手、分成两条带arm身份的样本，或采用受约束的任务表示，都会改变任务接口。不能把两臂平均或静默截断并称为无损统一。

Dynin只说明活跃双臂来源被规约到同一7D view，未在已提供正文中给出完整规约规则；不能据此断言它支持完整双臂7D控制。我们的V1可以直接保留原生14D再补到32D，不需要做这种规约。

## 无State模型与预处理所需信息

为将关节目标转换成EEF delta，离线适配器可能需要当前q/pose。它用于生成标签，不意味着把State作为JUICE输入。如果连这些标签转换所需信息也没有，就不能伪造可靠的7D接口；V1应保留已定义清楚的原生动作，或排除不能对齐的数据源。

当前未提供具体机器人数据源、URDF或控制约定，因此没有加入猜测性的通用7D转换器。第一版实现的是有官方源码依据的padding/direct-action机制；来源专属7D adapter需在拿到数据定义后逐个实现和验收。
