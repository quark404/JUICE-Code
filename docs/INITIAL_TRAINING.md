# Native Joint Flow VLA — 第一版开发文档（V0.4）

> **目标**：基于 JUICE 的 Joint Multimodal Flow，构建支持 **Language Planning、Future Vision、Action** 的统一生成框架，允许单目标、双目标和三目标联合去噪。
> **阶段**：V0.4 / 总体设计与初步训练计划；不是已完成的机器人训练结果。
> **原则**：架构支持三模态，实验分阶段推进；先验证直接连续 Action Policy，再验证联合建模收益。**Robot State 是可选条件**，不增加第四种生成目标。

## 0. 整体方案与技术边界

### 0.1 输入与输出

- **常规条件**：当前图像 `I_t`、原始任务指令 `L_inst`。
- **可选 Robot State**：当前状态 `S_t`，例如关节位置、夹爪开度或末端位姿；没有 State 的数据仍可参与训练和推理。
- **任务相关条件**：已知动作块、目标视觉或已知语言计划，仅在对应任务中提供。
- **生成目标**：语言计划 `L_plan`、未来视觉 `V_(t+H)`、动作块 `A_[t:t+H)`。

条件与目标按**变量组**区分：当前图像与未来图像、原始指令与语言计划分别编码。每个生成目标独立采样 Flow Time `tau_L、tau_V、tau_A`，所有已知条件保持干净。

总体建模目标：

`p(L_plan, V_(t+H), A_[t:t+H) | I_t, L_inst, [S_t])`

方括号 `[S_t]` 表示可选输入。State 不加噪，不作为预测目标。

### 0.2 第一版范围与实现状态

| 项目 | 第一版选择 | 当前状态 |
|---|---|---|
| Backbone | JUICE 双向 Joint Attention，V/L/A 三流 | 已有代码 |
| Vision / Language | 冻结预训练 Encoder，分别编码各变量组 | 已有编码器与特征缓存训练路径 |
| Action | 直接连续动作，归一化后补齐至32维 | 已有代码，默认入口 |
| Robot State | 可选归一化条件，经轻量投影注入主干 | **设计项，尚未接入代码** |
| Objective | Clean-state prediction，速度空间 Flow loss | 已有代码 |
| Sampling | Euler；多目标默认同步推进 | 已有代码 |
| 数据准备 | `.pt` 窗口特征缓存与 manifest | 缓存加载已有；原始轨迹构造待补齐 |
| 真实训练与控制 | Action baseline、联合训练、闭环评估 | **尚未验证** |
| 暂不纳入主线 | X-Tokenizer、复杂时间调度、Looped Backbone、额外一致性损失 | 保留独立实验或后续研究 |

这里“已有代码”表示接口和代码测试，不代表完成了真实机器人训练。

## 1. 数据准备（Data Preparation）

**目标**：以完整轨迹为单位，将同一条轨迹展开为不同条件/目标组合的训练窗口。需要计划监督的实验假设 subtask 与计划文本已标注；Action-only 不依赖这些标注。

### 1.1 原始数据单位：Trajectory

```text
trajectory = {L_inst, I_0,[S_0], A_0, I_1,[S_1], ..., A_(T-1), I_T,[S_T], Subtasks}
```

- `I_t`：当前视觉；第一版先使用单帧，视觉特征可预计算。
- `S_t`：可选当前机器人状态，不允许用未来状态代替。
- `A_t`：从 t 到 t+1 的控制命令；记录字段顺序、单位、坐标系、旋转表示、absolute/delta、夹爪极性和频率。
- `Subtasks`：任务区间、计划、目标帧和成功标记等，按具体监督任务提供。
- `T`：动作转移数；图像原则上有 T+1 帧，有 State 的来源应按相同时间戳对齐。

**时间约定**：t 是机器人轨迹时间；tau 是 Flow 去噪进度。二者不能混用。

### 1.2 窗口切分与对齐

| 切分单位 | 内容 | 用途 |
|---|---|---|
| Transition | 当前/下一观测与单步动作 | 单步世界模型、逆动力学 |
| Action Window | `A_[t:t+H)` 与 `I_(t+H)` | Action Policy、Joint V–A |
| Subtask Segment | 当前观测、计划与子任务终点 | Planning、Goal Prediction |
| Whole Trajectory / Clip | 起终点或按时间排序的观测 | 目标预测、任务理解；多帧接口后续扩展 |

首版先用完整窗口：`t+H <= T`；不跨 subtask 时再要求 `t+H <= subtask_end`。H 示例为16。动作维度补零不等于时间长度补齐。

先按 trajectory 划分 train/val/test，再生成窗口。每条样本记录 trajectory_id、t、H、subtask_id 和目标时间点；窗口末端、下一帧和任务终点不能混称。

### 1.3 统一样本接口

以下为**拟定的逻辑接口**，不是当前缓存加载器可直接读取的嵌套格式：

```yaml
sample_id: traj_001/t_020/H_016/joint_lva
trajectory_id: traj_001
t: 20
H: 16
objective: joint_lva
conditions:
  current_image: I_20
  instruction: L_inst
  robot_state: S_20       # 可省略
  state_present: true
  state_valid_mask: valid_fields
targets:
  language_plan: L_plan_20
  future_vision: I_36
  action_chunk: A_[20:36]
metadata:
  action_native_dim: 7
  future_step: 36
  subtask_id: grasp_cup
  normalization_id: source_train_stats
```

实际已有训练路径使用扁平 `.pt` 字典：`current_vision`、`instruction`、`instruction_mask`、`action`，按任务再提供 `future_vision`、`goal_vision`、`language_plan` 等。State 接口完成后再增加 `robot_state`、`state_present`、`state_valid_mask`。

当前图像与未来图像独立编码；Instruction 与 Plan 独立编码，不能先拼接进双向 T5 再拆开，避免目标信息泄漏。归一化统计只使用训练集；缓存动作已归一化，训练时再采样噪声。

### 1.4 待办与交付物

- [x] 冻结特征缓存的 manifest 加载、组可用性与有效性处理。
- [ ] 原始 trajectory/subtask schema、窗口索引与来源专属动作转换器。
- [ ] State 字段、缺失状态标记和归一化接口。
- [ ] 时序、跨 subtask、数据划分和坐标系检查。

**交付物**：可审计的轨迹窗口缓存、来源统计和数据一致性检查。

## 2. 任务定义（Task Definition）

**策略**：单目标 Baseline → 双目标 Joint → 三目标 Joint → Unified。单目标指一个带噪生成变量组，不代表条件只包含一种模态。

记 `C = (I_t, L_inst, [S_t])`；Task Understanding 不提供作为标签的指令条件。

### 2.1 单目标生成 Baseline

| 编号 / 配置任务名 | 干净条件 | 唯一生成目标 | 优先级 |
|---|---|---|---|
| B1 `action_policy` | C | Action | 首先验证 |
| B2 `world_model` | C、已知 Action | Future Vision | 独立视觉基线 |
| B3 `language_planning` | C | Language Plan | 独立语言基线 |
| B4 `inverse_dynamics` | C、真实 Future Vision | Action | 离线辅助 |
| B5 `goal_prediction` | 初始/当前观测、指令、可选 State | Goal Vision | 与窗口末端视觉分开标注 |
| B6 `task_understanding` | 视觉、可选 State | Instruction | 不能同时暴露真实指令 |
| B7 `plan_conditioned_action` | C、已知 Plan | Action | 计划条件对照 |
| B8 `goal_conditioned_action` | C、已知 Goal Vision | Action | 区分在线可用目标与离线标签 |

### 2.2 联合目标

| 编号 / 配置任务名 | 干净条件 | 联合生成目标 | 阶段 |
|---|---|---|---|
| J1 `joint_va` | C | Future Vision、Action | 优先联合实验 |
| J2 `joint_la` | C | Language Plan、Action | 双目标 |
| J3 `joint_lv` | C | Language Plan、Future Vision | 双目标 |
| J4 `joint_lva` | C | Language Plan、Future Vision、Action | 三目标 |

这些任务的无 State 路由已有代码；State 条件待接入。当前 `joint_lv` 使用 Future Vision 插槽；若改用子任务 Goal Vision，需要显式新增映射，不能只改标签名称。

默认配置只启用 B1。联合损失接口可以保留，但不等于第一阶段同时训练所有任务。已知真实动作、未来帧和计划标签不能泄漏到在线 Action Policy。

**交付物**：任务矩阵、标签需求、采样权重及对应独立基线。

## 3. 模态表征（Modality Representations）

### 3.1 表征选择

| 变量 | 首版表征 | 是否训练表征模块 |
|---|---|---|
| Current / Future Vision | 冻结 SigLIP2 或 WebSSL-DINO 连续特征 | Encoder 冻结；主干投影可训练 |
| Instruction / Plan | 独立编码的 Flan-T5-Small 连续特征 | Encoder 冻结；主干投影可训练 |
| Action | 归一化原生连续控制量 | Linear 输入/输出投影可训练 |
| Robot State（可选） | 按字段归一化的当前状态 | 拟采用轻量 Linear 条件投影 |

### 3.2 Action：直接连续处理

```text
原生动作 [H,d] → 原生通道归一化 → 尾部补零 [H,32]
              → 连续 Flow → 截回原生 d 维 → 反归一化 → 外部控制器
```

32 是模型宽度，不是机器人自由度。7D、8D、14D 等原生动作均可补齐，但不会自动统一关节/末端语义。缓存保留原生宽度；DataLoader 只补齐一次。每个动作时间步形成一个 Token，因此 H 步对应 H 个 Action Tokens。

归一化采用训练集的逐通道统计，部署复用同一来源统计。它是第一版的尺度选择，不是 Flow Matching 的数学必要条件。第一版不经过 X-Tokenizer、量化或 Action latent Decoder。

### 3.3 Robot State：可选条件设计

拟定 `use_robot_state` 开关，默认关闭；该配置项尚未实现。

- S0：当前状态归一化后，经过 Linear 投影形成干净条件 Token。
- State 维度与 Action 维度独立，分别保存字段定义和归一化统计。
- 用 `state_present` 区分有/无 State，用 `state_valid_mask` 标识有效字段。缺失不能仅用零值表达。
- 关闭或缺失时屏蔽 State attention key；有效零状态仍是有效输入。
- 第一版先使用一个字段定义明确的机器人；State History、Expert、FiLM/调制作为后续消融。

State 不加噪、不生成、不计算去噪损失；不把它当作第四条目标流。

### 3.4 参考依据与证据边界

| 参考 | 本项目采用或借鉴的内容 | 不作出的推论 |
|---|---|---|
| JUICE | 连续表征、联合注意力、clean-state Flow | 图文效果不能直接当作机器人策略效果 |
| π0.5 / OpenPI | 固定模型宽度、动作补齐和输出截取 | 不复现完整架构，不因此自动获得动作语义对齐 |
| Dynin-Robotics | Policy / WM / GP / TU 的条件与目标布局 | 其离散量化、masked diffusion 与本项目不同 |
| X-Tokenizer | 作为独立候选编解码实验 | 量化向量不是未经量化的连续潜空间；不纳入首版主线 |

State 条件 Token 是本项目拟定的可测试接口，不声称已证明优于其他注入方式。本文件不复述未经本次核对的其他论文指标。

**交付物**：明确的 V/L/A 表征、来源动作统计和可选 State 输入协议。

## 4. 输入构造与噪声时间（Token & Time Construction）

**目标**：将干净条件与带噪目标按变量组组织，保持标签与在线信息隔离。

```text
干净条件：Current Vision + Instruction + [Optional State] + [任务相关已知条件]
带噪目标：[Future Vision] + [Language Plan] + [Action]
                            ↓
                   共享双向 Joint Attention
                            ↓
                   预测所选目标的干净状态
```

- [x] 按任务选择条件组、目标组和可见性；为 V/L/A 目标独立采样时间与噪声。
- [x] 时间约定：`tau=0` 是噪声，`tau=1` 是数据；`z_m = tau_m*x_m + (1-tau_m)*epsilon_m`。
- [x] 当前 VLA 训练目标采用 `sigmoid(N(0,1))` 的时间采样，条件保持干净。
- [ ] State 的字段 Mask、样本缺失 Mask、投影和条件角色标记。

数据有效性 Mask、attention 可见性和目标损失 Mask 分别处理；不能用真实目标长度作为生成条件。State 也不能包含未来状态标签。

**交付物**：统一的条件/目标组织与可选 State batch 接口。

## 5. 模型主干（Joint Flow Backbone）

**目标**：一个 Backbone 同时支持单目标、双目标和三目标前向。

- [x] V/L/A 各自拥有输入投影、Q/K/V、时间条件和输出头，在同一次联合 Attention 中交互。
- [x] Current/Future Vision 与 Instruction/Plan 使用角色标记区分。
- [x] Action 默认通过 Linear 连接32维连续动作与主干 width；不建立单独的 Action 生成 Backbone。
- [ ] 将可选 State 投影为条件上下文，提供角色标记与 attention Mask。
- [ ] 将 State 开关、字段定义和投影配置保存到 checkpoint，并检查恢复一致性。

拟定逻辑接口：`model(conditions, noisy_targets, times, masks)`；这是设计概括，不是当前 Python 函数签名。

**交付物**：共享三流前向、可选 State 条件接口与兼容性检查。

## 6. 训练方案（Flow Training）

### 6.1 默认配置与目标损失

当前无 State 的第一版入口：`configs/vla/action_baseline.yaml`。

```yaml
action_mode: padded
action_representation: continuous
action_dim: 32
action_native_dim: 7   # 无标签推理记录的示例回退宽度，不是7D转换器
action_horizon: 16
action_adapter: continuous
vla_tasks: {action_policy: 1.0}
```

模型预测干净目标，在速度空间计算误差：

`L_m = mean_active [((x_hat_m-x_m) / max(1-tau_m,0.05))^2]`

总损失：`L = lambda_V*L_V + lambda_L*L_L + lambda_A*L_A`，只监督所选目标。

第一版 Action 对32维求平均，包括 clean target 为零的 padding 通道；padding 也加噪，不能误写成有效维度 Mask loss。语言按有效内容与 EOS 监督；各目标分别记录损失和有效 token 数。State 没有生成损失。

### 6.2 实验推进

1. B1 无 State：验证原生动作、补齐、Flow 和反归一化链路。
2. B1 有 State：State 接口完成后，在相同数据与预算下作对照。
3. B2/B3：分别验证 Future Vision 和 Language Plan 的单目标训练。
4. J1：优先比较 Action-only 与 Joint V–A；再扩展 J2/J3/J4。
5. Unified：按标签可用性与配置权重混合任务，保留 Action-only，对比正/负迁移。

阶段是实验功能顺序，不强制每阶段 checkpoint 依次初始化。若希望同一 checkpoint 支持有/无 State，可增加条件丢弃训练，单独记录丢弃概率和两种推理结果。

### 6.3 待办与交付物

- [x] 无 State 的三流 Flow loss、任务路由、优化器、梯度累积、EMA、日志与 checkpoint 接口。
- [ ] State 数据、主干和采样接入及恢复兼容测试。
- [ ] 真实数据小规模训练、单目标基线与联合训练对照。

**交付物**：可复现配置、训练日志与实际验证结果；不把接口测试当作训练收敛。

## 7. 推理与生成（Inference & Sampling）

- [x] Euler 多步采样；生成目标从独立高斯噪声开始，不编码真实目标动作。
- [x] 条件保持不变；多目标在同一次前向中交互，默认同步时间网格推进。
- [x] Action 输出 `A` 为模型宽度，`A_native` 为截回各样本原生宽度的列表。
- [ ] 推理时可选输入 State，关闭或缺失时正确屏蔽对应条件。
- [ ] Language Plan 的文字解码、机器人控制与闭环重规划评估。

对 `A_native` 使用同一来源统计反归一化，再交给匹配字段/控制约定的外部控制器。Future Vision 首版先在 latent 层评估，不要求生成像素图像。

H 是动作执行步数，默认25次 Euler 更新是去噪步数；两者不同。训练时间独立采样不等于当前采样器已经实现模态先后去噪调度。

**交付物**：统一采样接口、可选 State 输入及各任务输出检查。

## 8. 评估与消融（Evaluation & Ablation）

**目标**：先确认实现与数据正确，再衡量联合建模是否改善动作策略。

| 类别 | 检查或指标 |
|---|---|
| 数据与接口 | 时序、轨迹划分、统计、原生维度、padding/截取、未来信息隔离 |
| State 接口 | 屏蔽 State 数值变化不影响输出；有效 State 接收梯度；无 State 路径可用 |
| Action | 原生有效维度上的归一化/物理误差、闭环成功率、推理耗时 |
| Future Vision | 未来视觉 latent 误差和预测一致性 |
| Language Plan | 可解码性、任务相关性与窗口对齐 |
| 对照 | 无/有 State；Action-only/Joint；固定 checkpoint 的单/多目标采样 |

固定数据、归一化、训练预算和推理调用数再比较方法。不将全32维 padding 损失作为唯一的动作质量指标，也不将未来视觉预测混称为 proprioceptive State 预测。

- [x] 当前无 State 连续动作及已有扩展的选定代码回归测试。
- [ ] 可选 State 测试、真实机器人数据训练及闭环验证。

**交付物**：测试结果、基线指标与有控制变量的消融记录。

## 9. V0.4 开发里程碑

| 里程碑 | 内容 | 验收重点 |
|---|---|---|
| M1 | 数据与任务接口 | 真实来源的完整窗口、统计与标签需求可检查 |
| M2 | 三流与可选 State | State 开关、缺失 Mask、梯度和 checkpoint 检查 |
| M3 | Action-only | 原生动作误差、采样、反归一化与基础控制 |
| M4 | 独立 Vision / Language 基线 | B2/B3 目标与解码/评估可独立验证 |
| M5 | Joint V–A，随后 L–A/L–V–A | 相同预算下衡量实际收益 |
| M6 | Unified 与消融 | 有/无 State、多任务与推理组合对照 |

**下一项实现工作**：打通 State 的 Dataset → 条件投影 → Attention Mask → 推理输入 → checkpoint 与测试。

## 10. 后续研究与验收边界

后续再研究 State History、State Modulation/Expert、跨机器人字段语义、冻结 Action Codec、模态优先时间调度、Chunk-Causal History、Looped Refinement 或额外一致性损失。

**第一版验收标准**：直接连续 Action Policy 可训练、可采样、可还原到原生控制量；State 可选择提供或关闭且行为正确；V/L/A 单/双/三目标接口保持统一。联合训练的收益和闭环效果由后续实际实验确认。

参考：[JUICE](https://arxiv.org/abs/2610.28801)；[π0.5](https://arxiv.org/abs/2504.16054) 与 [OpenPI 官方实现](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L328)；[Dynin-Robotics](https://arxiv.org/abs/2609.13053)。π0.5 的具体padding机制依据已核对的OpenPI源码，不声称已核对其论文全文；本文件没有新增论文性能结论。
