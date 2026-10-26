# JUICE VLA 初步训练方案

第一版使用 JUICE 联合 Flow Matching，直接生成连续 Action；参考 OpenPI/π0.5 的动作补齐方式，不使用 X-Tokenizer。先验证 Action Policy，再尝试视觉、语言与动作的联合生成。

**状态说明：这是训练设计，不是已完成的机器人实验报告。连续 Action 路径已实现；本文新增的可选 Robot State 接口尚待代码实现。真实数据训练与闭环效果尚未验证。**

## 1. 数据准备

以机器人轨迹为单位，先划分训练/验证集，再切出长度为 H 的动作窗口。第一版示例 H=16。

| 数据 | 含义 | 是否必需 |
|---|---|---|
| 当前图像 I_t | 动作开始前的视觉观测 | 必需 |
| 指令 L_inst | 当前任务的语言指令 | 必需 |
| 动作 A_[t:t+H) | 未来 H 步连续控制命令，原生维度 d | Action 训练必需 |
| 当前机器人状态 S_t | 关节位置、夹爪开度或末端位姿等 | **可选** |
| 未来图像 I_(t+H) | 执行完整动作窗口后的观测 | 视觉目标任务必需 |
| 语言计划 L_plan | 与当前窗口对齐的计划或子任务描述 | 计划目标任务必需 |

统一动作字段顺序、单位、坐标系、旋转表示和控制周期；记录每个来源的归一化统计。状态必须是动作开始前可观测的 S_t，不能用未来状态 S_(t+H) 代替。当前图像与未来图像分别编码，指令与计划分别编码，避免标签泄漏。

当前代码使用冻结视觉/语言 Encoder 预先生成特征缓存；每个窗口保存一个 `.pt` 文件，由 manifest 列出路径。State 接入时，缓存需额外保存 `robot_state`、`state_present` 和 `state_valid_mask`，分别描述数值、是否提供状态和哪些状态字段有效。

## 2. 任务定义

用 C 表示公共条件 `(I_t, L_inst, 可选 S_t)`。

| 任务 | 干净条件 | 生成目标 | 阶段 |
|---|---|---|---|
| Action Policy | C | Action | 第一版默认 |
| World Model | C、已知 Action | Future Vision | 单目标扩展 |
| Language Planning | C | Language Plan | 单目标扩展 |
| Joint V–A | C | Future Vision、Action | 优先联合实验 |
| Joint L–A | C | Language Plan、Action | 后续联合实验 |
| Joint L–V–A | C | Language Plan、Future Vision、Action | 三目标实验 |

有 State 与无 State 使用相同的目标定义。已知真实 Action 只用于 World Model 等离线条件任务，在线 Action Policy 不读取真实未来动作、未来图像或计划标签。

## 3. 模态表征

- **Vision**：复用冻结 SigLIP2 或 WebSSL-DINO，使用 JUICE 的归一化连续视觉特征。
- **Language**：复用冻结 Flan-T5-Small，指令和计划使用独立编码的连续特征。
- **Action**：直接使用归一化连续控制量，不量化、不经过 Action Tokenizer。原生 `[H,d]` 在尾部补零至 `[H,32]`；32 是模型宽度，不是机器人自由度。
- **Robot State（可选设计）**：使用训练集统计单独归一化当前状态，通过轻量 Linear 投影成条件 Token。State 的维度与 Action 维度独立，不要求相同，也不复用动作归一化界限。

不同机器人若使用不同 State 字段，需显式对齐字段语义，并保留逐字段有效性 Mask；缺失字段不能被当成真实零值。第一版 State 实验先使用一个已明确字段定义的机器人，减少接口混杂。

## 4. 输入架构

```text
干净条件：当前视觉 Tokens + 指令 Tokens + [可选 State Token]
带噪目标：[Future Vision Tokens] + [Language Plan Tokens] + [Action Tokens]
                         ↓
                 同一个 Joint Flow 模型
                         ↓
              预测所选目标的干净连续状态
```

条件与目标按变量区分：当前视觉是条件，未来视觉可以是目标；指令是条件，语言计划可以是目标。

State 为可关闭的条件输入，不增加第四种生成目标。拟定开关为 `use_robot_state`，默认关闭；该配置项尚未实现。关闭 State 时不加入可见 State Token；混合 batch 中缺少 State 的样本屏蔽对应 attention key。不能仅用零向量表示缺失，因为零可能是有效状态。启用时仅投影有效字段，并提供字段有效性信息。

State 始终干净，不加 Flow 噪声，不预测未来 State，也没有 State 去噪损失。可选 State 的开关、字段定义和投影配置需保存进 checkpoint，训练和推理保持一致。

## 5. 模型主干

保留 JUICE 双向 Joint Attention。V、L、A 使用各自的输入投影、Q/K/V、时间条件和输出头，在同一次 Attention 中交换信息。

Action 的每个控制时间步投影为一个 Token，因此 H 步对应 H 个 Action Tokens。默认使用 Linear 输入/输出投影，直接监督连续动作坐标。

State 接入设计为额外干净条件 Token，加入联合注意力的上下文，通过输入投影和角色标记与其他变量区分；不另建 State 生成分支。实现时必须测试：被屏蔽的 State 数值变化不会影响输出，有效 State 能参与反向传播。

## 6. 训练方式

对每个选中的生成目标独立采样时间与高斯噪声：

`z_m = tau_m * x_m + (1 - tau_m) * epsilon_m`，其中 m 为 V、L 或 A。

`tau=0` 表示纯噪声，`tau=1` 表示干净数据。Action-only 只对 Action 加噪；联合训练分别采样 tau_V、tau_L、tau_A。所有条件保持干净。

模型预测干净目标，按当前 JUICE 实现计算速度空间误差：

`L_m = mean_active [((x_hat_m - x_m) / max(1 - tau_m, 0.05))^2]`

总损失为所选目标损失的加权和。第一版 padded 模式的动作损失覆盖32维，包括 clean target 为零的 padding 通道；这与逐维 Mask 损失是不同实验设置。

初步对照：① 无 State 的 Action-only；② 有 State 的 Action-only；③ 在相同数据和预算下增加 Joint V–A。若要同一模型同时支持有/无 State，可在 State 接口完成后加入条件丢弃训练，并分别评估两种推理输入；丢弃概率作为实验配置记录，不预设收益。

## 7. 推理与验收

Action 从 `[B,H,32]` 高斯噪声开始，经多步 Euler 更新生成；当前视觉、指令和可选 State 保持不变。输出按每个样本原生 d 维截取，再使用同一来源统计反归一化，交给匹配动作定义的外部控制器。H 个动作执行步与 Flow 去噪步数不同。

当前已实现的无 State 基线入口：`configs/vla/action_baseline.yaml`。其中 `action_mode: padded`、`action_representation: continuous`。**当前配置没有 Robot State 开关；需先完成本文定义的数据、模型和采样接口，才能运行有 State 的对照。**

验收顺序：数据时序与统计检查 → 小批量前向/反向 → 可选 State 的屏蔽与梯度测试 → 固定数据小规模训练 → 动作物理误差与闭环成功率。记录有效动作误差、各目标损失、推理耗时及有/无 State 的差异，不将代码测试通过等同于策略训练成功。
