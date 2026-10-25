# 模态表征：V / L / A

V：复用冻结 SigLIP2 或 WebSSL-DINO，当前图像和目标图像独立编码。当前/目标 token 用角色 embedding 区分；每个视觉组保留二维位置编码。

L：复用冻结 Flan-T5-Small。Instruction 与 Plan 独立编码；Instruction 是条件时使用有效内容+EOS Mask。TU 中 Instruction 是生成目标，不能作为可见条件。目标语言全 canvas 加噪并参与注意力，损失只计算有效 token；目标真实长度不通过 attention mask 暴露。

A0（默认）：训练集界限归一化的连续 Action Chunk `[H,d_a]`。高斯噪声、Flow 插值和监督均直接在这些坐标中计算。每个时间步一个 Action Token，Linear 映射到 Transformer width，再由 Linear 输出物理维度。

A0b（可选）：`action_adapter: mlp`，输入与输出用可训练 MLP，但仍监督连续动作，不把可训练隐变量当作 Flow 标签。该设计不是预训练 tokenizer，也不是可变压缩率 latent-action 模型。

Dynin §3.2.1：按通道稳健缩放后用 B_A bins 均匀量化，每个动作标量一个离散 token；Stage 1 用 256 bins。本版只借鉴缩放与任务布局，不引入量化、codebook、离散 CE 或 expected-bin MAE 辅助项。

Robot State / Sensor：完全不输入，不保留 State Encoder 或 State Token。Dynin 的 optional sensor interface 在报告配置中未激活。后续如果引入预训练 Action VAE/Tokenizer，应单独验证动作语义、重建、冻结接口和控制效果，不能视为已完成能力。
