# 异构 Action 的论文依据与官方实现对照

本页区分论文方法、已核对的官方代码和本仓库设计。统一数值宽度不等于自动统一物理控制语义。没有将全部动作维度假定为可互换的 Cartesian fields。

## π0.5 / OpenPI：固定模型宽度＋机器人数据适配

论文：[π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)。官方入口：[OpenPI README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md)。本次论文 PDF 下载受网络 403 限制；以下具体实现结论依据官方源码，未声称已从论文全文核对这些代码细节。

核对的官方源码版本 `215abfb217dbac7d5f1273282331b9b1866c0479`：

- [pi0_config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)：模型默认 action_dim=32；pi05 对共享 flow 模型作专门配置。
- [transforms.py / PadStatesAndActions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L328)：将 state/actions 补齐到模型动作宽度。
- [training/config.py / ModelTransformFactory](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py#L108)：PI05 路径明确调用该 padding transform；数据适配器另行重排字段、归一化、选择 delta/absolute 转换。
- [droid_policy.py / DroidOutputs](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py)：示例输出只取前 8 个原生动作维度。这不是 Dynin 的 7D EEF 接口。
- [pi0.py / compute_loss](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L189)：所核对路径对动作维度执行 squared-error mean；不能将本项目的逐维 Mask 与按样本有效元素平均说成复制自它的 loss。

借鉴范围：固定主干宽度、在数据/输出边界处理机器人原生维度。我们选择显式语义字段映射与 Mask，是针对缺失维度的额外设计；不复制它的 State 条件，也不声称不同机器人的原生关节变量必须具有同一语义。

## Dynin-Robotics：统一 7D 末端控制接口

论文：https://arxiv.org/abs/2609.13053，本次依据用户提供 PDF 的 §3.2.1、§3.2.2、§4.1 核对。

来源专属转换先把动作映射为 translation(3)+rotation(3)+gripper(1)，运动通道用 per-source p01/p99，夹爪保持 absolute，常量或未用通道置零，再逐标量均匀量化。双臂来源也规约到同一 7D view。因此数据来源机器人 DoF 多样不等于模型原生支持任意输出宽度。报告配置不启用 sensor。

借鉴范围：优先确保控制语义、统计和坐标系统一；不继承其离散动作量化。

## X-Tokenizer：具名 26D 布局＋DOF Mask＋机器人身份

论文：https://arxiv.org/abs/2606.14752，用户提供 v2 PDF。§3.1/3.2 定义 E→SRQ→D；§3.4 的 frozen tokenizer 是训练期离散语义监督，并非本文新实现的 latent Flow。

核对的官方源码版本 `9733990b16a07d51fb8b5d305ad8c16abb57aef2`：

- [README](https://github.com/X-Square-Robot/X-Tokenizer/blob/9733990b16a07d51fb8b5d305ad8c16abb57aef2/README.md)：26D 布局、缺失部件的 DOF Mask、rotation 6D 与统计流程。
- [action_layout.py](https://github.com/X-Square-Robot/X-Tokenizer/blob/9733990b16a07d51fb8b5d305ad8c16abb57aef2/xtokenizer/data/action_layout.py)：按具名字段组织双臂、底盘、升降和头部，按字段选择 SO(3) delta/线性差值/保留 absolute 等操作。
- [api.py](https://github.com/X-Square-Robot/X-Tokenizer/blob/9733990b16a07d51fb8b5d305ad8c16abb57aef2/xtokenizer/api.py)：encode 暴露 continuous_features、quantized_features、indices；**公共 decode 接收 indices，不接收任意连续向量**。
- [model/tokenizer.py](https://github.com/X-Square-Robot/X-Tokenizer/blob/9733990b16a07d51fb8b5d305ad8c16abb57aef2/xtokenizer/model/tokenizer.py)：底层 decode 接收 quantized_features；decode_from_indices 先重建码向量再调用它。原生 encoder 接收 obs_state/robot_type_id，但不会自动根据 dof_mask 清零输入。
- [robot_types.yaml](https://github.com/X-Square-Robot/X-Tokenizer/blob/9733990b16a07d51fb8b5d305ad8c16abb57aef2/xtokenizer/configs/robot_types.yaml)：机器人身份注册表；不能将本项目 layout ID 直接视为这个预训练 embedding ID。

借鉴范围：显式标准字段、缺失维度屏蔽、机器人身份上下文。论文 anchor 的依赖和官方 API 的可选 obs_state 不等价于已验证无 anchor 的重建能力。

## 本项目的路径

| 路径 | 模型 A 空间 | 维度统一方式 | 依据/边界 |
|---|---|---|---|
| V1 padded continuous | 原生连续动作尾部补齐到32D | padding后直接Flow，输出截回原生宽度 | OpenPI式动作接口，不强制7D转换 |
| 现有 fixed continuous | 归一化原生连续动作 | 同一模型固定 action_dim | 原有 baseline 保留 |
| heterogeneous continuous | 公共字段连续动作 | 原生字段→canonical schema，逐维 Mask，layout embedding | 参考标准布局/padding，Mask loss 是本项目设计 |
| frozen X-Tokenizer | 冻结 Q(E(A)) 的 latent | 上游先组织成 checkpoint 26D，再用 DOF Mask 与预训练 robot ID | 冻结模块调用有源码依据；latent Flow 效果尚待实验 |

后两项可以分别使用，但不会把原始动作维度 Mask 直接套到 latent 维度。Frozen codec 支持已经组织为同一 checkpoint 布局的不同机器人样本；自动原生布局 packing 使用前一条路径，或者在缓存生成时显式做一次转换。
