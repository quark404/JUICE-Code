# 训练与 Baseline

入口：`python -m scripts.train.train --config configs/vla/action_baseline.yaml`。分布式沿用 torchrun。action_dim>0 时使用 frozen-feature `.pt` manifest loader，不在训练时重新下载编码器；action_dim=0 保持原 WebDataset 训练路径。

每个样本按配置权重在标签齐备的任务中采样，没有可用任务则报错。默认 `vla_tasks: {action_policy: 1.0}`。更换为 world_model、language_planning 可做独立单目标基线；joint_va/joint_lva 是本项目额外训练实验。

预测 clean state，以速度空间误差训练：`((x0_hat-x0)/max(1-tau,0.05))^2`。每流先对维度平均，再对 active token 平均，最后乘 lambda_img/lambda_txt/lambda_action。条件组没有目标损失。每个任务记录 rows，每个模态记录 loss/active tokens。

`configs/vla/dynin_objectives.yaml` 提供 PO/WM/GP/TU 等权任务采样示例，不是论文优化配置。论文 §4.2 的 objective schedule multiplicity、lambda=(1,.2,.05,.2)、离散 inverse-mask-rate CE 和 Policy bin MAE 与本项目不同。不要把这些 loss 系数直接当作任务抽样概率。

实际训练前需要数据缓存、训练集归一化统计和 NVIDIA GPU。CPU 测试只验证接口、反向传播及信息隔离，不证明策略已训练成功。Baseline 的 action_dim=7 是例子，必须按机器人动作布局修改。

可选异构模式增加逐元素 Action Mask，并采用先样本内、再样本间平均的动作损失；固定模式公式不变。接口与迁移见 [异构 Action](12_heterogeneous_actions.md)。
