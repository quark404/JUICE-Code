# Native Joint Flow VLA — No-State Action Baseline

将 JUICE 图文连续 Flow 扩展为 Vision–Language–Action。**A 仅表示 Action；不输入机器人 proprioceptive State，也不启用 sensor 分支。** 当前图像与未来视觉、原始指令与语言规划是不同变量组。

默认模型：`p(A_[t:t+H) | I_t, L_inst)`。Action 直接在归一化连续坐标中做 Flow；Linear 是主干维度适配，不是 Action Tokenizer。MLP 是可选边界投影，损失仍在原始归一化动作坐标上计算，避免同时训练目标编码器导致表征坍缩。

参考 Dynin-Robotics 的四项训练查询与可见/目标划分，**不复现它的离散 masked diffusion**：Dynin §3.2.1 使用均匀动作量化，本项目不分桶、不用离散 CE。Dynin §3.2.2 说明报告的训练/推理不启用 sensor；文中的 visual state 是视觉观测。

代码、可运行接口与限制见 [仓库使用说明](https://github.com/quark404/JUICE-Code/blob/codex/vla-action-baseline/README.md)。设计文档不是机器人实测结果。

## 文档索引

1. [总体目标](docs/00_overview.md)
2. [数据准备](docs/01_data_preparation.md)
3. [任务定义](docs/02_task_definition.md)
4. [模态表征](docs/03_modality_encoding.md)
5. [输入与时间](docs/04_token_construction.md)
6. [模型](docs/05_model_architecture.md)
7. [训练](docs/06_training.md)
8. [采样](docs/07_inference.md)
9. [评估](docs/08_evaluation.md)
10. [里程碑](docs/09_milestones.md)
11. [后续研究](docs/10_future_research.md)
12. [参考与差异](docs/11_references.md)
