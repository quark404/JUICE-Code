# 后续研究（不属于当前 Baseline）

可研究短时历史、多视角、连续 Action VAE、跨机器人动作接口、Language-first/Vision-first 调度、目标引导生成及候选筛选、物理一致性和循环细化。

机器人 State/Sensor 不属于本版输入或待安装依赖。如果未来研究要添加，必须显式改接口并单独消融，不能默认开启。

Dynin 的联合 Action–World inference-only composite 与本项目联合监督训练是不同实验变量，需分别报告。没有预设 Joint 一定优于 Action-only。
