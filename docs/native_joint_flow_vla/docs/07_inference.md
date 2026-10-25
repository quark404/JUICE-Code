# 推理与采样

`juice.vla_sampling.sample_vla(model, enc, task="action_policy", steps=25)` 用与训练一致的 frozen-feature batch，目标初值是独立高斯噪声；传入目标值只用于形状，不作为真实标签条件。

条件保持不变，用 clean-state prediction 构造速度并做 Euler 更新。默认同步时间网格；训练目标各自独立采样时间。噪声尺度默认 V=1、L=2、A=1，自定义训练尺度必须通过 noise_scales 传入采样。

输出归一化连续 V/L/A；Action 需用训练期 lower/upper 反归一化，再由外部控制器执行。Future Vision 首版只评估 latent，Language Plan 未自动解码为文字。

支持指定联合任务采样；goal_conditioned_action 使用已知 goal_vision。完整“先生成目标再策略采样”的封装、候选重排、置信度筛选和缓存加速未实现。Dynin 离散 token commitment 不能直接搬作连续 Euler 算法。
