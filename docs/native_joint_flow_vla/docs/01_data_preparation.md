# 数据准备：没有机器人 State

轨迹：`trajectory = {instruction, I_0,A_0,I_1,...,A_(T-1),I_T,subtasks}`。保留 trajectory_id、机器人型号、坐标系、控制频率、动作字段顺序与 subtask 边界。机器人型号等只作数据元信息，不作为额外输入模态。

先按轨迹划分 train/val/test，再切窗口。首版使用完整 `[t,t+H)` 动作窗口；`t+H <= T`，不跨 subtask 时再要求 `t+H <= subtask_end`。必须区分未来窗口帧 `I_(t+H)`、紧邻帧 `I_(t+1)` 和成功终点 `I_T`。Dynin WM 的 next-observation 约定与本项目 chunk-end world-model 实验不能混称。

从原始数据分别用冻结视觉编码器和 T5 编码各变量，保存一个窗口一个 `.pt` 张量字典，manifest 每行一个文件路径，路径相对 manifest。禁止将 instruction 与真实 plan 拼起来编码，否则双向 T5 会泄漏目标。视觉特征遵循 JUICE LayerNorm；语言特征除以 TEXT_LATENT_STD=0.2。

必填：
- `current_vision`: `[256,D_v]`，SigLIP2 D_v=1152，DINO=1024。
- `instruction`: `[text_canvas,512]`，独立编码。
- `instruction_mask`: `[text_canvas]`，内容与 EOS 的有效标记；应显式提供。

按任务提供：
- `action`: `[H,d_a]`，连续动作归一化到 [-1,1]，不量化。
- `future_vision`: `[256,D_v]`，下一观测/窗口末端帧，须在元信息声明 target_step。
- `goal_vision`: `[256,D_v]`，成功终点/子任务目标，与 future_vision 独立保存。
- `language_plan`: `[text_canvas,512]`，独立编码；`language_plan_mask` 包含内容+EOS。
- `action_mask`、`future_vision_mask`、`current_vision_mask` 可指定 token 有效性；缺省整组有效。Baseline 使用完整动作块。

缺失的目标组用零占位并标记 absent；缺标签的样本不会被分配给需要它的任务。四任务混合中的 Task Understanding 以 instruction 为目标，不要求 language_plan。GP 使用 goal_vision，不会误用 future_vision。模型不读取 trajectory_id 等元数据；记录它们用于预处理和审计。

动作界限只用训练集统计，运动通道可参考 Dynin p01/p99。夹爪保持绝对控制语义，不能累加为 delta。保存 lower/upper、字段/单位/坐标系、频率、source_id；推理必须使用同一变换。`ActionNormalizer` 提供裁剪、归一化/反归一化，常量通道归零。异构机器人需上游统一动作接口；本版不自动转换。

原始 trajectory/window 构造器、多相机/多帧 token 组织与自动时序校验尚未实现；不能把缓存接口等同于完整机器人数据流水线。
