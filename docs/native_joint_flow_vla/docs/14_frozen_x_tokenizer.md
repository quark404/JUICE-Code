# 冻结 X-Tokenizer 作为 Action 编解码器

## 结论与选择

不训练 tokenizer；E、SRQ 和 D 全部冻结、eval，且不注册到 JUICE 模型参数或优化器里。JUICE 自己的 Action flow/projection/robot-context 参数仍然训练。

选择 `z_q = SRQ(E(A_delta, anchor?, robot_id))` 作为 Flow 的固定标签：

```text
训练：normalized A_delta → frozen E → frozen SRQ → z_q / scale → JUICE Action Flow
推理：噪声 → JUICE → predicted z_q / scale → frozen D → normalized A_delta
```

Action 的 Flow loss 在 latent 坐标中计算，不是原生动作空间 MSE；本版不穿过冻结 Decoder 添加反传损失。原生动作误差由独立重建检查和后续策略评估衡量。

选择量化后的码向量和，而非 indices，也不把量化前 continuous_features 直接交给 D。依据是官方 Decoder 正是在 z_q 上训练的。该方案只保留 frozen 编解码接口，不增加离散 CE 或 tokenizer 训练。

**这是本项目的 latent Flow 改造，不是 X-Tokenizer 论文已验证的下游训练方法。** 论文是 frozen discrete-code 辅助监督＋独立连续动作专家，推理时不用 tokenizer。我们生成的任意 latent 可能偏离码向量分布，不能因 Decoder 已预训练就保证生成动作准确。本版直接 D 解码，不额外对 z_q 再执行一次 RVQ；重新量化不一定幂等，也不自动提高控制精度。

## 与无 State 要求的关系

默认 `x_tokenizer_use_anchor: false`，整个 codec 调用传 obs_state=None，JUICE 不增加 State Token。但这只对应官方支持的接口调用，不代表已验证预训练模型无 anchor 也能高质量重建。

若重建不足，可选择保留 direct-action baseline，或明确启用 `x_tokenizer_use_anchor: true`，让 codec 使用 `[B,1,26]` 的归一化参考观测；这个 anchor 不加入 Joint Attention，但系统整体就不再是完全不依赖 proprioception，必须在实验中单独标注。不要以启用 anchor 的结果声称“无 State baseline”。

即使 encoder/decoder 不接收 anchor，把绝对命令转换为相对动作、再还原到绝对命令的外部控制流程，也可能需要参考帧。直接生成已有 delta 控制时可保持外部 delta 接口；不能混用 absolute 和 delta 标签。

## 数据与接口

示例：`configs/vla/x_tokenizer_frozen.yaml`。`action_dim` 是 checkpoint 原生输入宽度（发布版26），模型的 A hidden width 和 latent token 数由 checkpoint architecture 推断，不需要手工猜测。

缓存仍保存当前视觉、指令和原生动作特征；其中 action 必须已经是**符合 X-Tokenizer 字段顺序与归一化统计的 normalized delta**。不能把 Dynin p01/p99 的 7D delta 直接送入该26D checkpoint。官方代码使用自己的字段 delta 约定和 q001/q999 统计；SO(3) rotation 6D 不等于三个旋转角补零。

新增可选字段：

- `codec_dof_mask: bool[H,26]`，默认全部有效；不同机器人可以缺少不同身体部件。缓存中无效字段预先清零；输入字段仍须有限且在 [-1,1]。
- `codec_robot_type_id: int`，对应预训练 tokenizer 注册表，可覆盖配置默认 ID。它不是 ActionSchema 的 layout ID。
- `codec_obs_state: float[1,26]`，仅 anchor 模式允许，必须使用相同参考观测归一化约定。

支持同一 batch 不同 robot ID 和不同 DOF Mask，但所有记录都先转换为同一个 checkpoint 的26D布局。训练器将 raw Action 转成 frozen latents 后走同一个三流 Loss。JUICE Action 输入也接受原生字段 Mask 和 robot ID 的上下文 embedding，以便生成时知道目标 embodiment；它们是已知元信息，不是 State。

当前要求完整有效时间窗口；不支持 codec 内部分时间 padding。latent 的每个坐标不是独立物理关节，因此不会把 `[H,26]` Mask 误用于 `[H/ratio,latent_dim]` Loss。

## 预训练权重和配置校验

需要用户提供本地官方格式 checkpoint，以及与数据匹配的 X-Tokenizer 库和预处理统计。当前任务不安装环境、不下载权重。包装器用 weights_only=True 加载，严格 load_state_dict，不回退为任意 pickle 反序列化。

codec checkpoint 的 SHA256、维度、horizon、latent scale、anchor设置和 robot 类型容量保存到 JUICE architecture。Resume 和推理校验一致性，codec 权重不复制进训练 checkpoint；部署时仍需保留完全相同的外部文件。

`x_tokenizer_latent_scale` 是固定正标量；若调整，应仅根据训练集编码统计校准并保存。Qwen 视觉语言对齐也不等于 JUICE 的 SigLIP/T5 已对齐，效果需要独立验证。

## 启用前重建检查

```bash
python -m scripts.validate_x_tokenizer \
  --config configs/vla/x_tokenizer_frozen.yaml \
  --record data/robot/window.pt --device cpu
```

它执行真实 E→SRQ→D，输出有效原生动作元素的 MAE/RMSE/max error。按控制精度确定是否可接受；不以一个任意统一阈值替代任务验收。需要覆盖不同机器人、轨迹、夹爪和运动范围。

通过重建检查后，再开始 JUICE 训练：

```bash
python -m scripts.train.train --config configs/vla/x_tokenizer_frozen.yaml
```

推理：先按配置构造 `FrozenXTokenizer.from_config(cfg, device)`，恢复同一 architecture/weights 的 JUICE，再调用 `sample_vla(model.eval(), raw_batch, action_codec=codec)`。返回 `A` 为解码后的 normalized delta，另有 `A_latent`。生成 Action 时绝不会把真实 Action 输入 Encoder；WM 的已知 Action 条件才会编码。

仍需按同一统计反归一化，并由外部动作适配/控制器执行，不自动接管机器人。

## 已验证和未验证

合成 backend 测试验证冻结/eval、只使用 quantized_features、尺度与形状、梯度和优化器隔离、Mask、anchor选择、目标信息隔离、WM条件编码、checkpoint指纹及配置检查。官方源码接口已核对。

**真实 pretrained checkpoint 尚未提供/运行，因此真实重建、无 anchor 质量、latent Flow 策略和闭环成功率未验证。** 不能将合成接口测试说成预训练效果已通过。固定连续动作和异构连续动作分支可独立使用，不受此依赖影响。
