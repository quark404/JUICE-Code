# 异构 Action：额外可选分支

## 两种模式

- `action_mode: fixed`（默认）：保持原 `[H,action_dim]` 动作接口、Linear/MLP 参数、损失和采样。现有配置无需修改。
- `action_mode: heterogeneous`：同一模型、同一 batch 支持不同原生动作维度，通过显式字段布局映射到统一宽度。

这里的异构是**数据接口与有效维度的异构**。Transformer 内仍是固定宽度 Tensor；不引入 X-Tokenizer、离散量化、机器人 State 或第四模态。

## 依据

具体论文与官方源码参见 [π0.5 / Dynin / X-Tokenizer 对照](13_action_references.md)。公共字段与 DOF Mask 借鉴标准接口；逐样本有效元素平均是本项目的设计，不声称是 π0.5 的原始损失。

## 为什么选择统一布局

参考 Dynin 的“先统一控制语义”和 X-Tokenizer 的“标准字段布局＋DOF Mask”，采用 canonical scalar fields 的并集。同名字段必须具备相同物理语义、单位和坐标系。不能将 joint angle 与 Cartesian displacement 仅因为都是标量就映射为同一字段。

示例 `configs/vla/heterogeneous_action.yaml` 使用 14D 双臂布局：单臂记录原生 7D，双臂记录原生 14D，均映射到 `[H,14]`。可以定义 26D 布局，也可以注册不同原生顺序。字段名不自动执行旋转转换、正逆运动学或频率重采样，这些工作由上游负责。

## 配置和缓存记录

配置中：

```yaml
action_mode: heterogeneous
action_dim: 3
action_fields: [dx, dy, gripper]
action_layouts:
  planar: [dx, dy, gripper]
  one_axis: [gripper, dx]  # 明确原生字段顺序，而非简单在末尾补零
```

一条 frozen-feature 记录新增：

```python
record["action_layout"] = "one_axis"
record["action"] = normalized_native_action  # [H,2]，按该 layout 的原生顺序
record["action_dim_mask"] = bool_native_mask # 可选 [2] 或 [H,2]
record["action_mask"] = valid_time_steps     # 可选 [H]，与逐维 Mask 取交集
```

不同记录可使用不同原生宽度，但 H 仍使用配置的固定 horizon。缺失动作标签时可省略 action，必须保留 action_layout；结构 Mask 仍保留用于推理。没有有效动作元素的布局/记录会报错。有效元素必须是有限数且归一化到 [-1,1]；被 Mask 的无效元素允许 NaN，加载时置零。

DataLoader 统一输出 `A: [B,H,D]`、`action_dim_mask: bool[B,H,D]`、`action_layout_ids: long[B]`。layout ID 由名称排序得到，与 YAML 字典顺序无关。`valid[A]` 表示有标签的有效时间步；`action_dim_mask` 表示结构/元素有效性，不能因为推理时没有标签就将其全部置假。

## 模型、损失和推理

异构 Adapter 在原 Linear/MLP 上加入 Mask 投影和 layout embedding，使有效零动作区别于缺失维度，也区分不同控制布局。所有有效 Action Token 仍参与三流 Joint Attention；完全无效的时间步不作为 Action attention key。

加噪前后将无效动作元素置零；Loss 只计算生成目标的有效元素。每个样本先对有效 `(time,dimension)` 元素平均，再对具有动作目标的样本平均：

`L_A = mean_b [sum_(h,d) mask_bhd * velocity_error_bhd^2 / sum_(h,d) mask_bhd]`

这样高维机器人不因输出维度更大而自动获得更多权重。固定模式继续使用原来的损失公式。日志增加 active_A_elements。

`sample_vla` 使用与训练相同的维度 Mask 和 layout ID，在初始化和每个 Euler 步骤保持无效维度为零。已知动作作为 WM 条件时也执行相同屏蔽。输出为 canonical 归一化动作：

```python
from juice.action_schema import ActionSchema
schema = ActionSchema(cfg.action_fields, cfg.action_layouts)
canonical = sample_vla(model.eval(), batch)["A"]
native = schema.unpack(canonical[row], "one_axis")  # [H,2]，取回原生顺序
```

不同机器人输出宽度不同，所以应逐样本解包成列表，不能强行 stack。unpack 仅选字段，不执行反归一化、控制转换或逆运动学；应使用每个数据源的训练集统计再交给相应控制器。

## Checkpoint 和迁移

完整 architecture 会保存 action_mode、action_fields、action_layouts。Resume 在加载权重前校验它们，即使宽度相同也拒绝字段顺序/映射变更。旧 fixed checkpoint 未包含新增字段时仍按 fixed 解释。

异构模式支持从匹配 schema 的 checkpoint `--resume`。要迁移固定宽度或不同布局的模型，显式设置 `init_from` 与 `init_backbone_only: true`：只迁移兼容的 V/L 主干，所有 Action 专属参数重新初始化。不会隐式按位置迁移动作权重。

固定模式不会新增 Mask/Layout 参数。已有的 fixed Action 与旧图文 checkpoint 行为继续保留。扩充 schema 或 registry 会影响模型接口，必须显式迁移，不能只改 YAML 后继续 resume。

## 验证与边界

测试覆盖不同原生维度混合加载、非尾部字段映射、无标签生成结构、逐维/时间 Mask、无效 NaN 隔离、12 类任务的反向传播和优化器更新、按样本平衡 Loss、条件动作屏蔽、采样零约束、schema 校验、固定模式回归及主干迁移。

尚未验证真实混合机器人数据、GPU DDP、闭环控制或跨机器人迁移收益。布局统一只解决数值接口；动作语义转换、训练统计、坐标系/控制频率和有效字段元信息仍须由数据准备确保。

第一版训练配置现使用单独的padded direct-action模式；本页的fixed默认指省略action_mode时的兼容默认值。详见 [V1说明](15_v1_direct_actions_and_7d.md)。
