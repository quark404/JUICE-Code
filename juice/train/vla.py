"""Variable-group flow objectives. No proprioceptive State input.

Current vision / instruction are separate from future vision / language plan.
Actions always use normalized physical coordinates; adapters live inside MMRiT.
"""
import math
import torch
from juice.train.loss import T_EPS, sample_time

# (extra clean target groups, generated groups, use instruction)
TASKS = {
    "action_policy": ("", "A", True),
    "world_model": ("A", "V", True),
    "language_planning": ("", "L", True),
    "inverse_dynamics": ("V", "A", True),
    "goal_prediction": ("", "V", True),
    "task_understanding": ("", "L", False),
    "plan_conditioned_action": ("L", "A", True),
    "goal_conditioned_action": ("V", "A", True),
    "joint_va": ("", "VA", True),
    "joint_la": ("", "LA", True),
    "joint_lv": ("", "LV", True),
    "joint_lva": ("", "VLA", True),
}


def validate_tasks(tasks):
    if not tasks or any(not math.isfinite(w) or w < 0 for w in tasks.values()) or sum(tasks.values()) <= 0:
        raise ValueError("vla_tasks needs finite nonnegative weights with positive total")
    if set(tasks) - TASKS.keys():
        raise ValueError(f"unknown VLA tasks: {set(tasks) - TASKS.keys()}")


def vla_flow_loss(model, enc, cfg):
    states = {m: enc[m].float() for m in "VLA"}
    heterogeneous = getattr(cfg, "action_mode", "fixed") == "heterogeneous"
    action_kwargs = {}
    if heterogeneous:
        dim_mask = enc["action_dim_mask"]
        action_kwargs = {"action_dim_mask": dim_mask, "action_layout_ids": enc["action_layout_ids"]}
        states["A"] = torch.where(dim_mask, states["A"], torch.zeros_like(states["A"]))
    if "action_codec_mask" in enc:
        action_kwargs.update(action_codec_mask=enc["action_codec_mask"], action_codec_ids=enc["action_codec_ids"])
    b, device = states["A"].shape[0], states["A"].device
    tasks = list(cfg.vla_tasks)
    def available(task, modality):
        if modality == "V" and task in ("goal_prediction", "goal_conditioned_action"):
            return enc["goal_present"]
        if modality == "L" and task == "task_understanding":
            return torch.ones(b, device=device, dtype=torch.bool)
        return enc["present"][modality]
    eligible = torch.stack([torch.stack([available(t, m) for m in set(TASKS[t][0] + TASKS[t][1])]).all(0)
                            for t in tasks], 1)
    weights = torch.tensor(list(cfg.vla_tasks.values()), device=device)[None] * eligible
    if bool((weights.sum(1) == 0).any()):
        raise ValueError("sample has no eligible configured task; supply required target groups or change task mixture")
    choices = torch.multinomial(weights, 1).squeeze(1)
    objective = torch.tensor([list(TASKS).index(t) for t in tasks], device=device)[choices]
    goal_rows = torch.tensor([t in ("goal_prediction", "goal_conditioned_action") for t in tasks], device=device)[choices]
    understanding = torch.tensor([t == "task_understanding" for t in tasks], device=device)[choices]
    states["V"] = torch.where(goal_rows[:, None, None], enc["goal_vision"].float(), states["V"])
    states["L"] = torch.where(understanding[:, None, None], enc["instruction"].float(), states["L"])
    times, noisy, visible, active = {}, {}, {}, {}
    for m, x in states.items():
        target = torch.tensor([m in TASKS[t][1] for t in tasks], device=device)[choices, None]
        condition = torch.tensor([m in TASKS[t][0] for t in tasks], device=device)[choices, None]
        valid = enc["valid"][m].bool() & enc["present"][m][:, None]
        if m == "V":
            valid = torch.where(goal_rows[:, None], enc["goal_valid"], valid)
        if m == "L":
            valid = torch.where(understanding[:, None], enc["instruction_mask"], valid)
        active[m] = target & valid
        t = sample_time(b, 0., 1., 1., device)[:, None].expand(x.shape[:2])
        # No target length is exposed through attention or time embeddings.
        times[m] = torch.where(target, t, torch.ones_like(t))
        visible[m] = (condition & valid) | target.expand_as(valid)
        scale = {"V": cfg.noise_scale, "L": cfg.text_noise_scale, "A": cfg.action_noise_scale}[m]
        z = times[m][..., None] * x + (1 - times[m][..., None]) * torch.randn_like(x) * scale
        noisy[m] = torch.where(visible[m][..., None], z, torch.zeros_like(z))
        if m == "A" and heterogeneous:
            noisy[m] = torch.where(dim_mask, noisy[m], torch.zeros_like(noisy[m]))
            visible[m] &= dim_mask.any(-1)
    use_instruction = torch.tensor([TASKS[t][2] for t in tasks], device=device)[choices, None]
    instruction_mask = enc["instruction_mask"] & use_instruction
    prediction = model(noisy["V"], times["V"], noisy["L"], times["L"], txt_key_mask=visible["L"],
        action=noisy["A"], t_action=times["A"], image_key_mask=visible["V"], action_key_mask=visible["A"],
        image_condition=enc["current_vision"], language_condition=enc["instruction"],
        image_condition_mask=enc["current_vision_mask"], language_condition_mask=instruction_mask, objective=objective, **action_kwargs)
    loss, metrics = 0., {}
    for m, pred in zip("VLA", prediction):
        delta = (pred - states[m]) / (1 - times[m][..., None]).clamp_min(T_EPS)
        weight = active[m].float()
        if m == "A" and heterogeneous:
            elements = dim_mask & active[m][..., None]
            error = torch.where(elements, delta, torch.zeros_like(delta)).square()
            counts = elements.sum((1, 2))
            per_row = error.sum((1, 2)) / counts.clamp_min(1)
            value = per_row.sum() / (counts > 0).sum().clamp_min(1)
            metrics["active_A_elements"] = counts.sum().detach()
        else:
            error = delta.square().mean(-1)
            value = (error * weight).sum() / weight.sum().clamp_min(1)
        loss = loss + {"V": cfg.lambda_img, "L": cfg.lambda_txt, "A": cfg.lambda_action}[m] * value
        metrics[f"loss_{m}"] = value.detach()
        metrics[f"active_{m}"] = weight.sum().detach()
    for i, task in enumerate(tasks):
        metrics[f"rows_{task}"] = (choices == i).sum().detach()
    metrics["loss"] = loss.detach()
    return loss, metrics
