"""Euler sampling for normalized V/L/A states; no robot State input."""
import torch
from juice.train.vla import TASKS


@torch.no_grad()
def sample_vla(model, enc, task="action_policy", steps=25, generator=None, noise_scales=None):
    """Use the same frozen-feature batch as training. Targets are initialized from
    fresh noise; supplied target values only specify shapes, never conditions.
    Returns V/L/A normalized continuous states (no pixel/text/robot decoder).
    """
    if steps <= 0 or task not in TASKS:
        raise ValueError("invalid task or step count")
    clean, targets, use_instruction = TASKS[task]
    scales = {"V": 1., "L": 2., "A": 1.} if noise_scales is None else noise_scales
    states = {m: enc[m] for m in "VLA"}
    validity = dict(enc["valid"])
    if task in ("goal_prediction", "goal_conditioned_action"):
        states["V"] = enc["goal_vision"]
        validity["V"] = enc["goal_valid"]
    if task == "task_understanding":
        states["L"] = enc["instruction"]
        validity["L"] = enc["instruction_mask"]
    x, masks = {}, {}
    for m in "VLA":
        if m in clean and not bool((enc["goal_present"] if m == "V" and task == "goal_conditioned_action" else enc["present"][m]).all()):
            raise ValueError(f"missing clean condition {m}")
        ref = states[m]
        if m in targets:
            x[m] = torch.randn(ref.shape, generator=generator, device=ref.device, dtype=ref.dtype) * scales[m]
            masks[m] = torch.ones(ref.shape[:2], device=ref.device, dtype=torch.bool)
        elif m in clean:
            x[m], masks[m] = ref.clone(), validity[m]
        else:
            x[m], masks[m] = torch.zeros_like(ref), torch.zeros(ref.shape[:2], device=ref.device, dtype=torch.bool)
    instruction_mask = enc["instruction_mask"] if use_instruction else torch.zeros_like(enc["instruction_mask"])
    for i in range(steps):
        times = {m: x[m].new_full(x[m].shape[:2], i / steps if m in targets else 1) for m in "VLA"}
        outputs = model(x["V"], times["V"], x["L"], times["L"], txt_key_mask=masks["L"],
            action=x["A"], t_action=times["A"], image_key_mask=masks["V"], action_key_mask=masks["A"],
            image_condition=enc["current_vision"], language_condition=enc["instruction"],
            image_condition_mask=enc["current_vision_mask"], language_condition_mask=instruction_mask,
            objective=torch.full((enc["A"].shape[0],), list(TASKS).index(task), device=enc["A"].device, dtype=torch.long))
        for m, pred in zip("VLA", outputs):
            if m in targets:
                x[m] = x[m] + (pred - x[m]) / max(1 - i / steps, .05) / steps
    return x
