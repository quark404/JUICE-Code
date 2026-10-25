"""Euler sampling for normalized V/L/A states; no robot State input."""
import torch
from juice.train.vla import TASKS


@torch.no_grad()
def _sample_states(model, enc, task="action_policy", steps=25, generator=None, noise_scales=None):
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
    action_kwargs = {}
    dim_mask = enc.get("action_dim_mask")
    if dim_mask is not None:
        action_kwargs = {"action_dim_mask": dim_mask, "action_layout_ids": enc["action_layout_ids"]}
    if "action_codec_mask" in enc:
        action_kwargs.update(action_codec_mask=enc["action_codec_mask"], action_codec_ids=enc["action_codec_ids"])
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
    if dim_mask is not None:
        x["A"] = torch.where(dim_mask, x["A"], torch.zeros_like(x["A"]))
        masks["A"] &= dim_mask.any(-1)
    instruction_mask = enc["instruction_mask"] if use_instruction else torch.zeros_like(enc["instruction_mask"])
    for i in range(steps):
        times = {m: x[m].new_full(x[m].shape[:2], i / steps if m in targets else 1) for m in "VLA"}
        outputs = model(x["V"], times["V"], x["L"], times["L"], txt_key_mask=masks["L"],
            action=x["A"], t_action=times["A"], image_key_mask=masks["V"], action_key_mask=masks["A"],
            image_condition=enc["current_vision"], language_condition=enc["instruction"],
            image_condition_mask=enc["current_vision_mask"], language_condition_mask=instruction_mask,
            objective=torch.full((enc["A"].shape[0],), list(TASKS).index(task), device=enc["A"].device, dtype=torch.long), **action_kwargs)
        for m, pred in zip("VLA", outputs):
            if m in targets:
                x[m] = x[m] + (pred - x[m]) / max(1 - i / steps, .05) / steps
                if m == "A" and dim_mask is not None:
                    x[m] = torch.where(dim_mask, x[m], torch.zeros_like(x[m]))
    return x


@torch.no_grad()
def sample_vla(model, enc, task="action_policy", steps=25, generator=None, noise_scales=None, action_codec=None):
    """Sample physical normalized actions, or frozen-codec latents followed by decoding.

    Codec mode additionally returns A_latent. No ground-truth target action is encoded.
    Returned codec actions are normalized delta controls, not absolute robot commands.
    """
    representation = getattr(model.cfg, "action_representation", "continuous")
    if action_codec is None:
        if representation == "x_tokenizer":
            raise ValueError("sample_vla requires the matching frozen action_codec")
        result = _sample_states(model, enc, task, steps, generator, noise_scales)
        if getattr(model.cfg, "action_mode", "fixed") == "padded":
            from juice.action import unpad_actions
            result["A_native"] = unpad_actions(result["A"], enc["action_native_dims"])
        return result
    if representation != "x_tokenizer" or model.cfg.action_codec != action_codec.metadata:
        raise ValueError("model and action codec metadata do not match")
    if task not in TASKS:
        raise ValueError("invalid task")
    clean, targets, _ = TASKS[task]
    latent_batch = action_codec.encode_batch(enc, encode_actions="A" in clean)
    out = _sample_states(model, latent_batch, task, steps, generator, noise_scales)
    out["A_latent"] = out["A"]
    if "A" in targets:
        out["A"] = action_codec.decode(out["A_latent"], enc)
    elif "A" in clean:
        out["A"] = torch.where(enc["codec_dof_mask"], enc["A"], torch.zeros_like(enc["A"]))
    else:
        out["A"] = torch.zeros_like(enc["A"])
    return out
