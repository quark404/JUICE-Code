"""Explicit scalar-field layouts for heterogeneous continuous actions.

A schema aligns semantics, not merely vector lengths. Upstream data adapters must
convert units, frames, rotation conventions and control rates before normalization.
"""
import torch
from torch import nn
from juice.action import ActionAdapter


class ActionSchema:
    def __init__(self, fields, layouts):
        if (not isinstance(fields, list) or not fields or any(not isinstance(f, str) or not f.strip() for f in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError("action_fields must contain unique nonempty scalar field names")
        if not isinstance(layouts, dict) or not layouts:
            raise ValueError("action_layouts must be a nonempty mapping")
        for name, names in layouts.items():
            if (not isinstance(name, str) or not name.strip() or not isinstance(names, list) or not names
                    or any(not isinstance(f, str) for f in names) or len(set(names)) != len(names)
                    or set(names) - set(fields)):
                raise ValueError(f"invalid action layout {name!r}")
        self.fields = list(fields)
        self.layouts = {k: list(v) for k, v in layouts.items()}
        self.names = sorted(layouts)  # IDs are stable across YAML mapping order.
        self.indices = {k: [self.fields.index(f) for f in layouts[k]] for k in self.names}

    def layout_id(self, name):
        if name not in self.layouts:
            raise ValueError(f"unregistered action_layout {name!r}")
        return self.names.index(name)

    def pack(self, action, layout, horizon, dim_mask=None, time_mask=None):
        """Native [H,d_i] -> canonical [H,D], element validity [H,D].

        dim_mask is [d_i] or [H,d_i]; masked values may be NaN. Missing
        action labels still produce a structural mask for inference generation.
        """
        self.layout_id(layout)
        indices = self.indices[layout]
        shape = (horizon, len(indices))
        x = torch.zeros(shape) if action is None else torch.as_tensor(action, dtype=torch.float32)
        if tuple(x.shape) != shape:
            raise ValueError(f"{layout}: expected native action shape {shape}, got {tuple(x.shape)}")
        valid = torch.ones(shape, dtype=torch.bool, device=x.device)
        if dim_mask is not None:
            supplied = torch.as_tensor(dim_mask, device=x.device)
            if supplied.dtype != torch.bool or tuple(supplied.shape) not in ((len(indices),), shape):
                raise ValueError("action_dim_mask must be bool [native_dim] or [H,native_dim]")
            valid &= supplied
        if time_mask is not None:
            tm = torch.as_tensor(time_mask, device=x.device)
            if tuple(tm.shape) != (horizon,) or not bool(((tm == 0) | (tm == 1)).all()):
                raise ValueError("action_mask must be binary [H]")
            valid &= tm.bool()[:, None]
        if not bool(valid.any()):
            raise ValueError("action layout has no valid elements")
        if not bool(torch.isfinite(x[valid]).all()) or bool((x[valid].abs() > 1.00001).any()):
            raise ValueError("valid native actions must be finite and normalized to [-1,1]")
        out = x.new_zeros(horizon, len(self.fields))
        mask = torch.zeros_like(out, dtype=torch.bool)
        out[:, indices] = torch.where(valid, x, torch.zeros_like(x))
        mask[:, indices] = valid
        return out, mask

    def unpack(self, canonical, layout):
        """Select native fields in the registered native order; no unit conversion."""
        self.layout_id(layout)
        if canonical.shape[-1] != len(self.fields):
            raise ValueError("canonical action width does not match schema")
        return canonical[..., self.indices[layout]]


def validate_action_mode(cfg):
    mode = getattr(cfg, "action_mode", "fixed")
    fields = getattr(cfg, "action_fields", [])
    layouts = getattr(cfg, "action_layouts", {})
    if mode not in ("fixed", "padded", "heterogeneous"):
        raise ValueError("action_mode must be fixed, padded or heterogeneous")
    if mode == "padded":
        native = getattr(cfg, "action_native_dim", None)
        if cfg.action_dim <= 0 or (native is not None and (type(native) is not int or not 0 < native <= cfg.action_dim)):
            raise ValueError("padded mode requires a positive model width and a valid optional action_native_dim")
    if mode == "heterogeneous":
        schema = ActionSchema(fields, layouts)
        if cfg.action_dim != len(schema.fields):
            raise ValueError("heterogeneous action_dim must equal len(action_fields)")
    elif fields or layouts:
        raise ValueError("action_fields/action_layouts require action_mode=heterogeneous")


def check_action_checkpoint(cfg, architecture):
    """Prevent silent semantic corruption even if tensor shapes are identical."""
    representation = getattr(cfg, "action_representation", "continuous")
    if representation != architecture.get("action_representation", "continuous"):
        raise ValueError("checkpoint action_representation mismatch")
    if representation == "x_tokenizer" and cfg.action_codec != architecture.get("action_codec"):
        raise ValueError("checkpoint action_codec mismatch (weights, normalization scale or anchor settings changed)")
    mode = getattr(cfg, "action_mode", "fixed")
    previous_mode = architecture.get("action_mode", "fixed")
    if mode != previous_mode:
        raise ValueError("checkpoint action_mode mismatch; explicit migration is required")
    if mode == "heterogeneous":
        for key in ("action_dim", "action_fields", "action_layouts"):
            if getattr(cfg, key) != architecture.get(key):
                raise ValueError(f"checkpoint {key} mismatch; explicit migration is required")


class HeterogeneousActionAdapter(ActionAdapter):
    def __init__(self, action_dim, width, mode, hidden_dim, fields, layouts):
        super().__init__(action_dim, width, mode, hidden_dim)
        self.schema = ActionSchema(fields, layouts)
        self.mask_projection = nn.Linear(action_dim, width, bias=False)
        self.layout_embedding = nn.Embedding(len(self.schema.names), width)
        structural = torch.zeros(len(self.schema.names), action_dim, dtype=torch.bool)
        for i, name in enumerate(self.schema.names):
            structural[i, self.schema.indices[name]] = True
        self.register_buffer("structural_mask", structural, persistent=False)

    def validate_mask(self, actions, dim_mask, layout_ids):
        if (dim_mask is None or dim_mask.dtype != torch.bool or dim_mask.shape != actions.shape
                or layout_ids is None or layout_ids.dtype != torch.long
                or layout_ids.shape != (actions.shape[0],)):
            raise ValueError("heterogeneous actions require bool [B,H,D] mask and long [B] layout IDs")
        if bool(((layout_ids < 0) | (layout_ids >= len(self.schema.names))).any()):
            raise ValueError("action layout ID out of range")
        allowed = self.structural_mask[layout_ids][:, None]
        if bool((dim_mask & ~allowed).any()):
            raise ValueError("action mask enables fields absent from its registered layout")

    def encode(self, actions, dim_mask, layout_ids):
        self.validate_mask(actions, dim_mask, layout_ids)
        x = torch.where(dim_mask, actions, torch.zeros_like(actions))
        return (super().encode(x) + self.mask_projection(dim_mask.to(x.dtype))
                + self.layout_embedding(layout_ids)[:, None])

    def decode(self, tokens, dim_mask):
        pred = super().decode(tokens)
        return torch.where(dim_mask, pred, torch.zeros_like(pred))
