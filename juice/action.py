"""Action states stay in normalized physical coordinates, including the flow loss.

Adapters only project at the backbone boundary; the MLP cannot collapse a learned
latent target. Dataset-specific normalization must be applied before packing.
"""
from torch import nn


class ActionAdapter(nn.Module):
    def __init__(self, action_dim: int, width: int, mode: str = "continuous", hidden_dim: int = 256):
        super().__init__()
        if min(action_dim, width, hidden_dim) <= 0:
            raise ValueError("action dimensions must be positive")
        if mode == "continuous":
            self.encoder = nn.Linear(action_dim, width)
            self.decoder = nn.Linear(width, action_dim)
        elif mode == "mlp":
            self.encoder = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, width))
            self.decoder = nn.Sequential(nn.Linear(width, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim))
        else:
            raise ValueError(f"unknown action adapter {mode!r}")
        self.action_dim = action_dim

    def encode(self, actions):
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"actions must have shape [B, H, {self.action_dim}]")
        return self.encoder(actions)

    def decode(self, tokens):
        return self.decoder(tokens)


class ActionNormalizer:
    """Dynin Eq. (3) robust scaling, without the discrete binning step.

    Fit bounds on training data only (typically motion-channel p01/p99). Keep the
    gripper absolute; pass its calibrated bounds without accumulating deltas.
    Constant channels normalize to zero. Clipping makes inversion lossy outside bounds.
    """
    def __init__(self, lower, upper, eps=1e-6):
        import torch
        self.lower = torch.as_tensor(lower, dtype=torch.float32)
        self.upper = torch.as_tensor(upper, dtype=torch.float32)
        if (self.lower.ndim != 1 or self.lower.numel() == 0 or self.lower.shape != self.upper.shape
                or not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all()
                or (self.upper < self.lower).any() or eps <= 0):
            raise ValueError("invalid per-channel action bounds")
        self.eps = eps

    def _bounds(self, x):
        if x.shape[-1] != self.lower.numel():
            raise ValueError("action dimension does not match normalization bounds")
        return self.lower.to(x), self.upper.to(x)

    def normalize(self, x):
        import torch
        lower, upper = self._bounds(x)
        z = (2 * (x - lower) / (upper - lower).clamp_min(self.eps) - 1).clamp(-1, 1)
        return torch.where(upper - lower > self.eps, z, torch.zeros_like(z))

    def denormalize(self, z):
        lower, upper = self._bounds(z)
        return lower + (z.clamp(-1, 1) + 1) * .5 * (upper - lower)
