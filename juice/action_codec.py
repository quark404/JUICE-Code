"""Optional frozen X-Tokenizer latent interface, independent of JUICE parameters.

Uses Q(E(A)) targets and D(z_q) decoding, not the untrained D(E(A)) bypass.
Input is already normalized in the checkpoint's delta-action convention.
"""
import hashlib
import math
from pathlib import Path
import torch


class FrozenXTokenizer:
    def __init__(self, backend, architecture, horizon, *, scale=1., use_anchor=False,
                 fingerprint="injected-test-backend"):
        if horizon <= 0 or horizon > int(architecture["max_chunk_length"]) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid codec horizon or latent scale")
        self.backend = backend.eval().requires_grad_(False)
        self.raw_dim = int(architecture["action_dim"])
        self.latent_dim = int(architecture["hidden_dim"])
        self.latent_horizon = math.ceil(horizon / int(architecture["compression_ratio"]))
        self.horizon, self.scale, self.use_anchor = horizon, scale, use_anchor
        self.max_robot_types = int(architecture["max_robot_types"])
        self.metadata = {"type": "x_tokenizer", "space": "quantized_features", "sha256": fingerprint,
            "raw_dim": self.raw_dim, "latent_dim": self.latent_dim, "horizon": horizon,
            "latent_horizon": self.latent_horizon, "scale": scale, "use_anchor": use_anchor,
            "max_robot_types": self.max_robot_types}

    @classmethod
    def from_config(cls, cfg, device):
        path = Path(cfg.x_tokenizer_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"X-Tokenizer checkpoint not found: {path}")
        # Keep restricted deserialization; never fall back to weights_only=False.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        try:
            from xtokenizer.model.tokenizer import InferenceTokenizerModel
        except ImportError as exc:
            raise ImportError("Frozen X-Tokenizer requires its optional official package; continuous Action modes do not") from exc
        architecture = payload["architecture"]
        backend = InferenceTokenizerModel(architecture)
        backend.load_state_dict(payload["model_state_dict"], strict=True)
        backend.to(device)
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        codec = cls(backend, architecture, cfg.action_horizon, scale=cfg.x_tokenizer_latent_scale,
                    use_anchor=cfg.x_tokenizer_use_anchor, fingerprint=digest.hexdigest())
        if codec.raw_dim != cfg.action_dim:
            raise ValueError("action_dim must match the X-Tokenizer checkpoint's native input layout")
        return codec

    def _context(self, batch):
        x = batch["A"]
        if x.shape[1:] != (self.horizon, self.raw_dim):
            raise ValueError("codec expects raw [B,H,D] actions matching the checkpoint")
        mask, ids = batch["codec_dof_mask"], batch["codec_robot_type_ids"]
        if mask.dtype != torch.bool or mask.shape != x.shape or not bool(mask.flatten(1).any(1).all()):
            raise ValueError("codec_dof_mask must be bool [B,H,D] with valid elements per sample")
        if ids.dtype != torch.long or ids.shape != (x.shape[0],) or bool(((ids < 0) | (ids >= self.max_robot_types)).any()):
            raise ValueError("invalid codec robot type IDs")
        obs = batch.get("codec_obs_state")
        if self.use_anchor:
            if obs is None or obs.shape != (x.shape[0], 1, self.raw_dim) or not bool(torch.isfinite(obs).all()):
                raise ValueError("anchored codec requires finite codec_obs_state [B,1,D]")
        elif obs is not None:
            raise ValueError("codec anchor supplied but x_tokenizer_use_anchor=false")
        self.backend.eval()  # prevents dropout/codebook EMA updates even if caller changed mode
        return {"dof_mask": mask, "robot_type_id": ids, "obs_state": obs}

    @torch.no_grad()
    def encode_batch(self, batch, *, encode_actions=True):
        context = self._context(batch)
        x = batch["A"]
        if encode_actions:
            valid = batch["valid"]["A"]
            if bool((batch["present"]["A"] & ~valid.all(1)).any()):
                raise ValueError("frozen codec currently requires complete action chunks")
            x = torch.where(context["dof_mask"], x, torch.zeros_like(x))
            if not bool(torch.isfinite(x).all()) or bool((x.abs() > 1.00001).any()):
                raise ValueError("valid codec actions must be finite normalized deltas in [-1,1]")
            # Official encoder does not apply dof_mask itself: zero invalid fields above.
            result = self.backend.encode(x, obs_state=context["obs_state"], robot_type_id=context["robot_type_id"])
            z = result["quantized_features"].detach().float() / self.scale
        else:
            # Sampling a target Action must never encode the ground-truth action.
            z = x.new_zeros(x.shape[0], self.latent_horizon, self.latent_dim)
        if z.shape != (x.shape[0], self.latent_horizon, self.latent_dim) or not bool(torch.isfinite(z).all()):
            raise ValueError("unexpected/nonfinite quantized feature shape")
        out = dict(batch)
        out["A"] = z
        out["action_codec_mask"] = context["dof_mask"].any(1)
        out["action_codec_ids"] = context["robot_type_id"]
        out["valid"] = {**batch["valid"], "A": torch.ones(z.shape[:2], device=z.device, dtype=torch.bool)}
        return out

    @torch.no_grad()
    def decode(self, latents, raw_batch):
        context = self._context(raw_batch)
        expected = (raw_batch["A"].shape[0], self.latent_horizon, self.latent_dim)
        if latents.shape != expected or not bool(torch.isfinite(latents).all()):
            raise ValueError("invalid predicted codec latents")
        # D is pretrained on code-vector sums. No continuous_features bypass or extra RVQ pass.
        actions = self.backend.decode(latents.float() * self.scale, target_length=self.horizon, **context)
        if actions.shape != raw_batch["A"].shape or not bool(torch.isfinite(actions).all()):
            raise ValueError("frozen decoder returned invalid/nonfinite actions")
        return torch.where(context["dof_mask"], actions, torch.zeros_like(actions))

    @torch.no_grad()
    def reconstruction_metrics(self, raw_batch):
        """Run E->Q->D before accepting a checkpoint/data convention for Flow training."""
        encoded = self.encode_batch(raw_batch)
        reconstruction = self.decode(encoded["A"], raw_batch)
        mask = raw_batch["codec_dof_mask"] & raw_batch["present"]["A"][:, None, None]
        if not bool(mask.any()):
            raise ValueError("reconstruction validation requires labeled actions")
        error = (reconstruction - raw_batch["A"])[mask]
        return {"valid_elements": int(mask.sum()), "mae": float(error.abs().mean()),
                "rmse": float(error.square().mean().sqrt()), "max_abs": float(error.abs().max())}
