"""Frozen-feature training records for the no-State Action baseline.

Manifests contain paths to torch.save dictionaries, one window per file. Only
weights_only=True deserialization is used. Encoders run independently per variable
group upstream; never encode instruction + target plan as a single T5 sequence.
"""
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, IterableDataset
from juice.representations import IMAGE_SPACES
from juice.train.data import read_manifest, balanced_slot_assignment


def prepare_record(record, cfg):
    dims = {"V": (256, IMAGE_SPACES[cfg.image_space].latent_dim),
            "L": (cfg.text_canvas, 512), "A": (cfg.action_horizon, cfg.action_dim)}
    out = {}
    heterogeneous = getattr(cfg, "action_mode", "fixed") == "heterogeneous"
    padded = getattr(cfg, "action_mode", "fixed") == "padded"
    padded_action = None
    if padded:
        from juice.action import pad_actions
        raw_action = record.get("action")
        raw_action = None if raw_action is None else torch.as_tensor(raw_action, dtype=torch.float32)
        if raw_action is not None and (raw_action.ndim != 2 or raw_action.shape[0] != cfg.action_horizon):
            raise ValueError("padded action must have native shape [H,d_i]")
        native = record.get("action_native_dim", raw_action.shape[-1] if raw_action is not None
                            else getattr(cfg, "action_native_dim", None))
        if type(native) is not int or not 0 < native <= cfg.action_dim:
            raise ValueError("padded records need a native width within model action_dim")
        if raw_action is not None and raw_action.shape[-1] != native:
            raise ValueError("action_native_dim does not match native action tensor")
        padded_action = torch.zeros(dims["A"]) if raw_action is None else pad_actions(raw_action, cfg.action_dim)
        out["action_native_dims"] = torch.tensor(native, dtype=torch.long)
    packed_action = element_mask = None
    if heterogeneous:
        from juice.action_schema import ActionSchema
        schema = ActionSchema(cfg.action_fields, cfg.action_layouts)
        layout = record.get("action_layout")
        packed_action, element_mask = schema.pack(record.get("action"), layout, cfg.action_horizon,
            record.get("action_dim_mask"), record.get("action_mask"))
        out["action_layout_ids"] = torch.tensor(schema.layout_id(layout), dtype=torch.long)
        out["action_dim_mask"] = element_mask
    elif "action_dim_mask" in record or "action_layout" in record:
        raise ValueError("native action layout metadata requires action_mode=heterogeneous")
    for key, shape in {"current_vision": dims["V"], "instruction": dims["L"]}.items():
        value = torch.as_tensor(record[key], dtype=torch.float32)
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"{key}: expected finite {shape}")
        out[key] = value
        mask_value = record["instruction_mask"] if key == "instruction" else record.get(f"{key}_mask", torch.ones(shape[0]))
        out[f"{key}_mask"] = torch.as_tensor(mask_value, dtype=torch.bool)
        if out[f"{key}_mask"].shape != (shape[0],) or not out[f"{key}_mask"].any():
            raise ValueError(f"invalid {key}_mask")
    out["present"], out["valid"] = {}, {}
    for m, key in zip("VLA", ("future_vision", "language_plan", "action")):
        present = key in record
        if key == "language_plan" and present and "language_plan_mask" not in record:
            raise ValueError("language_plan requires language_plan_mask")
        x = torch.as_tensor(record[key], dtype=torch.float32) if present else torch.zeros(dims[m])
        if m == "A" and heterogeneous:
            x = packed_action
        elif m == "A" and padded:
            x = padded_action
        mask = torch.as_tensor(record.get(f"{key}_mask", torch.ones(dims[m][0]) if present else torch.zeros(dims[m][0])), dtype=torch.bool)
        if m == "A" and heterogeneous:
            mask = element_mask.any(-1) & present
        if tuple(x.shape) != dims[m] or not torch.isfinite(x).all() or mask.shape != dims[m][:1]:
            raise ValueError(f"{key}: invalid shape, mask, or nonfinite values")
        if present and not mask.any():
            raise ValueError(f"{key}: present group has no valid tokens")
        out[m] = x
        out["valid"][m] = mask
        out["present"][m] = torch.tensor(present)
    if getattr(cfg, "action_representation", "continuous") == "x_tokenizer":
        dm = torch.as_tensor(record.get("codec_dof_mask", torch.ones(dims["A"], dtype=torch.bool)))
        if dm.dtype != torch.bool or dm.shape != dims["A"] or not dm.any():
            raise ValueError("codec_dof_mask must be bool [H,native_dim]")
        out["codec_dof_mask"] = dm
        rid = record.get("codec_robot_type_id", cfg.x_tokenizer_robot_type_id)
        if not isinstance(rid, int) or rid < 0:
            raise ValueError("codec_robot_type_id must be a nonnegative integer")
        out["codec_robot_type_ids"] = torch.tensor(rid, dtype=torch.long)
        if cfg.x_tokenizer_use_anchor:
            obs = torch.as_tensor(record["codec_obs_state"], dtype=torch.float32)
            if obs.shape != (1, cfg.action_dim) or not torch.isfinite(obs).all():
                raise ValueError("codec_obs_state must be normalized finite [1,native_dim]")
            out["codec_obs_state"] = obs
        elif "codec_obs_state" in record:
            raise ValueError("codec_obs_state requires x_tokenizer_use_anchor=true")
    goal = record.get("goal_vision")
    out["goal_present"] = torch.tensor(goal is not None)
    out["goal_vision"] = torch.zeros(dims["V"]) if goal is None else torch.as_tensor(goal, dtype=torch.float32)
    out["goal_valid"] = torch.full(dims["V"][:1], goal is not None, dtype=torch.bool)
    if out["goal_vision"].shape != dims["V"] or not torch.isfinite(out["goal_vision"]).all():
        raise ValueError("invalid goal_vision")
    if out["present"]["A"] and out["A"].abs().max() > 1.00001:
        raise ValueError("action must be normalized to [-1,1] using training-split bounds")
    return out


class FeatureStream(IterableDataset):
    def __init__(self, cfg, rank, world, resume_key):
        self.cfg, self.rank, self.world = cfg, rank, world
        self.seed = cfg.seed + resume_key * 7368787
        self.paths, self.weights = [], []
        for manifest, count in zip(cfg.train_manifests, cfg.train_manifest_counts):
            entries = read_manifest(manifest)
            if not entries or count <= 0:
                raise ValueError(f"empty manifest or invalid count: {manifest}")
            self.paths.extend(entries)
            self.weights.extend([count / len(entries)] * len(entries))
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("duplicate feature records in manifests")
        for p in self.paths:
            if not Path(p).is_file():
                raise FileNotFoundError(p)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        slot = self.rank * workers + worker
        indices = balanced_slot_assignment(list(range(len(self.paths))), self.weights, self.world * workers)[slot]
        if not indices:
            raise ValueError("fewer feature records than workers; reduce num_workers")
        rng = random.Random(self.seed + slot)
        while True:
            i = rng.choices(indices, weights=[self.weights[j] for j in indices])[0]
            yield prepare_record(torch.load(self.paths[i], map_location="cpu", weights_only=True), self.cfg)


def make_vla_loader(cfg, rank, world, resume_key=0):
    return DataLoader(FeatureStream(cfg, rank, world, resume_key), batch_size=cfg.micro_batch_size,
        num_workers=cfg.num_workers, drop_last=True,
        prefetch_factor=cfg.prefetch if cfg.num_workers else None)
