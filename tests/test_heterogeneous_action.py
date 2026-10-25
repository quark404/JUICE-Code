"""Mixed native widths, field semantics, masked flow and checkpoint compatibility."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import default_collate
from juice.action_schema import ActionSchema, check_action_checkpoint
from juice.model import MMRiT, MMRiTConfig
from juice.train.vla import TASKS, vla_flow_loss
from juice.train.vla_data import prepare_record, make_vla_loader
from juice.train.config import TrainConfig
from juice.vla_sampling import sample_vla

FIELDS = ["x", "gripper", "right_x", "right_gripper"]
LAYOUTS = {"single": ["gripper", "x"], "dual": FIELDS}


def cfg():
    return SimpleNamespace(action_mode="heterogeneous", action_dim=4, action_fields=FIELDS,
        action_layouts=LAYOUTS, action_horizon=3, image_space="siglip2", text_canvas=6,
        noise_scale=1., text_noise_scale=2., action_noise_scale=1., lambda_img=1., lambda_txt=.25,
        lambda_action=1., vla_tasks={"action_policy": 1.}, num_workers=0, prefetch=2,
        seed=1, micro_batch_size=2)


def tiny(mode="continuous"):
    return MMRiT(MMRiTConfig(image_latent_dim=8, image_grid=2, text_dim=8, text_bottleneck_dim=4,
        hidden_size=16, depth=1, text_preamble_depth=1, num_heads=2, head_dim=8, time_tokens=1,
        action_dim=4, action_mode="heterogeneous", action_fields=FIELDS, action_layouts=LAYOUTS,
        action_adapter=mode))


def enc():
    from tests.test_vla import batch
    out = batch()
    out["A"] = torch.rand(2, 3, 4)
    out["valid"]["A"] = torch.ones(2, 3, dtype=torch.bool)
    out["action_dim_mask"] = torch.ones(2, 3, 4, dtype=torch.bool)
    out["action_dim_mask"][0, :, 2:] = False
    out["action_layout_ids"] = torch.tensor([1, 0])  # IDs use sorted names: dual, single.
    return out


def raw(layout):
    return {"action_layout": layout, "action": torch.zeros(3, len(LAYOUTS[layout])),
        "current_vision": torch.zeros(256, 1152), "instruction": torch.zeros(6, 512),
        "instruction_mask": torch.ones(6, dtype=torch.bool)}


class HeterogeneousActionTest(unittest.TestCase):
    def test_semantic_pack_and_unpack(self):
        schema = ActionSchema(FIELDS, LAYOUTS)
        x = torch.tensor([[.2, .8], [.1, .9], [.3, .7]])
        packed, mask = schema.pack(x, "single", 3)
        torch.testing.assert_close(packed[:, 0], x[:, 1])
        torch.testing.assert_close(packed[:, 1], x[:, 0])
        self.assertFalse(mask[:, 2:].any())
        torch.testing.assert_close(schema.unpack(packed, "single"), x)
        self.assertEqual(schema.layout_id("single"), ActionSchema(FIELDS, dict(reversed(list(LAYOUTS.items())))).layout_id("single"))

    def test_mixed_dataset_and_absent_labels(self):
        first, second = prepare_record(raw("single"), cfg()), prepare_record(raw("dual"), cfg())
        out = default_collate([first, second])
        self.assertEqual(out["A"].shape, (2, 3, 4))
        self.assertEqual(out["action_dim_mask"].sum((1, 2)).tolist(), [6, 12])
        missing = raw("single")
        del missing["action"]
        out = prepare_record(missing, cfg())
        self.assertFalse(out["present"]["A"])
        self.assertFalse(out["valid"]["A"].any())
        self.assertTrue(out["action_dim_mask"].any())  # structural mask survives for generation.

    def test_masked_nan_and_partial_time(self):
        data = raw("single")
        data["action"][1, 0] = float("nan")
        data["action_dim_mask"] = torch.ones(3, 2, dtype=torch.bool)
        data["action_dim_mask"][1, 0] = False
        data["action_mask"] = torch.tensor([1, 1, 0])
        out = prepare_record(data, cfg())
        self.assertTrue(torch.isfinite(out["A"]).all())
        self.assertEqual(int(out["action_dim_mask"].sum()), 3)
        self.assertFalse(out["valid"]["A"][2])

    def test_all_tasks_backward_and_optimizer(self):
        for mode in ("continuous", "mlp"):
            model = tiny(mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for task in TASKS:
                with self.subTest(mode=mode, task=task):
                    c = cfg()
                    c.vla_tasks = {task: 1.}
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = vla_flow_loss(model, enc(), c)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    optimizer.step()

    def test_invalid_values_do_not_change_loss_or_sampling(self):
        data, model = enc(), tiny()
        torch.manual_seed(8)
        loss, metrics = vla_flow_loss(model, data, cfg())
        before = sample_vla(model, data, steps=3, generator=torch.Generator().manual_seed(9))
        data["A"][~data["action_dim_mask"]] = float("nan")
        torch.manual_seed(8)
        same, _ = vla_flow_loss(model, data, cfg())
        after = sample_vla(model, data, steps=3, generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(loss, same, rtol=0, atol=0)
        torch.testing.assert_close(before["A"], after["A"], rtol=0, atol=0)
        self.assertEqual(int(metrics["active_A_elements"]), 18)
        self.assertTrue((after["A"][~data["action_dim_mask"]] == 0).all())

    def test_world_model_condition_is_masked(self):
        model, data, c = tiny(), enc(), cfg()
        c.vla_tasks = {"world_model": 1.}
        torch.manual_seed(10)
        loss, _ = vla_flow_loss(model, data, c)
        data["A"][~data["action_dim_mask"]] = 1e8
        torch.manual_seed(10)
        same, _ = vla_flow_loss(model, data, c)
        torch.testing.assert_close(loss, same, rtol=0, atol=0)
        out = sample_vla(model, data, task="world_model", steps=2)
        torch.testing.assert_close(out["A"][data["action_dim_mask"]], data["A"][data["action_dim_mask"]])
        self.assertTrue((out["A"][~data["action_dim_mask"]] == 0).all())

    def test_equal_per_sample_loss_not_dimension_weighted(self):
        data, c = enc(), cfg()
        class ExactErrors:
            def __call__(self, v, tv, l, tl, action, t_action, **kwargs):
                error = torch.tensor([1., 2.])[:, None, None]
                return torch.zeros_like(v), torch.zeros_like(l), data["A"] + (1-t_action[..., None]) * error
        loss, _ = vla_flow_loss(ExactErrors(), data, c)
        # Each sample has mean squared velocity error 1 or 4, regardless of native width.
        torch.testing.assert_close(loss, torch.tensor(2.5))

    def test_invalid_layouts_and_masks_rejected(self):
        for fields, layouts in ((["x", "x"], LAYOUTS), (FIELDS, {"bad": ["unknown"]})):
            with self.assertRaises(ValueError):
                ActionSchema(fields, layouts)
        for changes in ({"action_layout": "unknown"}, {"action": torch.zeros(3, 3)},
                        {"action_dim_mask": torch.ones(2)}, {"action": torch.full((3, 2), 2.)}):
            with self.assertRaises(ValueError):
                prepare_record({**raw("single"), **changes}, cfg())
        data, model = enc(), tiny()
        data["action_dim_mask"][0, :, 2] = True
        with self.assertRaisesRegex(ValueError, "absent"):
            vla_flow_loss(model, data, cfg())

    def test_checkpoint_and_fixed_compatibility(self):
        model = tiny()
        architecture = model.cfg.to_dict()
        check_action_checkpoint(model.cfg, architecture)
        altered = copy.deepcopy(architecture)
        altered["action_fields"] = list(reversed(FIELDS))
        with self.assertRaisesRegex(ValueError, "action_fields"):
            check_action_checkpoint(model.cfg, altered)
        with self.assertRaisesRegex(ValueError, "action_mode"):
            check_action_checkpoint(model.cfg, {})
        restored = MMRiT(MMRiTConfig(**architecture))
        restored.load_state_dict(model.state_dict(), strict=True)
        # Old fixed checkpoints omit the new fields and remain accepted.
        fixed = MMRiTConfig(action_dim=7)
        check_action_checkpoint(fixed, {"action_dim": 7})
        for kwargs in ({"action_mode": "invalid"}, {"action_mode": "heterogeneous", "action_dim": 7},
                       {"action_fields": ["x"]}):
            with self.assertRaises(ValueError):
                TrainConfig(run_name="bad", train_manifests=[], train_manifest_counts=[], **kwargs)

    def test_explicit_backbone_migration(self):
        from juice.train.trainer import load_backbone_weights
        source_cfg = tiny().cfg.to_dict()
        source_cfg.update(action_mode="fixed", action_fields=[], action_layouts={}, action_dim=2)
        source = MMRiT(MMRiTConfig(**source_cfg))
        target = tiny()
        old_action = target.action.encoder.weight.detach().clone()
        load_backbone_weights(target, source.state_dict())
        torch.testing.assert_close(target.latent_in.weight, source.latent_in.weight, rtol=0, atol=0)
        torch.testing.assert_close(target.action.encoder.weight, old_action, rtol=0, atol=0)
        self.assertFalse(any("mask_projection" in k or "layout_embedding" in k for k in source.state_dict()))
        incompatible = dict(source.state_dict())
        del incompatible["latent_in.weight"]
        with self.assertRaisesRegex(ValueError, "incompatible backbone"):
            load_backbone_weights(tiny(), incompatible)

    def test_mixed_manifest_loader(self):
        c = cfg()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in LAYOUTS:
                torch.save(raw(name), p / f"{name}.pt")
            (p / "train.txt").write_text("single.pt\ndual.pt\n")
            c.train_manifests, c.train_manifest_counts = [str(p / "train.txt")], [2]
            out = next(iter(make_vla_loader(c, 0, 1)))
            self.assertEqual(out["A"].shape, (2, 3, 4))
            self.assertEqual(out["action_dim_mask"].shape, out["A"].shape)


if __name__ == "__main__":
    unittest.main()
