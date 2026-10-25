"""V1 direct-action OpenPI-style width padding; no latent/semantic codec."""
import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import default_collate
from juice.action import pad_actions, unpad_actions
from juice.model import MMRiTConfig, MMRiT
from juice.train.vla_data import prepare_record
from juice.train.vla import vla_flow_loss
from juice.train.config import TrainConfig
from juice.vla_sampling import sample_vla


class PaddedActionTest(unittest.TestCase):
    def config(self):
        return SimpleNamespace(action_mode="padded", action_native_dim=7, action_dim=32, action_horizon=4,
            image_space="siglip2", text_canvas=8)

    def test_mixed_width_dataset(self):
        from tests.test_vla_data import record
        examples = []
        for width in (7, 14):
            r = record()
            r["action"] = torch.rand(4, width)
            parsed = prepare_record(r, self.config())
            torch.testing.assert_close(parsed["A"][:, :width], r["action"])
            self.assertTrue((parsed["A"][:, width:] == 0).all())
            self.assertNotIn("action_dim_mask", parsed)  # loss includes padding, as documented.
            examples.append(parsed)
        mixed = default_collate(examples)
        self.assertEqual(mixed["A"].shape, (2, 4, 32))
        native = unpad_actions(mixed["A"], mixed["action_native_dims"])
        self.assertEqual([x.shape[-1] for x in native], [7, 14])

    def test_missing_labels_need_native_metadata(self):
        from tests.test_vla_data import record
        r = record()
        del r["action"]
        parsed = prepare_record(r, self.config())
        self.assertFalse(parsed["present"]["A"])
        self.assertEqual(parsed["action_native_dims"], 7)
        c = self.config()
        c.action_native_dim = None
        with self.assertRaisesRegex(ValueError, "native width"):
            prepare_record(r, c)
        r["action_native_dim"] = 14
        self.assertEqual(prepare_record(r, c)["action_native_dims"], 14)

    def test_invalid_widths_and_shapes(self):
        for x in (torch.zeros(4, 33), torch.zeros(4, 0)):
            with self.assertRaises(ValueError):
                pad_actions(x, 32)
        from tests.test_vla_data import record
        for changes in ({"action": torch.zeros(4, 33)}, {"action_native_dim": 7},
                        {"action": torch.zeros(5, 3)}, {"action_native_dim": True}):
            with self.assertRaises(ValueError):
                prepare_record({**record(), **changes}, self.config())
        with self.assertRaises(ValueError):
            TrainConfig(run_name="bad", train_manifests=[], train_manifest_counts=[], action_mode="padded", action_dim=7, action_native_dim=14)

    def test_loss_matches_fixed_and_native_sampling(self):
        from tests.test_vla import batch, config
        fixed_cfg = MMRiTConfig(image_latent_dim=8, image_grid=2, text_dim=8, text_bottleneck_dim=4,
            hidden_size=16, depth=1, text_preamble_depth=1, num_heads=2, head_dim=8,
            time_tokens=1, action_dim=4)
        fixed = MMRiT(fixed_cfg)
        padded_cfg = MMRiTConfig(**{**fixed_cfg.to_dict(), "action_mode": "padded"})
        padded = MMRiT(padded_cfg)
        padded.load_state_dict(fixed.state_dict(), strict=True)
        data = batch()
        data["A"] = pad_actions(torch.rand(2, 5, 2), 4)
        data["action_native_dims"] = torch.tensor([2, 2])
        torch.manual_seed(3)
        a, _ = vla_flow_loss(fixed, data, config("action_policy"))
        torch.manual_seed(3)
        b, _ = vla_flow_loss(padded, data, config("action_policy"))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        b.backward()
        self.assertTrue(torch.isfinite(b))
        out = sample_vla(padded, data, steps=2)
        self.assertEqual(out["A"].shape, (2, 5, 4))
        self.assertEqual([x.shape for x in out["A_native"]], [torch.Size([5, 2]), torch.Size([5, 2])])


if __name__ == "__main__":
    unittest.main()
