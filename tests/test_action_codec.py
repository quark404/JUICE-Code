"""Frozen-codec integration contract tests with a synthetic backend.

These do not assert reconstruction quality of the unavailable pretrained checkpoint.
"""
import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from juice.action_codec import FrozenXTokenizer
from juice.action_schema import check_action_checkpoint
from juice.model import MMRiT, MMRiTConfig
from juice.train.vla import vla_flow_loss
from juice.vla_sampling import sample_vla


ARCH = {"action_dim": 4, "hidden_dim": 3, "compression_ratio": 2,
        "max_chunk_length": 64, "max_robot_types": 18}


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 3)
        self.decoder = nn.Linear(3, 4)
        self.encode_calls = 0

    def encode(self, x, **kwargs):
        self.encode_calls += 1
        z = self.encoder(x.reshape(x.shape[0], 2, 2, 4).mean(2))
        return {"quantized_features": z.round(), "continuous_features": z + 10}

    def decode(self, z, target_length, **kwargs):
        return self.decoder(z).repeat_interleave(2, 1)[:, :target_length]


def codec(use_anchor=False):
    return FrozenXTokenizer(Backend(), ARCH, 4, scale=2., use_anchor=use_anchor)


def data():
    from tests.test_vla import batch
    out = batch()
    out["A"] = torch.rand(2, 4, 4) * 2 - 1
    out["valid"]["A"] = torch.ones(2, 4, dtype=torch.bool)
    out["codec_dof_mask"] = torch.ones(2, 4, 4, dtype=torch.bool)
    out["codec_dof_mask"][0, :, 2:] = False
    out["codec_robot_type_ids"] = torch.tensor([0, 2])
    return out


def model(c):
    return MMRiT(MMRiTConfig(image_latent_dim=8, image_grid=2, text_dim=8, text_bottleneck_dim=4,
        hidden_size=16, depth=1, text_preamble_depth=1, num_heads=2, head_dim=8, time_tokens=1,
        action_dim=c.latent_dim, action_representation="x_tokenizer", action_codec=c.metadata))


class CodecTest(unittest.TestCase):
    def test_frozen_quantized_targets_and_decode(self):
        c, batch = codec(), data()
        c.backend.train()  # wrapper restores eval; no gradients/EMA training allowed.
        encoded = c.encode_batch(batch)
        self.assertFalse(c.backend.training)
        self.assertFalse(encoded["A"].requires_grad)
        self.assertFalse(any(p.requires_grad for p in c.backend.parameters()))
        # Quantized target must not accidentally be continuous_features.
        torch.testing.assert_close(encoded["A"] * c.scale, (encoded["A"] * c.scale).round())
        self.assertEqual(encoded["A"].shape, (2, 2, 3))
        recon = c.decode(encoded["A"], batch)
        self.assertEqual(recon.shape, (2, 4, 4))
        self.assertTrue((recon[~batch["codec_dof_mask"]] == 0).all())
        self.assertEqual(c.reconstruction_metrics(batch)["valid_elements"], 24)

    def test_flow_backward_does_not_update_codec(self):
        from tests.test_vla import config
        c, batch = codec(), data()
        before = copy.deepcopy(c.backend.state_dict())
        m = model(c)
        optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
        loss, _ = vla_flow_loss(m, c.encode_batch(batch), config("joint_lva"))
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in c.backend.parameters()))
        for key, value in c.backend.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        main_parameters = {id(p) for p in m.parameters()}
        self.assertTrue(all(id(p) not in main_parameters for p in c.backend.parameters()))

    def test_sampling_never_encodes_target_action(self):
        c, batch = codec(), data()
        m = model(c)
        first = sample_vla(m, batch, steps=2, action_codec=c, generator=torch.Generator().manual_seed(3))
        batch["A"] += 1000
        second = sample_vla(m, batch, steps=2, action_codec=c, generator=torch.Generator().manual_seed(3))
        self.assertEqual(c.backend.encode_calls, 0)
        torch.testing.assert_close(first["A"], second["A"], rtol=0, atol=0)
        self.assertEqual(first["A_latent"].shape, (2, 2, 3))
        self.assertEqual(first["A"].shape, (2, 4, 4))
        with self.assertRaisesRegex(ValueError, "matching frozen"):
            sample_vla(m, batch)

    def test_world_model_encodes_known_actions(self):
        c, batch = codec(), data()
        out = sample_vla(model(c), batch, task="world_model", steps=2, action_codec=c)
        self.assertEqual(c.backend.encode_calls, 1)
        torch.testing.assert_close(out["A"][batch["codec_dof_mask"]], batch["A"][batch["codec_dof_mask"]])

    def test_anchor_opt_in_and_validation(self):
        c, batch = codec(True), data()
        with self.assertRaisesRegex(ValueError, "anchor"):
            c.encode_batch(batch)
        batch["codec_obs_state"] = torch.zeros(2, 1, 4)
        self.assertEqual(c.encode_batch(batch)["A"].shape, (2, 2, 3))
        with self.assertRaisesRegex(ValueError, "anchor supplied"):
            codec(False).encode_batch(batch)
        del batch["codec_obs_state"]
        batch["valid"]["A"][0, 0] = False
        with self.assertRaisesRegex(ValueError, "complete"):
            codec(False).encode_batch(batch)

    def test_feature_loader_codec_metadata(self):
        from juice.train.config import TrainConfig
        from juice.train.vla_data import prepare_record
        from tests.test_vla_data import record
        c = TrainConfig(run_name="codec", train_manifests=[], train_manifest_counts=[],
            action_dim=4, action_horizon=4, text_canvas=8, action_representation="x_tokenizer",
            x_tokenizer_checkpoint="local.pth", x_tokenizer_robot_type_id=2)
        raw = record()
        raw["action"] = torch.zeros(4, 4)
        parsed = prepare_record(raw, c)
        self.assertEqual(parsed["codec_robot_type_ids"], 2)
        self.assertEqual(parsed["codec_dof_mask"].shape, (4, 4))
        raw["codec_obs_state"] = torch.zeros(1, 4)
        with self.assertRaisesRegex(ValueError, "requires"):
            prepare_record(raw, c)
        c.x_tokenizer_use_anchor = True
        parsed = prepare_record(raw, c)
        self.assertEqual(parsed["codec_obs_state"].shape, (1, 4))

    def test_safe_checkpoint_loader_with_injected_official_class(self):
        import hashlib
        import tempfile
        import types
        from pathlib import Path
        from unittest.mock import patch
        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "codec.pth"
            torch.save({"architecture": ARCH, "model_state_dict": backend.state_dict()}, path)
            c = SimpleNamespace(x_tokenizer_checkpoint=str(path), action_horizon=4, action_dim=4,
                x_tokenizer_latent_scale=1., x_tokenizer_use_anchor=False)
            module = types.ModuleType("xtokenizer.model.tokenizer")
            module.InferenceTokenizerModel = lambda architecture: Backend()
            # No install/download: test only our loader contract, not official numerical behavior.
            with patch.dict("sys.modules", {"xtokenizer.model.tokenizer": module}):
                loaded = FrozenXTokenizer.from_config(c, "cpu")
            self.assertEqual(loaded.metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(any(p.requires_grad for p in loaded.backend.parameters()))
            for name, value in backend.state_dict().items():
                torch.testing.assert_close(value, loaded.backend.state_dict()[name])

    def test_checkpoint_metadata_and_config_guards(self):
        from juice.train.config import TrainConfig
        c = codec()
        m = model(c)
        check_action_checkpoint(m.cfg, m.cfg.to_dict())
        changed = m.cfg.to_dict()
        changed["action_codec"] = {**changed["action_codec"], "sha256": "different"}
        with self.assertRaisesRegex(ValueError, "action_codec"):
            check_action_checkpoint(m.cfg, changed)
        for args in ({"action_representation": "bad"},
                     {"action_representation": "x_tokenizer", "action_dim": 4},
                     {"action_representation": "x_tokenizer", "action_dim": 4,
                      "x_tokenizer_checkpoint": "codec.pth", "x_tokenizer_latent_scale": float("nan")}):
            with self.assertRaises(ValueError):
                TrainConfig(run_name="bad", train_manifests=[], train_manifest_counts=[], **args)


if __name__ == "__main__":
    unittest.main()
