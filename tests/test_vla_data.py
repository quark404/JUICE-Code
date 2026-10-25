import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from juice.action import ActionNormalizer
from juice.train.vla_data import prepare_record, make_vla_loader


def config():
    return SimpleNamespace(image_space="siglip2", text_canvas=8, action_dim=3, action_horizon=4,
        seed=1, micro_batch_size=1, num_workers=0, prefetch=2)


def record():
    return {"current_vision": torch.zeros(256, 1152), "instruction": torch.zeros(8, 512),
        "instruction_mask": torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]), "action": torch.zeros(4, 3)}


class VLADataTest(unittest.TestCase):
    def test_bounds_clip_constant_and_inverse(self):
        n = ActionNormalizer([-2, 0, 5], [2, 1, 5])
        x = torch.tensor([[1., .7, 5.]])
        z = n.normalize(x)
        torch.testing.assert_close(z, torch.tensor([[.5, .4, 0.]]))
        torch.testing.assert_close(n.denormalize(z), x)
        self.assertEqual(float(n.normalize(torch.tensor([[9., 0., 5.]]))[0, 0]), 1.)
        with self.assertRaises(ValueError):
            ActionNormalizer([2], [1])

    def test_record_validation(self):
        out = prepare_record(record(), config())
        self.assertTrue(out["present"]["A"])
        self.assertFalse(out["present"]["V"])
        self.assertFalse(out["goal_present"])
        for bad in (torch.zeros(5, 3), torch.full((4, 3), float("nan")), torch.full((4, 3), 2.)):
            with self.assertRaises(ValueError):
                prepare_record({**record(), "action": bad}, config())
        with self.assertRaises(KeyError):
            prepare_record({k: v for k, v in record().items() if k != "instruction_mask"}, config())

    def test_manifest_loader_roundtrip(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            torch.save(record(), path / "window.pt")
            (path / "train.txt").write_text("window.pt\n")
            cfg.train_manifests, cfg.train_manifest_counts = [str(path / "train.txt")], [1]
            batch = next(iter(make_vla_loader(cfg, 0, 1)))
            self.assertEqual(batch["A"].shape, (1, 4, 3))
            self.assertEqual(batch["current_vision"].shape, (1, 256, 1152))
            self.assertEqual(batch["instruction_mask"].sum(), 2)


if __name__ == "__main__":
    unittest.main()
