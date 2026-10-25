"""Small CPU regression tests, without pretrained weights or model downloads."""
import unittest
from types import SimpleNamespace
import torch
from juice.model import MMRiT, MMRiTConfig
from juice.train.vla import TASKS, validate_tasks, vla_flow_loss
from juice.vla_sampling import sample_vla


def model(mode="continuous", enabled=True):
    return MMRiT(MMRiTConfig(image_latent_dim=8, image_grid=2, text_dim=8,
        text_bottleneck_dim=4, hidden_size=16, depth=1, text_preamble_depth=1,
        num_heads=2, head_dim=8, time_tokens=1, action_dim=3 if enabled else 0, action_adapter=mode))


def batch():
    x = {"V": torch.randn(2, 4, 8), "L": torch.randn(2, 6, 8), "A": torch.randn(2, 5, 3),
         "current_vision": torch.randn(2, 4, 8), "instruction": torch.randn(2, 6, 8)}
    x["valid"] = {m: torch.ones(v.shape[:2], dtype=torch.bool) for m, v in x.items() if m in "VLA"}
    x["present"] = {m: torch.ones(2, dtype=torch.bool) for m in "VLA"}
    x["instruction_mask"] = torch.ones(2, 6, dtype=torch.bool)
    x["current_vision_mask"] = torch.ones(2, 4, dtype=torch.bool)
    x["goal_vision"] = torch.randn(2, 4, 8)
    x["goal_present"] = torch.ones(2, dtype=torch.bool)
    x["goal_valid"] = torch.ones(2, 4, dtype=torch.bool)
    return x


def config(task):
    return SimpleNamespace(vla_tasks={task: 1.}, noise_scale=1., text_noise_scale=2.,
                           action_noise_scale=1., lambda_img=1., lambda_txt=.25, lambda_action=1.)


class VLATest(unittest.TestCase):
    def test_every_task_backward_both_adapters(self):
        for mode in ("continuous", "mlp"):
            m = model(mode)
            for task in TASKS:
                with self.subTest(mode=mode, task=task):
                    m.zero_grad(set_to_none=True)
                    loss, metrics = vla_flow_loss(m, batch(), config(task))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                    for key in "VLA":
                        self.assertEqual(metrics[f"active_{key}"] > 0, key in TASKS[task][1])

    def test_action_policy_cannot_see_future_or_plan(self):
        m, enc = model(), batch()
        torch.manual_seed(42)
        first = vla_flow_loss(m, enc, config("action_policy"))[0]
        enc["V"] += 1000
        enc["L"] -= 1000
        enc["goal_vision"] += 1000
        torch.manual_seed(42)
        second = vla_flow_loss(m, enc, config("action_policy"))[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_legacy_no_action_keys_and_shapes(self):
        m, enc = model(enabled=False), batch()
        self.assertFalse(any("act_" in k or k.startswith("action.") for k in m.state_dict()))
        outputs = m(enc["V"], torch.rand(2), enc["L"], torch.rand(2))
        self.assertEqual(len(outputs), 2)

    def test_missing_targets_fail(self):
        enc = batch()
        enc["present"]["A"].fill_(False)
        with self.assertRaisesRegex(ValueError, "eligible"):
            vla_flow_loss(model(), enc, config("action_policy"))

    def test_sampler_hides_ground_truth_action(self):
        m, enc = model(), batch()
        first = sample_vla(m, enc, steps=3, generator=torch.Generator().manual_seed(2))
        enc["A"] += 1000
        second = sample_vla(m, enc, steps=3, generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(first["A"], second["A"], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first["A"]).all())

    def test_understanding_predicts_instruction_not_plan(self):
        m, enc = model(), batch()
        enc["present"]["L"].fill_(False)
        torch.manual_seed(5)
        loss, metrics = vla_flow_loss(m, enc, config("task_understanding"))
        enc["L"] += 1000
        torch.manual_seed(5)
        same, _ = vla_flow_loss(m, enc, config("task_understanding"))
        torch.testing.assert_close(loss, same, rtol=0, atol=0)
        self.assertEqual(int(metrics["active_L"]), int(enc["instruction_mask"].sum()))

    def test_goal_prediction_uses_goal_not_next_frame(self):
        m, enc = model(), batch()
        enc["present"]["V"].fill_(False)
        torch.manual_seed(6)
        first, _ = vla_flow_loss(m, enc, config("goal_prediction"))
        enc["V"] += 1000
        torch.manual_seed(6)
        second, _ = vla_flow_loss(m, enc, config("goal_prediction"))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_task_validation(self):
        for tasks in ({"bad": 1}, {"action_policy": -1}, {"action_policy": float("nan")}):
            with self.assertRaises(ValueError):
                validate_tasks(tasks)


if __name__ == "__main__":
    unittest.main()
