"""Measure frozen E->Q->D reconstruction on a prepared feature-cache record.

python -m scripts.validate_x_tokenizer --config configs/vla/x_tokenizer_frozen.yaml \
    --record data/robot/window.pt --device cpu

No model download or dependency installation is performed. MAE/RMSE are normalized
native-action errors, not robot success metrics; accept thresholds per control task.
"""
import argparse
import json
import torch
from torch.utils.data import default_collate
from juice.action_codec import FrozenXTokenizer
from juice.train.config import TrainConfig
from juice.train.vla_data import prepare_record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    if cfg.action_representation != "x_tokenizer":
        raise ValueError("configuration must select x_tokenizer")
    codec = FrozenXTokenizer.from_config(cfg, args.device)
    raw = prepare_record(torch.load(args.record, map_location="cpu", weights_only=True), cfg)
    def move(x):
        return {k: move(v) for k, v in x.items()} if isinstance(x, dict) else x.to(args.device)
    batch = move(default_collate([raw]))
    print(json.dumps({"codec": codec.metadata, "reconstruction": codec.reconstruction_metrics(batch)}, indent=2))


if __name__ == "__main__":
    main()
