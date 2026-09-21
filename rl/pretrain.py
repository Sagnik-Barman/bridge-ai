"""Supervised imitation warm start: fit the policy network to the labels
from rl/generate_labels.py, before self-play RL takes over.

Full pipeline:

    python -m rl.generate_labels --num-deals 3000 --out rl/checkpoints/imitation_labels.npz
    python -m rl.pretrain --labels rl/checkpoints/imitation_labels.npz --out rl/checkpoints/warm_start.pt
    python -m rl.train --resume-from rl/checkpoints/warm_start.pt --iterations 5000 \\
        --games-per-iter 20 --heuristic-opponent-prob 0.5 --out rl/checkpoints/latest.pt

The last step is your existing self-play training command, unchanged
except for `--resume-from` pointing at the warm-started checkpoint instead
of starting from random weights -- `rl/train.py` already supports
resuming, so nothing there needed to change.

See docs/rl_training.md ("A fourth attempt") for why this exists, and
rl/generate_labels.py's own docstring for the honest scope of what's
actually being imitated (exact near the end of the hand, DDS/PIMC-informed
at trick leads, silent about everything else).
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from rl.network import PolicyValueNet


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    xs, masks, actions = data["x"], data["mask"], data["action"]
    return list(zip(xs, masks, actions))


def pretrain(
    dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    out_path: str,
    device: str | None = None,
) -> PolicyValueNet:
    rng = random.Random(seed)
    net = PolicyValueNet(seed=seed, device=device)
    print(f"Pretraining on device: {net.device}  ({len(dataset)} labeled examples, {epochs} epochs)")

    for epoch in range(1, epochs + 1):
        rng.shuffle(dataset)
        losses = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            stats = net.pretrain_step(batch, lr=lr)
            losses.append(stats["loss"])
        print(f"epoch {epoch:3d}/{epochs}  avg_loss={np.mean(losses):.4f}")

    net.save(out_path)
    print(f"Saved warm-started checkpoint to {out_path}")
    print(
        "Next: python -m rl.train --resume-from "
        f"{out_path} --iterations 5000 --games-per-iter 20 --heuristic-opponent-prob 0.5"
    )
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised imitation warm start for the RL policy network.")
    parser.add_argument("--labels", type=str, required=True, help="Path to a dataset from rl/generate_labels.py.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="rl/checkpoints/warm_start.pt")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()

    dataset = load_dataset(args.labels)
    if not dataset:
        raise SystemExit(f"No examples found in {args.labels} -- run rl/generate_labels.py first.")

    pretrain(dataset, args.epochs, args.batch_size, args.lr, args.seed, args.out, device=args.device)


if __name__ == "__main__":
    main()
