"""Train the v2 card-play network on oracle-labelled positions (rl/datagen.py).

    python -m rl.train_v2 --data data/v2/round0 --obs visible --epochs 12 \
        --out rl/checkpoints/v2_visible.pt

Several --data directories can be given (DAgger: every round's positions are
kept and the network is retrained on all of them). --init warm-starts from an
earlier v2 checkpoint.

Loss per position (see rl/network_v2.py for the reasoning):
  set cross-entropy on the optimal cards + beta * expected regret under the
  policy + c_q * per-card value MSE + c_v * state value MSE.

Reported on a held-out 2% of games (whole games, so no position of a held-out
game is ever trained on):
  opt%     how often the policy's top card is double-dummy optimal
  regret   tricks thrown away per decision by the policy's top card
  q-opt%   the same for the argmax of the q head
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time
import zlib
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from rl.network_v2 import CardPlayNet


def load_dirs(dirs: List[str], obs: str):
    key = "x_full" if obs == "full" else "x_vis"
    xs, qs, games, tricks = [], [], [], []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(path)
            xs.append(z[key]); qs.append(z["q"]); games.append(z["game"] + zlib.crc32(d.encode()) % 1000 * 10**9)  # unique ids; the split below uses game % 50 only
            tricks.append(z["trick"])
    return np.concatenate(xs), np.concatenate(qs), np.concatenate(games), np.concatenate(tricks)


def losses(net, x, q, beta=0.5, c_q=0.05, c_v=0.05):
    mask = q >= 0
    qf = q.float()
    vstar = qf.masked_fill(~mask, -1).max(dim=1).values
    opt = mask & (qf == vstar[:, None])
    regret = (vstar[:, None] - qf).clamp(min=0) * mask
    logits, qhat, v = net(x)
    logp = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
    ce = -torch.logsumexp(logp.masked_fill(~opt, -1e9), dim=-1).mean()
    exp_regret = (logp.exp() * regret).sum(-1).mean()
    lq = (((qhat - qf) ** 2) * mask).sum() / mask.sum()
    lv = ((v - vstar) ** 2).mean()
    total = ce + beta * exp_regret + c_q * lq + c_v * lv
    with torch.no_grad():
        top = logits.masked_fill(~mask, -1e9).argmax(-1)
        qtop = qhat.masked_fill(~mask, -1e9).argmax(-1)
        stats = {
            "opt": opt.gather(1, top[:, None]).float().mean().item(),
            "regret": regret.gather(1, top[:, None]).float().mean().item(),
            "q_opt": opt.gather(1, qtop[:, None]).float().mean().item(),
            "q_regret": regret.gather(1, qtop[:, None]).float().mean().item(),
            "ce": ce.item(), "lq": lq.item(), "lv": lv.item(),
        }
    return total, stats


@torch.no_grad()
def evaluate(net, X, Q, bs=8192, device="cpu"):
    net.eval()
    agg, n = {}, 0
    for i in range(0, len(X), bs):
        x = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device) / 13.0
        q = torch.as_tensor(Q[i:i + bs], dtype=torch.int64, device=device)
        _, st = losses(net, x, q)
        for k, v in st.items():
            agg[k] = agg.get(k, 0.0) + v * len(x)
        n += len(x)
    net.train()
    return {k: v / n for k, v in agg.items()}


def train(data: List[str], obs: str, out: str, epochs: int = 12, bs: int = 1024, lr: float = 2e-3,
          width: int = 384, blocks: int = 3, init: str | None = None, seed: int = 0, threads: int = 0,
          beta: float = 0.5, dropout: float = 0.0, weight_decay: float = 1e-4,
          device: str | None = None):
    torch.manual_seed(seed)
    if threads:
        torch.set_num_threads(threads)
    X, Q, G, T = load_dirs(data, obs)
    val = (G % 50) == 0
    Xtr, Qtr, Xva, Qva = X[~val], Q[~val], X[val], Q[val]
    print(f"{len(Xtr)} train / {len(Xva)} held-out positions, obs={obs}, features={X.shape[1]}", flush=True)

    net = CardPlayNet.load(init) if init else CardPlayNet(obs=obs, width=width, blocks=blocks, dropout=dropout)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f"training on {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    Xtr_t = torch.from_numpy(Xtr)
    Qtr_t = torch.from_numpy(Qtr.astype(np.int64))
    history = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(Xtr_t))
        run = {}
        for k, i in enumerate(range(0, len(perm), bs)):
            b = perm[i:i + bs]
            x = Xtr_t[b].to(device, non_blocking=True).float() / 13.0
            loss, st = losses(net, x, Qtr_t[b].to(device, non_blocking=True), beta=beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            for kk, v in st.items():
                run[kk] = run.get(kk, 0.0) + v
        nb = k + 1
        ev = evaluate(net, Xva, Qva, device=device)
        history.append({"epoch": ep, "train_opt": run["opt"] / nb, **{f"val_{k}": v for k, v in ev.items()}})
        print(f"epoch {ep:2d}  train opt {run['opt'] / nb:.3f}  |  held-out: opt {ev['opt']:.3f}  "
              f"regret {ev['regret']:.3f}  q-opt {ev['q_opt']:.3f}  q-regret {ev['q_regret']:.3f}  "
              f"value-mse {ev['lv']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        net.save(out, {"data": data, "history": history, "positions": int(len(Xtr))})
    print(f"saved {out}")
    return net, history


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--obs", choices=["visible", "full"], default="visible")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None,
                    help="default: cuda if available, else cpu")
    ap.add_argument("--init", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    train(a.data, a.obs, a.out, a.epochs, a.bs, a.lr, a.width, a.blocks, a.init, a.seed, a.threads, a.beta,
          a.dropout, a.weight_decay, a.device)


if __name__ == "__main__":
    main()
