"""Treino do autoencoder SNN em eventos normais.
Uso: python -m src.train --epochs 30"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .model import SpikingAE


class EventDataset(Dataset):
    def __init__(self, root="data/normal"):
        self.files = sorted(Path(root).glob("*.npy"))
        if not self.files:
            raise SystemExit("sem dados — rode src.make_data")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return torch.from_numpy(np.load(self.files[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = EventDataset()
    n_test = max(1, int(0.15 * len(ds)))
    tr, te = random_split(ds, [len(ds) - n_test, n_test],
                          generator=torch.Generator().manual_seed(0))
    print(f"device={device} train={len(tr)} test={len(te)}")

    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    dl_te = DataLoader(te, batch_size=64)
    model = SpikingAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        for x in dl:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        tl = np.mean([loss_fn(model(x.to(device)), x.to(device)).item()
                      for x in dl_te])
        if tl < best:
            best = tl
            torch.save({"model": model.state_dict(), "test_mse": tl},
                       "checkpoints/spiking_ae.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep}: test_mse={tl:.5f} (best={best:.5f})")
    print(f"salvo checkpoints/spiking_ae.pt | best_mse={best:.5f}")


if __name__ == "__main__":
    main()
