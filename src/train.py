"""학습 — 한 subset(FD00x) × 모델(cnn/lstm). 엔진단위 val RMSE로 best 저장.

사용: python -m src.train --fd FD001 --model cnn --epochs 60
산출: experiments/<fd>_<model>/  (best.pt, scaler.npz, history.json, val_metrics.json)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from . import data as D
from .model import build
from .metrics import both
from .config import EXP, SEED


def run(fd="FD001", model="cnn", epochs=60, batch=512, lr=1e-3, window=30, patience=12):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = D.load(fd, window=window)
    out = EXP / f"{fd}_{model}"; out.mkdir(parents=True, exist_ok=True)

    tl = DataLoader(TensorDataset(torch.tensor(d["Xtr"]), torch.tensor(d["ytr"])),
                    batch_size=batch, shuffle=True, drop_last=False)
    Xva = torch.tensor(d["Xva"]).to(dev); yva = d["yva"]

    net = build(model, d["n_feat"], window).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    lossf = torch.nn.MSELoss()

    hist, best, best_ep, wait = [], 1e9, -1, 0
    for ep in range(1, epochs + 1):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xva).cpu().numpy()
        m = both(pv, yva); sched.step(m["rmse"])
        hist.append({"epoch": ep, **m})
        if m["rmse"] < best - 1e-4:
            best, best_ep, wait = m["rmse"], ep, 0
            torch.save(net.state_dict(), out / "best.pt")
            json.dump({"fd": fd, "model": model, "epoch": ep, **m,
                       "n_feat": d["n_feat"], "window": window,
                       "n_units_train": d["n_units_train"], "n_units_val": d["n_units_val"]},
                      open(out / "val_metrics.json", "w"), indent=2)
        else:
            wait += 1
        print(f"[{fd}/{model}] ep{ep:02d} val_rmse={m['rmse']:.3f} nasa={m['nasa_score']:.1f} best={best:.3f}@{best_ep}", flush=True)
        if wait >= patience:
            print(f"  early stop @ {ep}", flush=True); break

    np.savez(out / "scaler.npz", mu=d["mu"], sd=d["sd"], cols=np.array(d["cols"]))
    json.dump(hist, open(out / "history.json", "w"), indent=2)
    print(f"DONE {fd}/{model}: best val_rmse={best:.3f} @ep{best_ep}", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--model", default="cnn", choices=["cnn", "lstm"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--window", type=int, default=30)
    a = ap.parse_args()
    run(a.fd, a.model, a.epochs, a.batch, window=a.window)
