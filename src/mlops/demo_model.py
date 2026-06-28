"""콘솔/배포 데모용 단일 모델 학습·저장 (sweep best 나오기 전 임시 깨끗 모델).
TCN: conv 기반이라 ONNX export·Captum 모두 깔끔. experiments/demo/<fd>_<model>/."""
from __future__ import annotations
import json, pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from .. import data as D
from ..model import build
from ..losses import get as get_loss
from ..metrics import both
from ..config import EXP

DEMO = EXP / "demo"; DEMO.mkdir(parents=True, exist_ok=True)


def train(fd="FD001", model="tcn", hp=None, loss="mse", epochs=45, lr=1e-3, patience=10,
          window=None, seed=42, max_train=None):
    hp = hp or dict(ch=64, levels=4, p=0.2)
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = D.load(fd, window=window)
    Xtr, ytr = d["Xtr"], d["ytr"]
    if max_train and len(Xtr) > max_train:        # CPU 데모용 학습윈도 subsample(엔진 무관 균등)
        idx = np.random.RandomState(seed).choice(len(Xtr), max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    tl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=512, shuffle=True)
    Xva = torch.tensor(d["Xva"]).to(dev)
    net = build(model, d["n_feat"], d["window"], **hp).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    lf = get_loss(loss); best, bstate, wait = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            r = both(net(Xva).cpu().numpy(), d["yva"])["rmse"]
        sch.step(r)
        if r < best - 1e-4: best, bstate, wait = r, {k: v.cpu().clone() for k, v in net.state_dict().items()}, 0
        else: wait += 1
        if wait >= patience: break
    net.load_state_dict(bstate); net.eval()
    with torch.no_grad():
        pred = np.clip(net(torch.tensor(d["Xte"]).to(dev)).cpu().numpy(), 0, None)
    tm = both(pred, d["yte"])
    out = DEMO / f"{fd}_{model}"; out.mkdir(exist_ok=True)
    torch.save(net.state_dict(), out / "best.pt")
    pickle.dump({"bundle": d["bundle"], "window": d["window"], "model": model, "hp": hp,
                 "n_feat": d["n_feat"], "cols": d["cols"]}, open(out / "bundle.pkl", "wb"))
    json.dump({"fd": fd, "model": model, "hp": hp, "loss": loss, "val_rmse": round(best, 3),
               "test_rmse": round(tm["rmse"], 3), "nasa": round(tm["nasa_score"], 1)}, open(out / "metrics.json", "w"), indent=2)
    print(f"[demo {fd}/{model}] val={best:.2f} test={tm['rmse']:.2f} nasa={tm['nasa_score']:.0f} -> {out}", flush=True)
    return out


if __name__ == "__main__":
    train()
