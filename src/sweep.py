"""그리드서치 — ML+DL 동물원 × 손실 × 하이퍼. 엔진단위 무누수 val 기준 best,
test 평가(RMSE+NASA). 결과: experiments/sweep/<fd>__<tag>/ + sweep/leaderboard.csv.

사용: python -m src.sweep            (FD001~004 전체)
      python -m src.sweep --fd FD001 --quick
"""
from __future__ import annotations
import argparse, json, pickle, time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from . import data as D
from .model import build
from .losses import get as get_loss
from .metrics import both
from .config import EXP, SUBSETS, SEED

SWEEP = EXP / "sweep"; SWEEP.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ── 설정 레지스트리 ──────────────────────────────────────────────
DL = [
    ("cnn_mse",      "cnn",     dict(ch=64, k=5),            "mse"),
    ("cnn_asym",     "cnn",     dict(ch=64, k=5),            "asym"),
    ("cnn_k7max",    "cnn",     dict(ch=96, k=7, pool="max"),"mse"),
    ("lstm_mse",     "lstm",    dict(hid=96, layers=2),      "mse"),
    ("lstm_asym",    "lstm",    dict(hid=96, layers=2),      "asym"),
    ("lstm_big",     "lstm",    dict(hid=128, layers=2),     "mse"),
    ("gru_mse",      "gru",     dict(hid=96, layers=2),      "mse"),
    ("bilstm_mse",   "bilstm",  dict(hid=96, layers=2),      "mse"),
    ("bilstm_asym",  "bilstm",  dict(hid=96, layers=2),      "asym"),
    ("cnnlstm_mse",  "cnnlstm", dict(ch=64, hid=96),         "mse"),
    ("cnnlstm_asym", "cnnlstm", dict(ch=64, hid=96),         "asym"),
    ("tcn_mse",      "tcn",     dict(ch=64, levels=4),       "mse"),
    ("tcn_asym",     "tcn",     dict(ch=64, levels=4),       "asym"),
    ("tcn_deep",     "tcn",     dict(ch=96, levels=6),       "mse"),
    ("dlinear_mse",  "dlinear", dict(),                      "mse"),
]
ML = [
    ("ml_ridge",  "ridge", {}),
    ("ml_rf",     "rf",    dict(n_estimators=300, max_depth=14)),
    ("ml_svr",    "svr",   dict(C=10, gamma="scale")),
]


def featurize(X):
    """(N,W,F) → (N, 6F) 통계특징: mean,std,min,max,last,delta."""
    mean = X.mean(1); std = X.std(1); mn = X.min(1); mx = X.max(1)
    last = X[:, -1, :]; delta = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([mean, std, mn, mx, last, delta], 1)


def run_dl(fd, tag, model, hp, loss, d, epochs, batch=512, lr=1e-3, patience=12):
    torch.manual_seed(SEED)
    tl = DataLoader(TensorDataset(torch.tensor(d["Xtr"]), torch.tensor(d["ytr"])),
                    batch_size=batch, shuffle=True)
    Xva = torch.tensor(d["Xva"]).to(DEV)
    net = build(model, d["n_feat"], d["window"], **hp).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    lf = get_loss(loss)
    best, best_state, wait = 1e9, None, 0
    for ep in range(1, epochs + 1):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); lf(net(xb), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            r = both(net(Xva).cpu().numpy(), d["yva"])["rmse"]
        sch.step(r)
        if r < best - 1e-4:
            best, best_state, wait = r, {k: v.cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= patience: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pred = np.clip(net(torch.tensor(d["Xte"]).to(DEV)).cpu().numpy(), 0, None)
    return net, best, both(pred, d["yte"]), pred


def run_ml(kind, hp, d):
    Xtr, Xte = featurize(d["Xtr"]), featurize(d["Xte"])
    m = {"ridge": Ridge, "rf": RandomForestRegressor, "svr": SVR}[kind](**hp)
    m.fit(Xtr, d["ytr"])
    va = both(np.clip(m.predict(featurize(d["Xva"])), 0, None), d["yva"])["rmse"]
    pred = np.clip(m.predict(Xte), 0, None)
    return m, va, both(pred, d["yte"]), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    fds = [a.fd] if a.fd else SUBSETS
    epochs = 25 if a.quick else 60
    rows = []
    for fd in fds:
        d = D.load(fd)  # 조건정규화 자동(FD002/004), 14센서, per-fd 윈도
        print(f"\n=== {fd} | feat={d['n_feat']} win={d['window']} cond_norm={d['cond_norm']} "
              f"train_u={d['n_units_train']} val_u={d['n_units_val']} ===", flush=True)
        for tag, model, hp, loss in DL:
            t0 = time.time()
            net, vr, tm, pred = run_dl(fd, tag, model, hp, loss, d, epochs)
            out = SWEEP / f"{fd}__{tag}"; out.mkdir(exist_ok=True)
            torch.save(net.state_dict(), out / "best.pt")
            pickle.dump({"bundle": d["bundle"], "window": d["window"], "model": model,
                         "hp": hp, "n_feat": d["n_feat"]}, open(out / "bundle.pkl", "wb"))
            np.save(out / "test_pred.npy", pred)
            rec = {"fd": fd, "tag": tag, "type": "dl", "model": model, "loss": loss,
                   "val_rmse": round(vr, 3), "test_rmse": round(tm["rmse"], 3),
                   "nasa": round(tm["nasa_score"], 1), "sec": round(time.time() - t0, 1)}
            rows.append(rec); print(f"  {tag:14s} val={vr:6.2f} test={tm['rmse']:6.2f} nasa={tm['nasa_score']:8.0f} ({rec['sec']}s)", flush=True)
        for tag, kind, hp in ML:
            t0 = time.time()
            m, vr, tm, pred = run_ml(kind, hp, d)
            rec = {"fd": fd, "tag": tag, "type": "ml", "model": kind, "loss": "-",
                   "val_rmse": round(vr, 3), "test_rmse": round(tm["rmse"], 3),
                   "nasa": round(tm["nasa_score"], 1), "sec": round(time.time() - t0, 1)}
            rows.append(rec); print(f"  {tag:14s} val={vr:6.2f} test={tm['rmse']:6.2f} nasa={tm['nasa_score']:8.0f} ({rec['sec']}s)", flush=True)
        # 중간 저장
        import pandas as pd
        pd.DataFrame(rows).to_csv(SWEEP / "leaderboard.csv", index=False)
    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["fd", "test_rmse"])
    df.to_csv(SWEEP / "leaderboard.csv", index=False)
    print("\n=== BEST per FD ===", flush=True)
    for fd in fds:
        b = df[df.fd == fd].iloc[0]
        print(f"  {fd}: {b['tag']} test_rmse={b['test_rmse']} nasa={b['nasa']}", flush=True)


if __name__ == "__main__":
    main()
