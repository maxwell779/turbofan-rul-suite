"""대규모 그리드서치 — 하이퍼격자 × 손실 × 윈도 × 시드 × 4 subset.
A100에 샤딩 병렬(--shard k --of N). 설정당 가중치 미저장(메트릭만) → 경량/고속.
시드별 결과를 모아 mean±std 리더보드 생성(merge).

워커:  python -m src.sweep --shard 0 --of 6
병합:  python -m src.sweep --merge
빠른:  python -m src.sweep --quick --shard 0 --of 1
"""
from __future__ import annotations
import argparse, glob, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from . import data as D
from .model import build
from .losses import get as get_loss
from .metrics import both
from .config import EXP, SUBSETS

SWEEP = EXP / "sweep"; SWEEP.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ── 하이퍼파라미터 격자 ───────────────────────────────────────────
GRID = {
    "cnn":     [dict(ch=c, k=k, pool=p, p=d) for c in (32, 64, 96, 128) for k in (3, 5, 7) for p in ("avg", "max") for d in (0.2, 0.3)],
    "lstm":    [dict(hid=h, layers=l, p=d) for h in (64, 96, 128, 160) for l in (1, 2, 3) for d in (0.2, 0.3)],
    "gru":     [dict(hid=h, layers=l, p=d) for h in (64, 96, 128) for l in (1, 2, 3) for d in (0.2, 0.3)],
    "bilstm":  [dict(hid=h, layers=l, p=d) for h in (64, 96, 128) for l in (1, 2) for d in (0.2, 0.3)],
    "cnnlstm": [dict(ch=c, hid=h, p=d) for c in (32, 64) for h in (64, 96, 128) for d in (0.2, 0.3)],
    "tcn":     [dict(ch=c, levels=l, p=d) for c in (32, 64, 96) for l in (3, 4, 5, 6) for d in (0.1, 0.2)],
    "dlinear": [dict(kernel=k) for k in (15, 25, 35)],
}
LOSSES = {"dlinear": ["mse"]}            # 기본 [mse, asym]
ML = [("ridge", dict(alpha=1.0)), ("ridge", dict(alpha=10.0)),
      ("rf", dict(n_estimators=400, max_depth=16)), ("svr", dict(C=10, gamma="scale"))]


def hp_tag(hp):
    return ",".join(f"{k}{v}" for k, v in hp.items())


def gen_configs(fds, seeds, windows):
    cfgs = []
    for fd in fds:
        for model, hps in GRID.items():
            losses = LOSSES.get(model, ["mse", "asym"])
            for hp in hps:
                for loss in losses:
                    for w in windows:
                        for s in seeds:
                            cfgs.append(dict(fd=fd, model=model, hp=hp, loss=loss, window=w, seed=s))
    return cfgs


def featurize(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1, :], X[:, -1, :] - X[:, 0, :]], 1)


_CACHE = {}
def get_data(fd, window):
    key = (fd, window)
    if key not in _CACHE:
        _CACHE[key] = D.load(fd, window=window)
    return _CACHE[key]


def run_dl(cfg, d, epochs, patience=10, batch=512, lr=1e-3):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    tl = DataLoader(TensorDataset(torch.tensor(d["Xtr"]), torch.tensor(d["ytr"])), batch_size=batch, shuffle=True)
    Xva = torch.tensor(d["Xva"]).to(DEV)
    net = build(cfg["model"], d["n_feat"], d["window"], **cfg["hp"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    lf = get_loss(cfg["loss"]); best, bstate, wait = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEV), yb.to(DEV)
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
        pred = np.clip(net(torch.tensor(d["Xte"]).to(DEV)).cpu().numpy(), 0, None)
    tm = both(pred, d["yte"])
    return best, tm


def run_ml(kind, hp, d, seed):
    m = {"ridge": Ridge, "rf": RandomForestRegressor, "svr": SVR}[kind]
    kw = dict(hp);
    if kind == "rf": kw["random_state"] = seed
    model = m(**kw); model.fit(featurize(d["Xtr"]), d["ytr"])
    va = both(np.clip(model.predict(featurize(d["Xva"])), 0, None), d["yva"])["rmse"]
    tm = both(np.clip(model.predict(featurize(d["Xte"])), 0, None), d["yte"])
    return va, tm


def merge():
    files = glob.glob(str(SWEEP / "lb_shard*.csv"))
    if not files:
        print("no shards"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(SWEEP / "raw_all.csv", index=False)
    g = (df.groupby(["fd", "model", "loss", "window", "hp"])
           .agg(test_rmse=("test_rmse", "mean"), test_rmse_std=("test_rmse", "std"),
                nasa=("nasa", "mean"), val_rmse=("val_rmse", "mean"), n=("seed", "count"))
           .reset_index())
    g = g.round(3).sort_values(["fd", "test_rmse"])
    g.to_csv(SWEEP / "leaderboard.csv", index=False)
    print(f"merged {len(df)} runs → {len(g)} configs. BEST/fd:")
    for fd in sorted(g.fd.unique()):
        b = g[g.fd == fd].iloc[0]
        print(f"  {fd}: {b['model']}|{b['hp']}|{b['loss']}|w{b['window']}  rmse={b['test_rmse']}±{b['test_rmse_std']} nasa={b['nasa']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--seeds", default="42,7,123"); ap.add_argument("--windows", default="30,40,50")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--quick", action="store_true"); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge: merge(); return
    fds = [a.fd] if a.fd else SUBSETS
    seeds = [int(x) for x in a.seeds.split(",")]; windows = [int(x) for x in a.windows.split(",")]
    epochs = 20 if a.quick else a.epochs
    cfgs = gen_configs(fds, seeds, windows)
    mine = cfgs[a.shard::a.of]
    out = SWEEP / f"lb_shard{a.shard}.csv"; rows = []
    # ML은 shard 0에서만(빠름)
    if a.shard == 0:
        for fd in fds:
            d = get_data(fd, windows[0])
            for kind, hp in ML:
                t0 = time.time(); va, tm = run_ml(kind, hp, d, seeds[0])
                rows.append(dict(fd=fd, model=kind, loss="-", window=windows[0], hp=hp_tag(hp), seed=seeds[0],
                                 val_rmse=round(va, 3), test_rmse=round(tm["rmse"], 3), nasa=round(tm["nasa_score"], 1), sec=round(time.time()-t0,1)))
    print(f"[shard {a.shard}/{a.of}] configs={len(mine)} of {len(cfgs)} | seeds={seeds} windows={windows} epochs={epochs}", flush=True)
    for i, cfg in enumerate(mine):
        d = get_data(cfg["fd"], cfg["window"]); t0 = time.time()
        try:
            vr, tm = run_dl(cfg, d, epochs)
        except Exception as e:
            print(f"  ERR {cfg['model']} {cfg['hp']}: {str(e)[:60]}", flush=True); continue
        rows.append(dict(fd=cfg["fd"], model=cfg["model"], loss=cfg["loss"], window=cfg["window"], hp=hp_tag(cfg["hp"]),
                         seed=cfg["seed"], val_rmse=round(vr, 3), test_rmse=round(tm["rmse"], 3), nasa=round(tm["nasa_score"], 1), sec=round(time.time()-t0, 1)))
        if i % 20 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  [{a.shard}] {i+1}/{len(mine)} {cfg['fd']} {cfg['model']}|w{cfg['window']}|s{cfg['seed']} test={tm['rmse']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[shard {a.shard}] DONE {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
