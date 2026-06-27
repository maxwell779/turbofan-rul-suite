"""Conformal 예측구간 — 분포가정 없이 목표 커버리지를 보장하는 RUL 신뢰구간.
Split conformal: 무누수 val을 캘리브레이션으로 |pred-true| 잔차의 (1-α) 분위수 q →
test 구간 [pred-q, pred+q]. 경험적 커버리지가 목표(예 90%)에 수렴. PdM 안전 핵심.
출처: conformal prediction(Vovk), RUL conformal(arXiv 2212.14612).

사용: python -m src.mlops.conformal [--dir experiments/demo/FD001_tcn --alpha 0.1]
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch
from ..model import build
from .. import data as D
from ..config import EXP

OUT = EXP / "mlops"; OUT.mkdir(parents=True, exist_ok=True)


def predict(net, X, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net(torch.tensor(X[i:i + bs])).cpu().numpy())
    return np.concatenate(out)


def run(model_dir, alpha=0.1):
    p = Path(model_dir); b = pickle.load(open(p / "bundle.pkl", "rb")); fd = p.name.split("_")[0]
    net = build(b["model"], b["n_feat"], b["window"], **b["hp"])
    net.load_state_dict(torch.load(p / "best.pt", map_location="cpu")); net.eval()
    d = D.load(fd, window=b["window"])
    # 캘리브레이션 = 무누수 val 잔차
    res_cal = np.abs(predict(net, d["Xva"]) - d["yva"])
    n = len(res_cal); q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # finite-sample 보정
    q = float(np.quantile(res_cal, q_level))
    pred = np.clip(predict(net, d["Xte"]), 0, None); true = d["yte"]
    lo, hi = np.clip(pred - q, 0, None), pred + q
    cov = float(np.mean((true >= lo) & (true <= hi)))
    out = {"fd": fd, "model": b["model"], "alpha": alpha, "target_coverage": round(1 - alpha, 2),
           "empirical_coverage": round(cov, 3), "interval_halfwidth_q": round(q, 2),
           "mean_width": round(2 * q, 2), "n_cal": int(n), "n_test": int(len(true))}
    json.dump(out, open(OUT / f"conformal_{fd}_{b['model']}.json", "w"), indent=2)
    print(f"[conformal {fd}/{b['model']}] 목표 {1-alpha:.0%} → 실측 커버리지 {cov:.1%} | ±{q:.1f} 사이클", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(EXP / "demo" / "FD001_tcn")); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); run(a.dir, a.alpha)
