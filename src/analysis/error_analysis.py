"""오차 분석 — 단일 RMSE 너머 안전(놓침) 관점.
RUL 구간별 정확도(임박할수록 중요), 늦은예측(위험) vs 이른예측(보수), 최악 엔진.
모델 디렉토리(experiments/demo/<fd>_<model> 또는 best)에서 로드해 test 예측 분석.

사용: python -m src.analysis.error_analysis [--dir experiments/demo/FD001_tcn]
"""
from __future__ import annotations
import argparse, json, pickle
import numpy as np
import torch
from ..model import build
from .. import data as D
from ..metrics import both
from ..config import EXP

OUT = EXP / "analysis"; OUT.mkdir(parents=True, exist_ok=True)


def predict(net, X, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(net(torch.tensor(X[i:i + bs])).cpu().numpy())
    return np.concatenate(out)


def analyze(model_dir):
    from pathlib import Path
    p = Path(model_dir); b = pickle.load(open(p / "bundle.pkl", "rb"))
    fd = p.name.split("_")[0]
    net = build(b["model"], b["n_feat"], b["window"], **b["hp"])
    net.load_state_dict(torch.load(p / "best.pt", map_location="cpu")); net.eval()
    d = D.load(fd, window=b["window"])
    pred = np.clip(predict(net, d["Xte"]), 0, None); true = d["yte"]
    d_err = pred - true                                  # >0 늦음(위험), <0 이름(보수)

    # RUL 구간별 RMSE (임박=0~25가 안전상 가장 중요)
    buckets = [(0, 25, "임박(0-25)"), (25, 75, "중간(25-75)"), (75, 126, "건강(75+)")]
    by_bucket = []
    for lo, hi, name in buckets:
        m = (true >= lo) & (true < hi)
        if m.sum():
            by_bucket.append({"range": name, "n": int(m.sum()),
                              "rmse": round(float(np.sqrt(np.mean((pred[m] - true[m]) ** 2))), 2),
                              "mean_signed_err": round(float(d_err[m].mean()), 2)})
    late = int((d_err > 0).sum()); early = int((d_err < 0).sum())
    worst = np.argsort(-np.abs(d_err))[:10]
    res = {
        "fd": fd, "model": b["model"], "n_test": int(len(true)),
        "overall": {k: round(v, 2) for k, v in both(pred, true).items()},
        "by_rul_bucket": by_bucket,
        "safety": {"late_predictions(위험)": late, "early_predictions(보수)": early,
                   "late_ratio": round(late / len(true), 3),
                   "mean_signed_err": round(float(d_err.mean()), 2)},
        "worst_engines": [{"unit": int(worst[i] + 1), "true": round(float(true[worst[i]]), 1),
                           "pred": round(float(pred[worst[i]]), 1), "err": round(float(d_err[worst[i]]), 1)}
                          for i in range(len(worst))],
    }
    json.dump(res, open(OUT / f"error_{fd}_{b['model']}.json", "w"), ensure_ascii=False, indent=2)
    print(f"[error {fd}/{b['model']}] overall rmse={res['overall']['rmse']} | "
          f"임박RMSE={by_bucket[0]['rmse']} | 늦은예측 {late}/{len(true)}({res['safety']['late_ratio']*100:.0f}%)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(EXP / "demo" / "FD001_tcn"))
    a = ap.parse_args(); analyze(a.dir)
