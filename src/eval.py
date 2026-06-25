"""테스트 평가 — 학습된 모델로 test 엔진별 마지막 윈도 RUL 예측 → RMSE+NASA.

사용: python -m src.eval            (있는 모든 experiments/<fd>_<model> 평가 → 리더보드 CSV)
      python -m src.eval --fd FD001 --model cnn   (단일)
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
import torch

from . import data as D
from .model import build
from .metrics import both
from .config import EXP, SUBSETS


def eval_one(fd, model, window=30):
    out = EXP / f"{fd}_{model}"
    if not (out / "best.pt").exists():
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = D.load(fd, window=window)
    net = build(model, d["n_feat"], window).to(dev)
    net.load_state_dict(torch.load(out / "best.pt", map_location=dev)); net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(d["Xte"]).to(dev)).cpu().numpy()
    pred = np.clip(pred, 0, None)
    m = both(pred, d["yte"])
    json.dump({"fd": fd, "model": model, **m, "n_test": int(len(pred))},
              open(out / "test_metrics.json", "w"), indent=2)
    # 산점도용 예측 저장
    pd.DataFrame({"true": d["yte"], "pred": pred}).to_csv(out / "test_pred.csv", index=False)
    print(f"[TEST {fd}/{model}] rmse={m['rmse']:.3f}  nasa={m['nasa_score']:.1f}  (n={len(pred)})", flush=True)
    return {"fd": fd, "model": model, **m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd"); ap.add_argument("--model"); ap.add_argument("--window", type=int, default=30)
    a = ap.parse_args()
    fds = [a.fd] if a.fd else SUBSETS
    models = [a.model] if a.model else ["cnn", "lstm"]
    rows = []
    for fd in fds:
        for mdl in models:
            r = eval_one(fd, mdl, a.window)
            if r: rows.append(r)
    if rows:
        df = pd.DataFrame(rows).sort_values(["fd", "rmse"])
        df.to_csv(EXP / "leaderboard.csv", index=False)
        print("\n=== LEADERBOARD ===\n" + df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
