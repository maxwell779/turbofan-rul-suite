"""XAI — Captum Integrated Gradients로 RUL 예측 설명.
(시간×센서) 기여도 → 시간축 집계=센서 중요도, 센서축 집계=시간 saliency.
부호: 음수 기여=RUL을 끌어내림(고장 임박 신호). 출처: Captum IG, Sundararajan 2017.

사용: python -m src.mlops.xai [--fd FD001 --model tcn]
"""
from __future__ import annotations
import argparse, json, pickle
import numpy as np
import torch
from captum.attr import IntegratedGradients
from .. import data as D
from ..model import build
from ..config import EXP

OUT = EXP / "mlops"; OUT.mkdir(parents=True, exist_ok=True)


def run(fd="FD001", model="tcn", n_samples=80):
    p = EXP / "demo" / f"{fd}_{model}"
    b = pickle.load(open(p / "bundle.pkl", "rb")); cols = b["cols"]
    net = build(b["model"], b["n_feat"], b["window"], **b["hp"])
    net.load_state_dict(torch.load(p / "best.pt", map_location="cpu")); net.eval()
    d = D.load(fd, window=b["window"])
    X = torch.tensor(d["Xte"][:n_samples])            # 임박(낮은 RUL) 포함 test 엔진들
    ig = IntegratedGradients(net)
    attr = ig.attribute(X, baselines=torch.zeros_like(X), n_steps=48).detach().numpy()  # (N,W,F)

    sensor_imp = np.abs(attr).mean((0, 1))            # 센서별 평균 |기여|
    sensor_signed = attr.mean((0, 1))                 # 부호(음수=RUL↓ 유도)
    time_sal = np.abs(attr).mean((0, 2))              # 시간스텝별 saliency
    order = np.argsort(-sensor_imp)
    sensors = [{"sensor": cols[i], "importance": round(float(sensor_imp[i]), 5),
                "signed": round(float(sensor_signed[i]), 5)} for i in order]
    res = {"fd": fd, "model": model, "window": b["window"], "n_samples": int(len(X)),
           "method": "IntegratedGradients(n_steps=48, baseline=0)",
           "sensor_importance": sensors,
           "time_saliency": [round(float(v), 5) for v in time_sal]}
    json.dump(res, open(OUT / f"xai_{fd}_{model}.json", "w"), ensure_ascii=False, indent=2)
    top = ", ".join(f"{s['sensor']}({s['importance']:.3f})" for s in sensors[:5])
    print(f"[xai {fd}/{model}] top 센서 기여: {top}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fd", default="FD001"); ap.add_argument("--model", default="tcn")
    ap.add_argument("--n", type=int, default=80); a = ap.parse_args(); run(a.fd, a.model, a.n)
