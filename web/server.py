"""PdM-RUL 콘솔 백엔드 (FastAPI).

experiments/ 의 학습 결과(리더보드·history·test_pred)와 data/ 의 원시 시계열을
읽어 콘솔에 제공. /api/engine 은 학습 모델로 한 엔진의 사이클별 RUL을 추론한다.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D
from src.model import build
from src.metrics import both
from src.config import EXP, DATA, SUBSETS, WINDOW

app = FastAPI(title="PdM-RUL Console")
HERE = Path(__file__).resolve().parent
_cache = {}


def runs():
    out = []
    for d in sorted(EXP.glob("FD*_*")):
        fd, mdl = d.name.split("_")
        v = d / "val_metrics.json"; t = d / "test_metrics.json"
        out.append({"fd": fd, "model": mdl,
                    "val": json.load(open(v)) if v.exists() else None,
                    "test": json.load(open(t)) if t.exists() else None})
    return out


@app.get("/api/leaderboard")
def leaderboard():
    f = EXP / "leaderboard.csv"
    if not f.exists():
        return JSONResponse({"rows": []})
    return {"rows": pd.read_csv(f).to_dict("records")}


@app.get("/api/runs")
def api_runs():
    return {"runs": runs()}


@app.get("/api/history")
def history(fd: str, model: str):
    f = EXP / f"{fd}_{model}" / "history.json"
    return {"history": json.load(open(f)) if f.exists() else []}


@app.get("/api/pred")
def pred(fd: str, model: str):
    f = EXP / f"{fd}_{model}" / "test_pred.csv"
    if not f.exists():
        return {"rows": []}
    return {"rows": pd.read_csv(f).round(2).to_dict("records")}


@app.get("/api/engines")
def engines(fd: str):
    df = D._read(DATA / f"test_{fd}.txt")
    truth = pd.read_csv(DATA / f"RUL_{fd}.txt", header=None).iloc[:, 0].values
    us = sorted(df["unit"].unique().tolist())
    return {"engines": [{"unit": int(u), "cycles": int((df["unit"] == u).sum()),
                         "true_rul": float(truth[i])} for i, u in enumerate(us)]}


def _load_model(fd, model):
    key = (fd, model)
    if key in _cache:
        return _cache[key]
    d = EXP / f"{fd}_{model}"
    sc = np.load(d / "scaler.npz", allow_pickle=True)
    cols = list(sc["cols"]); mu = sc["mu"]; sd = sc["sd"]
    net = build(model, len(cols), WINDOW)
    net.load_state_dict(torch.load(d / "best.pt", map_location="cpu")); net.eval()
    _cache[key] = (net, cols, mu, sd)
    return _cache[key]


@app.get("/api/engine")
def engine(fd: str, model: str, unit: int):
    """한 test 엔진의 사이클별 RUL 예측 + 주요 센서 곡선."""
    net, cols, mu, sd = _load_model(fd, model)
    df = D._read(DATA / f"test_{fd}.txt")
    truth = pd.read_csv(DATA / f"RUL_{fd}.txt", header=None).iloc[:, 0].values
    us = sorted(df["unit"].unique().tolist())
    true_end = float(truth[us.index(unit)])
    g = df[df["unit"] == unit].sort_values("cycle")
    arr = ((g[cols].values - mu) / sd).astype(np.float32)
    n = len(arr)
    pad = arr
    if n < WINDOW:
        pad = np.concatenate([np.repeat(arr[:1], WINDOW - n, 0), arr], 0)
    wins = [pad[max(0, i - WINDOW + 1):i + 1] for i in range(len(pad))]
    wins = [w if len(w) == WINDOW else np.concatenate([np.repeat(w[:1], WINDOW - len(w), 0), w], 0) for w in wins]
    with torch.no_grad():
        pr = net(torch.tensor(np.stack(wins[-n:]))).numpy()
    pr = np.clip(pr, 0, None)
    cyc = g["cycle"].values.tolist()
    # 실제 RUL: 마지막 사이클이 true_end, 그 이전은 +1씩 (cap 125)
    true_line = np.clip(true_end + (np.array(cyc[-1]) - np.array(cyc)), 0, 125).tolist()
    # 대표 센서 4종(분산 큰 것)
    key_sensors = ["s2", "s3", "s4", "s11"]
    key_sensors = [s for s in key_sensors if s in g.columns][:4]
    sensors = {s: g[s].round(3).tolist() for s in key_sensors}
    return {"unit": unit, "cycle": cyc, "pred_rul": pr.round(2).tolist(),
            "true_rul": [round(x, 1) for x in true_line], "true_end": true_end,
            "sensors": sensors}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
