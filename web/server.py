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


def _json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


@app.get("/api/eda")
def api_eda():
    return _json(EXP / "eda" / "summary.json") or {}


@app.get("/api/drift")
def api_drift(fd: str = "FD001"):
    return _json(EXP / "mlops" / f"drift_{fd}.json") or {}


@app.get("/api/xai")
def api_xai(fd: str = "FD001", model: str = "tcn"):
    return _json(EXP / "mlops" / f"xai_{fd}_{model}.json") or {}


@app.get("/api/latency")
def api_latency(fd: str = "FD001", model: str = "tcn"):
    return _json(EXP / "mlops" / f"latency_{fd}_{model}.json") or {}


@app.get("/api/sweep")
def api_sweep():
    """sweep 리더보드(완료 시 leaderboard.csv, 진행중이면 shard 부분합)."""
    import glob
    lb = EXP / "sweep" / "leaderboard.csv"
    if lb.exists():
        df = pd.read_csv(lb)
        # 모델 선택은 무누수 val 기준 → (fd,model)별 val_rmse 최소 행만(전 FD·전 모델 커버,
        # 4736 config 정렬상 head(N)이 FD001만 반환하던 문제 해결)
        best = (df.sort_values("val_rmse").groupby(["fd", "model"], as_index=False).first()
                  .sort_values(["fd", "test_rmse"]))
        return {"status": "done", "rows": best.round(3).to_dict("records")}
    shards = glob.glob(str(EXP / "sweep" / "lb_shard*.csv"))
    if not shards:
        return {"status": "pending", "rows": []}
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    g = (df.sort_values("test_rmse").groupby("fd").head(15)
         if len(df) else df)
    return {"status": "running", "n": int(len(df)), "rows": g.round(3).to_dict("records")}


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
    """demo/best 번들(신 코드 bundle.pkl) 로드. 없으면 None."""
    import pickle
    key = (fd, model)
    if key in _cache:
        return _cache[key]
    d = EXP / "demo" / f"{fd}_{model}"
    if not (d / "bundle.pkl").exists():
        return None
    b = pickle.load(open(d / "bundle.pkl", "rb"))
    net = build(b["model"], b["n_feat"], b["window"], **b["hp"])
    net.load_state_dict(torch.load(d / "best.pt", map_location="cpu")); net.eval()
    _cache[key] = (net, b)
    return _cache[key]


@app.get("/api/engine")
def engine(fd: str, model: str, unit: int):
    """한 test 엔진의 사이클별 RUL 예측 + 주요 센서 곡선 (신 코드 번들·조건정규화)."""
    loaded = _load_model(fd, model)
    if loaded is None:
        return JSONResponse({"error": "model not available", "hint": "demo/best 모델 필요"}, status_code=404)
    net, b = loaded
    bundle, W, cols = b["bundle"], b["window"], b["cols"]
    df = D._read(DATA / f"test_{fd}.txt")
    truth = pd.read_csv(DATA / f"RUL_{fd}.txt", header=None).iloc[:, 0].values
    us = sorted(df["unit"].unique().tolist())
    true_end = float(min(truth[us.index(unit)], 125))
    g = df[df["unit"] == unit].sort_values("cycle")
    gz, zc = D.apply_transform(g, bundle)
    arr = gz[zc].values.astype(np.float32); n = len(arr)
    pad = arr if n >= W else np.concatenate([np.repeat(arr[:1], W - n, 0), arr], 0)
    wins = [pad[max(0, i - W + 1):i + 1] for i in range(len(pad))]
    wins = [w if len(w) == W else np.concatenate([np.repeat(w[:1], W - len(w), 0), w], 0) for w in wins]
    with torch.no_grad():
        pr = np.clip(net(torch.tensor(np.stack(wins[-n:]))).numpy(), 0, None)
    cyc = g["cycle"].values.tolist()
    true_line = np.clip(true_end + (cyc[-1] - np.array(cyc)), 0, 125).tolist()
    key_sensors = [s for s in ("s2", "s3", "s4", "s11") if s in g.columns][:4]
    sensors = {s: g[s].round(3).tolist() for s in key_sensors}
    return {"unit": unit, "cycle": cyc, "pred_rul": pr.round(2).tolist(),
            "true_rul": [round(x, 1) for x in true_line], "true_end": true_end, "sensors": sensors}


@app.get("/api/conformal")
def api_conformal(fd: str = "FD001", model: str = "tcn"):
    return _json(EXP / "mlops" / f"conformal_{fd}_{model}.json") or {}


@app.get("/api/error")
def api_error(fd: str = "FD001", model: str = "tcn"):
    return _json(EXP / "analysis" / f"error_{fd}_{model}.json") or {}


@app.get("/api/foundation")
def api_foundation():
    return {"frozen": _json(EXP / "foundation" / "summary.json") or {},
            "lora": _json(EXP / "foundation" / "lora_summary.json") or {}}


@app.get("/api/demo_models")
def api_demo_models():
    """오차분석·conformal·XAI·지연이 있는 demo/best 모델 목록."""
    out = []
    for d in sorted((EXP / "demo").glob("*_*")):
        out.append(d.name)
    return {"models": out}


_UI = HERE / "ui" / "dist"


@app.get("/")
def index():
    if (_UI / "index.html").exists():
        return FileResponse(_UI / "index.html")
    return FileResponse(HERE / "static" / "index.html")


if (_UI / "assets").exists():
    app.mount("/assets", StaticFiles(directory=_UI / "assets"), name="assets")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
