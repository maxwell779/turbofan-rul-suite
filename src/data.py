"""C-MAPSS 로더 v2 — 표준 14센서, piecewise RUL, 엔진단위 무누수 분할,
조건별 정규화(FD002/004), 슬라이딩 윈도.

누수 통제: train/val 분할은 **엔진 unit 단위**. 스케일러(전역 또는 조건별)는
**train 엔진만으로 적합** 후 val/test에 적용.

조건별 정규화: op-setting 3개(고도·Mach·TRA)를 KMeans(6)로 군집 → **조건별
StandardScaler**. 6개 운전조건이 섞인 FD002/004에서 핵심(원신호는 조건에 따라
평균이 이동해 열화신호가 묻힘). 단일조건 FD001/003은 전역 z-score.
출처: arXiv 2603.00745·2511.19124, TowardsDataScience LSTM-PdM.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .config import COLS, DATA, RUL_CAP, WINDOW, VAL_FRAC, SEED

# 표준 14센서(상수 7개 s1,5,6,10,16,18,19 제거) — 문헌 합의
SENSORS14 = [f"s{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
OPS = ["op1", "op2", "op3"]
WINDOW_BY_FD = {"FD001": 30, "FD002": 40, "FD003": 40, "FD004": 40}
MULTI_COND = {"FD002", "FD004"}


def _read(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    return df


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    last = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (last - df["cycle"]).clip(upper=cap)
    return df


def select_features(train: pd.DataFrame, sensors14=True):
    if sensors14:
        return [c for c in SENSORS14 if train[c].std() > 1e-6]
    return [c for c in COLS if c not in ("unit", "cycle") and train[c].std() > 1e-6]


def _fit_transform(tr_df, cols, cond_norm, k=6):
    """train으로 변환 적합. 반환 bundle + apply 함수."""
    if cond_norm:
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(tr_df[OPS].values)
        scalers = {}
        lab = km.predict(tr_df[OPS].values)
        for r in range(k):
            m = lab == r
            sc = StandardScaler().fit(tr_df.loc[m, cols].values)
            scalers[r] = sc
        bundle = {"kind": "cond", "cols": cols, "k": k,
                  "centers": km.cluster_centers_,
                  "mean": {r: scalers[r].mean_ for r in scalers},
                  "scale": {r: scalers[r].scale_ for r in scalers}}
    else:
        mu = tr_df[cols].mean().values.astype(np.float32)
        sd = tr_df[cols].std().replace(0, 1).values.astype(np.float32)
        bundle = {"kind": "global", "cols": cols, "mean": mu, "scale": sd}
    return bundle


def apply_transform(df, bundle):
    """bundle로 df의 cols를 정규화한 (N,F) 배열 컬럼 추가 → '_z' 접미 df 반환."""
    cols = bundle["cols"]; df = df.copy()
    if bundle["kind"] == "global":
        z = (df[cols].values - bundle["mean"]) / bundle["scale"]
    else:
        centers = np.asarray(bundle["centers"])
        d = ((df[OPS].values[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        z = np.empty((len(df), len(cols)), np.float32)
        for r in range(bundle["k"]):
            m = lab == r
            if m.any():
                z[m] = (df.loc[m, cols].values - bundle["mean"][r]) / bundle["scale"][r]
    zc = [f"{c}_z" for c in cols]
    df[zc] = z.astype(np.float32)
    return df, zc


def _windows(df, zc, window):
    X, y = [], []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        arr = g[zc].values.astype(np.float32); rul = g["RUL"].values.astype(np.float32)
        n = len(g)
        if n < window:
            arr = np.concatenate([np.repeat(arr[:1], window - n, 0), arr], 0)
            rul = np.concatenate([np.repeat(rul[:1], window - n), rul]); n = window
        for i in range(n - window + 1):
            X.append(arr[i:i + window]); y.append(rul[i + window - 1])
    return np.stack(X), np.asarray(y, np.float32)


def _last_window(df, zc, window):
    X = []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle"); arr = g[zc].values.astype(np.float32)
        if len(arr) < window:
            arr = np.concatenate([np.repeat(arr[:1], window - len(arr), 0), arr], 0)
        X.append(arr[-window:])
    return np.stack(X)


def load(fd: str, window: int | None = None, cond_norm: bool | None = None,
         sensors14: bool = True, val_frac: float = VAL_FRAC, seed: int = SEED, cap: int = RUL_CAP):
    if window is None:
        window = WINDOW_BY_FD.get(fd, WINDOW)
    if cond_norm is None:
        cond_norm = fd in MULTI_COND

    train = add_rul(_read(DATA / f"train_{fd}.txt"), cap)
    test = _read(DATA / f"test_{fd}.txt")
    rul_truth = pd.read_csv(DATA / f"RUL_{fd}.txt", header=None).iloc[:, 0].values.astype(np.float32)
    rul_truth = np.clip(rul_truth, 0, cap)
    cols = select_features(train, sensors14)

    units = train["unit"].unique().copy()
    rng = np.random.default_rng(seed); rng.shuffle(units)
    n_val = max(1, int(len(units) * val_frac)); val_u = set(units[:n_val].tolist())
    tr_df = train[~train["unit"].isin(val_u)]; va_df = train[train["unit"].isin(val_u)]

    bundle = _fit_transform(tr_df, cols, cond_norm)
    tr_df, zc = apply_transform(tr_df, bundle)
    va_df, _ = apply_transform(va_df, bundle)
    te_df, _ = apply_transform(test, bundle)

    Xtr, ytr = _windows(tr_df, zc, window)
    Xva, yva = _windows(va_df, zc, window)
    Xte = _last_window(te_df, zc, window)
    # 전역 케이스 하위호환 필드
    mu = bundle.get("mean") if bundle["kind"] == "global" else None
    sd = bundle.get("scale") if bundle["kind"] == "global" else None
    return {"Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte, "yte": rul_truth,
            "cols": cols, "mu": mu, "sd": sd, "bundle": bundle,
            "n_feat": len(cols), "window": window, "cond_norm": cond_norm,
            "n_units_train": len(units) - n_val, "n_units_val": n_val}
