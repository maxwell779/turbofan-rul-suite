"""C-MAPSS 로더 — piecewise RUL 라벨, 엔진단위(무누수) 분할, 슬라이딩 윈도.

누수 통제 원칙: train/val 분할은 **엔진 unit 단위**로만 한다. 같은 엔진의
윈도가 train과 val에 동시에 들어가지 않게 해, 시계열 자기상관에 의한 점수
인플레를 차단한다(버스바 프로젝트의 '부품 단위 무누수'와 동일 철학).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import COLS, DATA, RUL_CAP, WINDOW, VAL_FRAC, SEED


def _read(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    last = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (last - df["cycle"]).clip(upper=RUL_CAP)
    return df


def select_features(train: pd.DataFrame):
    """train 기준 분산 0(상수) 컬럼 제거 → 유효 센서/op 컬럼 목록."""
    feat = [c for c in COLS if c not in ("unit", "cycle")]
    keep = [c for c in feat if train[c].std() > 1e-6]
    return keep


def fit_scaler(train: pd.DataFrame, cols):
    mu = train[cols].mean().values.astype(np.float32)
    sd = train[cols].std().replace(0, 1).values.astype(np.float32)
    return mu, sd


def _windows(df, cols, mu, sd, window):
    """엔진별 슬라이딩 윈도 → (N, window, F) + 라벨(윈도 마지막 cycle의 RUL)."""
    X, y = [], []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        arr = ((g[cols].values - mu) / sd).astype(np.float32)
        rul = g["RUL"].values.astype(np.float32)
        n = len(g)
        if n < window:                       # 짧은 엔진: 앞쪽을 첫 행으로 패딩
            pad = np.repeat(arr[:1], window - n, axis=0)
            arr = np.concatenate([pad, arr], 0)
            rul = np.concatenate([np.repeat(rul[:1], window - n), rul])
            n = window
        for i in range(n - window + 1):
            X.append(arr[i:i + window])
            y.append(rul[i + window - 1])
    return np.stack(X), np.asarray(y, np.float32)


def _last_window(df, cols, mu, sd, window):
    """엔진별 마지막 윈도 1개 (test 평가용)."""
    X = []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        arr = ((g[cols].values - mu) / sd).astype(np.float32)
        if len(arr) < window:
            arr = np.concatenate([np.repeat(arr[:1], window - len(arr), 0), arr], 0)
        X.append(arr[-window:])
    return np.stack(X)


def load(fd: str, window: int = WINDOW, val_frac: float = VAL_FRAC, seed: int = SEED):
    """반환: dict(train/val 윈도, test 마지막윈도+정답RUL, 메타)."""
    train = add_rul(_read(DATA / f"train_{fd}.txt"))
    test = _read(DATA / f"test_{fd}.txt")
    rul_truth = pd.read_csv(DATA / f"RUL_{fd}.txt", header=None).iloc[:, 0].values.astype(np.float32)
    rul_truth = np.clip(rul_truth, 0, RUL_CAP)

    cols = select_features(train)

    # 엔진 단위 무누수 분할
    units = train["unit"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_frac))
    val_u = set(units[:n_val].tolist())
    tr_df = train[~train["unit"].isin(val_u)]
    va_df = train[train["unit"].isin(val_u)]

    mu, sd = fit_scaler(tr_df, cols)          # 스케일러는 train 엔진만으로 적합
    Xtr, ytr = _windows(tr_df, cols, mu, sd, window)
    Xva, yva = _windows(va_df, cols, mu, sd, window)
    Xte = _last_window(test, cols, mu, sd, window)

    return {
        "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva,
        "Xte": Xte, "yte": rul_truth,
        "cols": cols, "mu": mu, "sd": sd,
        "n_feat": len(cols), "window": window,
        "n_units_train": len(units) - n_val, "n_units_val": n_val,
    }
