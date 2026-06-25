"""고도화 EDA — C-MAPSS 시계열 진단.

산출:
  docs/images/eda/*.png   (수명분포·열화곡선·센서랭킹·상관·운전조건군집·정규화효과)
  experiments/eda/<fd>.json  (콘솔/리포트용 요약 지표)

핵심 PHM 특징선택 지표(센서가 RUL 예측에 유용한가):
  - monotonicity: 엔진별 단조 증가/감소 일관성 |#증가-#감소|/(n-1) 평균
  - |corr(RUL)|: 센서값과 RUL의 절대 상관 평균
  - prognosability: 고장시점 센서값의 엔진간 산포가 작을수록↑ (exp(-std/range))
운전조건 군집: op-setting을 KMeans로 군집(C-MAPSS는 FD002/004=6조건) → 조건별
센서 분포 차이를 보여 'FD002/004는 조건별 정규화가 필요'를 시각적으로 정당화.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller

from . import data as D
from .config import DATA, ROOT, SUBSETS

IMG = ROOT / "docs" / "images" / "eda"
OUT = ROOT / "experiments" / "eda"
IMG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
SENS = [f"s{i}" for i in range(1, 22)]
OPS = ["op1", "op2", "op3"]
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "Malgun Gothic", "axes.unicode_minus": False})


def _load(fd):
    df = D.add_rul(D._read(DATA / f"train_{fd}.txt"))
    return df


def monotonicity(df, col):
    vals = []
    for _, g in df.groupby("unit"):
        x = g.sort_values("cycle")[col].values
        d = np.diff(x)
        if len(d) == 0: continue
        vals.append(abs((d > 0).sum() - (d < 0).sum()) / len(d))
    return float(np.mean(vals)) if vals else 0.0


def prognosability(df, col):
    fails, ranges = [], []
    for _, g in df.groupby("unit"):
        x = g.sort_values("cycle")[col].values
        fails.append(x[-1]); ranges.append(x.max() - x.min())
    fails = np.array(fails); rng = np.mean(ranges) + 1e-9
    return float(np.exp(-np.std(fails) / rng))


def sensor_table(fd):
    df = _load(fd)
    rows = []
    for s in SENS:
        std = df[s].std()
        if std < 1e-6:
            rows.append({"sensor": s, "std": 0, "mono": 0, "corr_rul": 0, "prog": 0, "const": True})
            continue
        corr = abs(np.corrcoef(df[s], df["RUL"])[0, 1])
        rows.append({"sensor": s, "std": round(float(std), 4),
                     "mono": round(monotonicity(df, s), 4),
                     "corr_rul": round(float(corr), 4),
                     "prog": round(prognosability(df, s), 4), "const": False})
    t = pd.DataFrame(rows)
    t["score"] = (t["mono"] + t["corr_rul"]) / 2  # 종합 유용도
    return df, t.sort_values("score", ascending=False)


def fig_lifespan():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for fd in SUBSETS:
        df = D._read(DATA / f"train_{fd}.txt")
        life = df.groupby("unit")["cycle"].max()
        ax.hist(life, bins=25, alpha=0.5, label=f"{fd} (n={life.size}, μ={life.mean():.0f})")
    ax.set_xlabel("엔진 수명(사이클)"); ax.set_ylabel("엔진 수"); ax.legend(fontsize=7)
    ax.set_title("subset별 엔진 수명 분포")
    fig.tight_layout(); fig.savefig(IMG / "lifespan_dist.png"); plt.close(fig)


def fig_degradation(fd, top_sensors):
    df = _load(fd)
    units = df["unit"].unique()[:6]
    fig, axes = plt.subplots(1, len(top_sensors), figsize=(3.2 * len(top_sensors), 3))
    for ax, s in zip(np.atleast_1d(axes), top_sensors):
        for u in units:
            g = df[df["unit"] == u].sort_values("cycle")
            ax.plot(g["RUL"][::-1].values, g[s].values, alpha=0.7, lw=1)
        ax.invert_xaxis(); ax.set_xlabel("RUL →0(고장)"); ax.set_title(f"{s}")
    fig.suptitle(f"{fd} 열화 곡선 (RUL 감소에 따른 센서 추세, 엔진 6개)")
    fig.tight_layout(); fig.savefig(IMG / f"{fd}_degradation.png"); plt.close(fig)


def fig_sensor_rank(fd, t):
    tt = t[~t["const"]].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(tt))
    ax.barh(y - 0.2, tt["mono"], height=0.4, label="monotonicity", color="#3b82f6")
    ax.barh(y + 0.2, tt["corr_rul"], height=0.4, label="|corr(RUL)|", color="#f59e0b")
    ax.set_yticks(y); ax.set_yticklabels(tt["sensor"]); ax.invert_yaxis()
    ax.legend(fontsize=8); ax.set_title(f"{fd} 센서 유용도 랭킹 (단조성·RUL상관)")
    fig.tight_layout(); fig.savefig(IMG / f"{fd}_sensor_rank.png"); plt.close(fig)


def fig_corr(fd, df):
    cols = [s for s in SENS if df[s].std() > 1e-6]
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df[cols + ["RUL"]].corr(), cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"shrink": .7}, square=False)
    ax.set_title(f"{fd} 센서-센서/RUL 상관")
    fig.tight_layout(); fig.savefig(IMG / f"{fd}_corr.png"); plt.close(fig)


def fig_regimes(fd, k=6):
    df = D._read(DATA / f"train_{fd}.txt")
    X = df[OPS].values
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X)
    df = df.copy(); df["regime"] = km.labels_
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    sc = axes[0].scatter(df["op1"], df["op2"], c=df["regime"], cmap="tab10", s=4, alpha=.5)
    axes[0].set_xlabel("op1"); axes[0].set_ylabel("op2"); axes[0].set_title(f"{fd} 운전조건 {k}군집")
    # 대표 센서 한 개의 조건별 분포 차이
    s = "s2"
    for r in range(k):
        axes[1].hist(df[df["regime"] == r][s], bins=30, alpha=0.5)
    axes[1].set_xlabel(f"{s} 값"); axes[1].set_title(f"{s}: 운전조건별 분포가 분리됨 → 조건별 정규화 필요")
    fig.tight_layout(); fig.savefig(IMG / f"{fd}_regimes.png"); plt.close(fig)
    return int(k)


def fig_norm_effect(fd, k=6):
    """조건별 z-정규화 전/후 — 한 센서가 열화신호로 정렬되는 효과."""
    df = D.add_rul(D._read(DATA / f"train_{fd}.txt")).copy()
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(df[OPS].values)
    df["regime"] = km.labels_
    s = "s4"
    df["norm"] = df.groupby("regime")[s].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    u = df["unit"].unique()[0]; g = df[df["unit"] == u].sort_values("cycle")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(g["cycle"], g[s], lw=1); axes[0].set_title(f"{fd} {s} 원본(조건 뒤섞임)")
    axes[1].plot(g["cycle"], g["norm"], lw=1, color="#22c55e")
    axes[1].set_title(f"{s} 조건별 z-정규화 후(열화추세 또렷)")
    for a in axes: a.set_xlabel("cycle")
    fig.tight_layout(); fig.savefig(IMG / f"{fd}_norm_effect.png"); plt.close(fig)


def adf_summary(df, cols):
    out = {}
    for s in cols[:6]:
        try:
            p = adfuller(df[df["unit"] == df["unit"].iloc[0]][s].values, autolag="AIC")[1]
            out[s] = round(float(p), 4)
        except Exception:
            out[s] = None
    return out


def run():
    summary = {}
    fig_lifespan()
    for fd in SUBSETS:
        df, t = sensor_table(fd)
        good = [s for s in SENS if df[s].std() > 1e-6]
        top = t[~t["const"]]["sensor"].head(4).tolist()
        fig_degradation(fd, top)
        fig_sensor_rank(fd, t)
        fig_corr(fd, df)
        n_reg = 6 if fd in ("FD002", "FD004") else 1
        if n_reg > 1:
            fig_regimes(fd, n_reg); fig_norm_effect(fd, n_reg)
        life = df.groupby("unit")["cycle"].max()
        summary[fd] = {
            "n_engines": int(df["unit"].nunique()),
            "life_min": int(life.min()), "life_max": int(life.max()), "life_mean": round(float(life.mean()), 1),
            "n_regimes": n_reg,
            "constant_sensors": [s for s in SENS if df[s].std() <= 1e-6],
            "n_useful_sensors": len(good),
            "top_sensors": t[~t["const"]][["sensor", "mono", "corr_rul", "prog", "score"]].head(8).to_dict("records"),
            "adf_pvalue_sample": adf_summary(df, good),
        }
        print(f"[EDA {fd}] engines={summary[fd]['n_engines']} life μ={summary[fd]['life_mean']} "
              f"regimes={n_reg} useful_sensors={len(good)} top={top}", flush=True)
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=2)
    print("EDA done -> docs/images/eda/, experiments/eda/summary.json", flush=True)


if __name__ == "__main__":
    run()
