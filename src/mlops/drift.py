"""데이터 드리프트 모니터 — 센서 분포 변화(train=참조 vs test=운영) 탐지.

PSI(Population Stability Index): Σ (a%-e%)·ln(a%/e%). <0.1 안정 / 0.1~0.25 주의 / >0.25 경보.
KS 2표본: ECDF 최대거리 D, p<0.05면 분포 다름(대표본 과민 주의).
데이터셋 드리프트 = PSI>0.25 센서 비율. *주의: 진짜 열화 vs 센서 드리프트는 구분 한계.*
출처: fiddler PSI, deepchecks KS, Evidently.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from ..data import _read, SENSORS14
from ..config import DATA, EXP

OUT = EXP / "mlops"; OUT.mkdir(parents=True, exist_ok=True)


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(ref, edges)[0] / len(ref)
    a = np.histogram(cur, edges)[0] / len(cur)
    e = np.clip(e, 1e-4, None); a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run(fd):
    tr = _read(DATA / f"train_{fd}.txt"); te = _read(DATA / f"test_{fd}.txt")
    rows = []
    for s in SENSORS14:
        p = psi(tr[s].values, te[s].values)
        ks = ks_2samp(tr[s].values, te[s].values)
        rows.append({"sensor": s, "psi": round(p, 4),
                     "ks_stat": round(float(ks.statistic), 4), "ks_p": round(float(ks.pvalue), 4),
                     "level": "alert" if p > 0.25 else ("watch" if p > 0.1 else "stable")})
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    share = float((df["psi"] > 0.25).mean())
    summary = {"fd": fd, "n_sensors": len(rows),
               "dataset_drift": share >= 0.5, "drift_share": round(share, 3),
               "alert_sensors": df[df.level == "alert"]["sensor"].tolist(),
               "sensors": df.to_dict("records")}
    json.dump(summary, open(OUT / f"drift_{fd}.json", "w"), ensure_ascii=False, indent=2)
    print(f"[drift {fd}] drift_share={share:.2f} dataset_drift={summary['dataset_drift']} "
          f"top={df.iloc[0]['sensor']}(PSI {df.iloc[0]['psi']})", flush=True)
    return summary


if __name__ == "__main__":
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        run(fd)
    print("drift done -> experiments/mlops/drift_*.json", flush=True)
