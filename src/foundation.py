"""시계열 파운데이션 모델 few-shot RUL — frozen MOMENT 임베딩 + 경량 head.

연구상 C-MAPSS에 파운데이션 모델 적용은 거의 미개척(few-shot 일부뿐). zero-shot은
불가(예측/복원 사전학습이라 RUL 타깃 없음) → frozen 임베딩 + Ridge/SVR head가 정석.
윈도(30,14) → 채널별 512로 보간 → MOMENT 임베딩(512d) → head. full vs few-shot 10% 비교.
출처: MOMENT(ICML24), TSFM-for-RUL(arXiv 2606.11990), few-shot RUL(CMES 2025).

사용: python -m src.foundation [--fd FD001 --max_train 3000]
"""
from __future__ import annotations
import argparse, json, warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from . import data as D
from .metrics import both
from .config import EXP
warnings.filterwarnings("ignore")
OUT = EXP / "foundation"; OUT.mkdir(parents=True, exist_ok=True)


def embed(model, X, dev, bs=64):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs]).transpose(1, 2)             # (b,14,30)
        xb = F.interpolate(xb, size=512, mode="linear", align_corners=False).to(dev)
        mask = torch.ones(xb.size(0), 512, device=dev)
        with torch.no_grad():
            e = model(x_enc=xb, input_mask=mask).embeddings
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)


def run(fd="FD001", max_train=3000, seed=42):
    from momentfm import MOMENTPipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"})
    m.init(); m.to(dev).eval()
    d = D.load(fd)
    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(len(d["Xtr"]), min(max_train, len(d["Xtr"])), replace=False)
    Etr = embed(m, d["Xtr"][tr_idx], dev); ytr = d["ytr"][tr_idx]
    Ete = embed(m, d["Xte"], dev)
    print(f"[foundation {fd}] MOMENT emb: train {Etr.shape} test {Ete.shape}", flush=True)

    res = {"fd": fd, "backbone": "MOMENT-1-small(frozen)", "emb_dim": int(Etr.shape[1]),
           "n_train_emb": int(len(Etr)), "results": {}}
    for label, frac in [("full", 1.0), ("fewshot10", 0.1)]:
        n = max(20, int(len(Etr) * frac)); idx = rng.choice(len(Etr), n, replace=False)
        for hname, head in [("ridge", Ridge(alpha=10.0)), ("svr", SVR(C=10, gamma="scale"))]:
            head.fit(Etr[idx], ytr[idx])
            pred = np.clip(head.predict(Ete), 0, None)
            tm = both(pred, d["yte"])
            res["results"][f"{label}_{hname}"] = {"n_labels": int(n), "rmse": round(tm["rmse"], 3), "nasa": round(tm["nasa_score"], 1)}
            print(f"  {label:9s} {hname:5s} n={n:5d} test_rmse={tm['rmse']:.2f} nasa={tm['nasa_score']:.0f}", flush=True)
    json.dump(res, open(OUT / f"{fd}.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    from .config import SUBSETS
    ap = argparse.ArgumentParser(); ap.add_argument("--fd", default="all"); ap.add_argument("--max_train", type=int, default=3000)
    a = ap.parse_args()
    fds = SUBSETS if a.fd == "all" else [a.fd]
    summary = {}
    for fd in fds:
        try:
            summary[fd] = run(fd, a.max_train)["results"]
        except Exception as e:
            print(f"[foundation {fd}] ERR {str(e)[:80]}", flush=True)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print("foundation done -> experiments/foundation/", flush=True)
