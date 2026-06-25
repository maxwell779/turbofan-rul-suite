"""파운데이션 LoRA 파인튜닝 RUL — frozen MOMENT 백본에 LoRA 어댑터 + 회귀헤드.

baseline(frozen+head)과 달리 백본을 LoRA로 적응시켜 성능 향상 시도. 백본 가중치는
동결, LoRA 저랭크 행렬 + head만 학습(소비자GPU 친화). full vs few-shot10% 비교.
출처: MOMENT(ICML24) fine-tune, LoRA(Hu 2021), peft.

사용: python -m src.foundation_lora [--fd FD001 --epochs 15 --max_train 4000]
"""
from __future__ import annotations
import argparse, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from peft import LoraConfig, get_peft_model
from . import data as D
from .metrics import both
from .config import EXP, SUBSETS
warnings.filterwarnings("ignore")
OUT = EXP / "foundation"; OUT.mkdir(parents=True, exist_ok=True)


def to_moment(X):
    """(N,W,F) → (N,F,512) 보간 + mask(ones)."""
    x = torch.tensor(X).transpose(1, 2)
    x = F.interpolate(x, size=512, mode="linear", align_corners=False)
    return x


class MomentRUL(nn.Module):
    def __init__(self, moment, emb=512, p=0.2):
        super().__init__()
        self.moment = moment
        self.head = nn.Sequential(nn.Linear(emb, 128), nn.GELU(), nn.Dropout(p), nn.Linear(128, 1))

    def forward(self, x, mask):
        e = self.moment(x_enc=x, input_mask=mask).embeddings
        return self.head(e.float()).squeeze(-1)


def run(fd="FD001", epochs=15, max_train=4000, lr=1e-3, seed=42, smoke=False):
    from momentfm import MOMENTPipeline
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = D.load(fd)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(d["Xtr"]), min(max_train, len(d["Xtr"])), replace=False)
    Xtr = to_moment(d["Xtr"][idx]); ytr = torch.tensor(d["ytr"][idx])
    Xte = to_moment(d["Xte"]); yte = d["yte"]
    if smoke:
        Xtr, ytr, epochs = Xtr[:256], ytr[:256], 2

    res = {"fd": fd, "backbone": "MOMENT-1-small + LoRA(r8)", "results": {}}
    for label, frac in ([("full", 1.0)] if smoke else [("full", 1.0), ("fewshot10", 0.1)]):
        n = max(64, int(len(Xtr) * frac)); sidx = rng.choice(len(Xtr), n, replace=False)
        moment = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"})
        moment.init()
        lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q", "v"], lora_dropout=0.1, bias="none")
        moment = get_peft_model(moment, lora)
        net = MomentRUL(moment).to(dev)
        trainable = [p for p in net.parameters() if p.requires_grad]
        nt = sum(p.numel() for p in trainable); ntot = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        dl = DataLoader(TensorDataset(Xtr[sidx], ytr[sidx]), batch_size=32, shuffle=True)
        for ep in range(epochs):
            net.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev); mask = torch.ones(xb.size(0), 512, device=dev)
                opt.zero_grad(); loss = F.mse_loss(net(xb, mask), yb); loss.backward(); opt.step()
        net.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(Xte), 64):
                xb = Xte[i:i + 64].to(dev); mask = torch.ones(xb.size(0), 512, device=dev)
                preds.append(net(xb, mask).cpu().numpy())
        pred = np.clip(np.concatenate(preds), 0, None); tm = both(pred, yte)
        res["results"][label] = {"n_labels": int(n), "rmse": round(tm["rmse"], 3), "nasa": round(tm["nasa_score"], 1),
                                 "trainable_params": int(nt), "total_params": int(ntot)}
        print(f"[lora {fd}] {label:9s} n={n:5d} test_rmse={tm['rmse']:.2f} nasa={tm['nasa_score']:.0f} "
              f"(LoRA학습 {nt:,}/{ntot:,})", flush=True)
        del net, moment; torch.cuda.empty_cache()
    json.dump(res, open(OUT / f"lora_{fd}.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fd", default="all"); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max_train", type=int, default=4000); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    fds = SUBSETS if a.fd == "all" else [a.fd]
    summ = {}
    for fd in fds:
        try:
            summ[fd] = run(fd, a.epochs, a.max_train, smoke=a.smoke)["results"]
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[lora {fd}] ERR {str(e)[:100]}", flush=True)
    json.dump(summ, open(OUT / "lora_summary.json", "w"), indent=2)
    print("lora done -> experiments/foundation/lora_*.json", flush=True)
