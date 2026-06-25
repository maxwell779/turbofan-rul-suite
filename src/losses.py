"""RUL 손실함수 — MSE, Huber, 비대칭(안전, 늦은예측 가중), Quantile(pinball).

비대칭: 늦은 예측(pred>true, RUL 과대=고장 임박 놓침)에 더 큰 벌점 → NASA score와
같은 안전 철학을 미분가능하게. 지수형은 초기에 폭발하므로 '가중 MSE'로 안정화
(late_w>early_w). 출처: Hahn&Mechefske 동적가중손실(PMC7038523).
"""
import torch
import torch.nn.functional as F


def mse(pred, target):
    return F.mse_loss(pred, target)


def huber(pred, target, delta=1.0):
    return F.huber_loss(pred, target, delta=delta)


def asymmetric(pred, target, late_w=1.5, early_w=1.0):
    d = pred - target                      # >0: 늦음(위험)
    w = torch.where(d >= 0, late_w, early_w)
    return (w * d * d).mean()


def pinball(pred_q, target, quantiles=(0.1, 0.5, 0.9)):
    """pred_q: (B, n_q). 예측구간 학습용."""
    losses = []
    for i, q in enumerate(quantiles):
        e = target - pred_q[:, i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.stack(losses, 1).mean()


def get(name):
    return {"mse": mse, "huber": huber, "asym": asymmetric}[name]
