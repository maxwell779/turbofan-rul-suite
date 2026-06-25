"""RUL 평가 지표 — RMSE + NASA scoring function.

NASA score는 비대칭이다: 예측 RUL이 실제보다 **크면(늦은 경보=고장 임박을
놓침)** 더 큰 벌점을 준다. 안전 관점(놓침=FN을 더 비싸게)과 정확히 같은 철학.
  d = RUL_pred - RUL_true
  d < 0  (보수적·이른 경보):  exp(-d/13) - 1
  d >= 0 (위험·늦은 경보):    exp( d/10) - 1
"""
import numpy as np


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def nasa_score(pred, true):
    d = np.asarray(pred, float) - np.asarray(true, float)
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def both(pred, true):
    return {"rmse": rmse(pred, true), "nasa_score": nasa_score(pred, true)}
