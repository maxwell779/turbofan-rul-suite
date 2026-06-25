# Turbofan RUL — 예지보전 잔여수명 예측 (NASA C-MAPSS)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-console-009688?logo=fastapi&logoColor=white)
![Best RMSE](https://img.shields.io/badge/FD001%20RMSE-13.17-3b82f6)
![License](https://img.shields.io/badge/License-MIT-64748b)

터보팬 엔진 센서 시계열로 **잔여수명(RUL, Remaining Useful Life)** 을 예측하는 예지보전(PdM) 프로젝트. 정적 이미지 결함 검사를 넘어 **시간축 고장 예측**으로 확장했고, 산업 안전 관점(고장을 늦게 알수록 치명적)을 평가지표에 반영했다.

데이터: [NASA C-MAPSS Turbofan Degradation](https://www.nasa.gov/intelligent-systems-division/) · 4개 subset(FD001~004) · 운전조건 1~6종 · 결함모드 1~2종.

## 문제 정의
엔진은 정상 가동하다 특정 시점부터 열화가 진행돼 고장에 이른다. 매 사이클의 21개 센서 값으로 "앞으로 몇 사이클 더 쓸 수 있는가(RUL)"를 회귀로 예측한다.
- **안전 비대칭**: RUL을 실제보다 **크게** 예측하면(=고장 임박을 놓침) 정비 시점을 놓쳐 사고로 직결된다. 그래서 RMSE와 함께 **NASA scoring function**(늦은 예측에 더 큰 벌점)을 쓴다 — 비전 프로젝트에서 "놓침(FN)을 더 비싸게" 본 것과 같은 철학.
- **piecewise RUL**: 초기 정상 구간은 RUL을 125로 클립(열화 신호가 없는 구간에 큰 라벨을 주지 않음, 표준 관행).

## 방법
- **입력**: 30사이클 슬라이딩 윈도 × 유효 센서(train 분산 0 컬럼 자동 제거). 스케일러는 train 엔진으로만 적합.
- **모델**: 1D-CNN, LSTM 두 가지를 동일 기준으로 비교.
- **무누수 검증(핵심)**: train/val 분할을 **엔진 unit 단위**로만 한다. 같은 엔진의 윈도가 train·val에 동시에 들어가면 시계열 자기상관으로 점수가 부풀려진다 — 버스바 프로젝트의 "부품 단위 무누수"와 동일하게 통제.
- **평가**: test 엔진별 마지막 윈도로 RUL 예측 → 제공된 정답 RUL과 비교(RMSE + NASA score).

## 결과 (test)

| FD | 운전조건/결함모드 | 모델 | val RMSE(무누수) | test RMSE | NASA score |
|---|---|---|---|---|---|
| FD001 | 1 / 1 | **LSTM** | 13.50 | **13.17** | 310 |
| FD001 | 1 / 1 | CNN | 18.81 | 19.75 | 1,305 |
| FD002 | 6 / 1 | **CNN** | 22.24 | **21.67** | 2,771 |
| FD002 | 6 / 1 | LSTM | 41.76 | 43.43 | 60,482 |
| FD003 | 1 / 2 | **LSTM** | 11.72 | **15.86** | 1,170 |
| FD004 | 6 / 2 | **CNN** | 23.92 | **23.63** | 10,127 |

- **val ≈ test** (FD001/002/004)로 무누수 분할의 일반화가 확인된다(엔진단위 hold-out이 우연이 아님).
- **조건 수가 모델 선택을 가른다**: 단일 운전조건(FD001·FD003)은 **LSTM**이, 다중(6) 운전조건(FD002·FD004)은 **CNN**이 우세하다. LSTM은 6개 운전조건이 섞이면 RMSE가 급격히 악화(43~44)된다.

## 한계 / 개선 방향
- **FD002·FD004(6 운전조건)**: 전역 정규화로는 운전조건별 분포 차이를 못 흡수해 성능이 떨어진다. → **운전조건 클러스터별 정규화(condition-based normalization)** 가 다음 레버(문헌상 RMSE를 크게 낮춤).
- RUL 클립(125)·윈도 길이(30)는 표준값 고정 — 튜닝 여지 있음.
- 단일 모델 기준선이며 앙상블·attention은 후속.

## 데모 — RUL Console ([web/](web/))
FastAPI + 무빌드 SPA(라이트 테마, 외부 차트 라이브러리 없이 SVG). 리더보드, **엔진 RUL 뷰어**(사이클별 예측 RUL이 실제 고장 시점으로 수렴하는 과정 + 센서 곡선), 학습 곡선, 예측-실제 산점도, 결과표.

![RUL Console](docs/images/console.png)

```bash
bash scripts/download_data.sh          # C-MAPSS 12개 txt
python -m src.train --fd FD001 --model lstm --epochs 60
python -m src.eval                     # 전체 평가 → experiments/leaderboard.csv
python -m uvicorn web.server:app --port 8020   # http://127.0.0.1:8020
# 전체 재현: bash scripts/run_all.sh    (FD001~004 × {cnn,lstm})
```

## 구조
```
src/   data.py(무누수 로더·윈도) · model.py(CNN/LSTM) · train.py · eval.py · metrics.py(RMSE+NASA)
web/   server.py(FastAPI) · static/(콘솔)
scripts/ download_data.sh · run_all.sh
experiments/ <fd>_<model>/(history·val/test_metrics·test_pred) · leaderboard.csv
```
