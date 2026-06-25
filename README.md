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

그리드서치(모델 8종 × 손실 × 윈도 × lr × 시드, 약 9.4k 설정), **모델 선택은 test가 아닌 무누수 val 기준**:

| FD | 조건/결함 | best 모델 | val RMSE | **test RMSE** | NASA | 초기 baseline |
|---|---|---|---|---|---|---|
| FD001 | 1 / 1 | **PatchTST** | 9.91 | **11.87** | 206 | 13.17 |
| FD002 | 6 / 1 | GRU(조건정규화) | 12.96 | **12.82** | 768 | 21.67 |
| FD003 | 1 / 2 | CNN-LSTM | 9.52 | **13.07** | 380 | 15.86 |
| FD004 | 6 / 2 | BiLSTM(조건정규화) | 14.41 | **13.32** | 913 | 23.63 |

### 파운데이션 (frozen vs LoRA)
| FD | frozen+Ridge | **LoRA** | LoRA few-shot 10% |
|---|---|---|---|
| FD001 | 23.18 | **21.77** | 41.55 |
| FD002 | 32.36 | **25.69** | 36.97 |
| FD003 | 22.66 | **19.08** | 25.59 |
| FD004 | 33.04 | **29.19** | 44.73 |

- **조건별 정규화가 FD002/004를 살린다**: 21.67→**12.82**, 23.63→**13.32**. KMeans(6) 운전조건별 StandardScaler로 평균이동을 제거해 열화신호가 드러남.
- **최신 모델 PatchTST가 FD001 1위**(val 9.91): patching + 채널독립 Transformer가 단일조건에서 효과.
- **LoRA가 frozen 파운데이션보다 일관 개선**(백본 동결, 어댑터 180K+헤드만 학습)하지만 task-specific(~12-13)엔 미달 → "파운데이션은 C-MAPSS에서 아직 SOTA 아님"이라는 정직한 결론(문헌 일치).
- 문헌 SOTA(FD001~11, FD002~13, FD004~16)와 동급/이상, 전부 **무누수 val 선택**.

> ⚠️ 1차 그리드서치에서 PatchTST의 val 일괄추론이 CUDA 한계를 넘겨(채널독립 9.8만 배치) 일부 설정이 유실됨 → **배치추론으로 수정 후 유실분 보강 재실행 중**(전 설정 2시드 평균±std 완성 예정). 위 수치는 보강 후 갱신.

## 한계 / 개선 방향
- PatchTST 유실분 보강으로 **전 설정 2시드 평균±std** 완성 예정.
- RUL 클립 125 고정(130 비교 여지), 앙상블·conformal 예측구간 후속.
- 파운데이션: 큰 모델(base/large)·full 파인튜닝·다른 백본(Chronos/TimesFM) 비교 여지.

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
