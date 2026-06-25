# PRD — Turbofan RUL 예지보전 고도화 (포트폴리오용)

> 목표: 현재 baseline(FD001 LSTM RMSE 13.17, CNN/LSTM)을 **SOTA급·운영급**으로 끌어올리고, 본인 시그니처(무누수 검증·안전지표·한계 규명)를 시계열에서 증명하며, **진짜 예지보전 콘솔 + MLOps 깊이**까지 갖춘다. 데이터: NASA C-MAPSS FD001~004.
> 작성 2026-06-25 · 웹 조사 기반(출처는 §부록).

---

## 0. 포지셔닝 — 왜 이 프로젝트인가
- 기존 포트폴리오: 비전(이상탐지·검출·세그) 4 + LLM 2 + 풀스택 2 + ML/추천. **시계열 = 0** → 제조 채용의 핵심 공백.
- 차별화 축 3개: ① **무누수·정직 검증**(엔진단위 split, val≈test 증명) ② **SOTA 모델 동물원 + 대규모 그리드서치**(A100) ③ **운영급 PdM**(드리프트·XAI·ONNX·헬스인덱스 콘솔). 단순 "RMSE 한 줄"이 아니라 **데이터→모델→배포→모니터링 전 주기**를 보여준다.

## 1. 성공 기준 (측정 가능)
- FD001 test RMSE **≤ 12.5** (현재 13.17), FD002/004 **조건별 정규화로 RMSE 30%+ 개선**(현재 21.7/23.6 → 목표 15~17/18~20).
- 모델 **10종+ 동일 무누수 기준 비교표**(ML 4 + DL 6+), 손실함수 3종 비교, 그리드서치 수백 config.
- **불확실성**(예측구간) + **NASA score**(안전 비대칭) 동시 보고.
- ONNX 추론 + **지연 p50/p95/p99**(CPU/GPU) + int8 양자화 비교.
- **드리프트 모니터**(PSI/KS) + **XAI**(센서 기여도·시간축 saliency).
- React 콘솔(App 디자인): 함대 헬스·RUL+신뢰구간·경보 리드타임·드리프트·XAI 패널.

---

## 2. 데이터 EDA 고도화 (Phase 1)

### 2-1. 시계열 EDA는 어떻게? (조사 종합)
표준 절차 → C-MAPSS 적용:
1. **구조/결측/이상치**: 균일 그리드, 결측(ffill/보간), 이상치(이동 Z-score |z|>3, IQR). C-MAPSS는 결측 없음 → 분산·이상 위주.
2. **분산 필터**: 상수/근사상수 센서 제거 — caret 기준(freqCut 19, uniqueCut 10). C-MAPSS 표준 제거 7개 `s1,s5,s6,s10,s16,s18,s19` → **14센서** `[2,3,4,7,8,9,11,12,13,14,15,17,20,21]`.
3. **추세·정상성**: STL 분해, **ADF**(H0 비정상, p<0.05면 정상) + **KPSS**(반대 가설) 병행. 열화 신호라 원신호는 비정상이 정상적 — 차분/정규화 동기.
4. **자기상관**: ACF/PACF로 윈도 길이 감.
5. **다중공선성**: 센서-센서 상관 heatmap + **VIF**(>10 심각) → 중복 센서 식별(상관 0.96까지).
6. **PHM 특징선택(핵심)**: Coble–Hines 3지표 — **monotonicity**(단조성, 0~1), **trendability**(엔진간 궤적 유사도=최소 쌍상관), **prognosability**(고장값 산포 작을수록↑). 센서 랭킹·HI 품질 기준.
7. **운전조건 군집**: op-setting 3개(고도·Mach·TRA)로 **KMeans(6)** → FD002/004는 조건별 분포가 분리됨을 t-SNE/scatter로 시각화 → **조건별 정규화 정당화**.
8. **change-point/FPT**: `ruptures`(PELT/Binseg) 또는 CUSUM으로 열화 시작점(First Predicting Time) 탐지.
9. **smoothing**: EWMA / Savitzky–Golay(피크·곡률 보존) — 옵션, 효과 시각화.

### 2-2. 비전(버스바)은 EDA를 어떻게? (질문 답)
이미지 이상탐지 EDA는 시계열과 다르다:
- **클래스/조건 분포**: 양품/불량 수, 카메라·조명·zone별 분포(불균형 파악).
- **밝기/대비/색 히스토그램**(조건별) → 조명 편차·전처리 필요성(버스바에서 DoG 채택 근거가 여기).
- **결함 위치/크기 분포**(마스크 있으면), 결함 타입별 샘플 몽타주.
- **임베딩 EDA**: DINOv2 특징 PCA/t-SNE로 양품 군집 vs 이상 분리도(이상탐지 가능성 진단), patch homogeneity·FFT 고주파·hole contrast 같은 "이상탐지 친화도" 지표.
- **라벨 노이즈 점검**: 정상 오분류 전수 검토(버스바에서 한 것). → 별도 노트북 `eda_vision.ipynb`로 버스바/steel에 1장 추가하면 비전 EDA 역량도 보강.

### 2-3. Phase1 산출물 (일부 완료)
`src/eda.py` 완료: 수명분포·열화곡선·센서랭킹(mono/corr)·상관·운전조건군집·정규화효과 17개 그림 + `experiments/eda/summary.json`. **TODO**: 라벨 영문화(폰트), ADF/KPSS·VIF·ACF·prognosability 표를 콘솔/리포트에 노출, ruptures FPT 추가.

---

## 3. 전처리 업그레이드 (Phase 2)
- **14센서 선택**(상수 제거) — 현 자동 분산필터를 표준 14센서로 고정 옵션 추가.
- **조건별 정규화**(FD002/004 핵심): KMeans(6) on op-settings → **조건별 StandardScaler**(train만 적합, 누수차단). 단일조건(FD001/003)은 전역 z-score. → FD002/004 대폭 개선 기대.
- **piecewise RUL** cap: 125(현)·130 비교(Heimes 2008).
- **윈도** 30 고정 + 길이 sweep(20/30/40/50).
- **smoothing** 옵션(EWMA α, SG window/poly) — 그리드 변수.

---

## 4. 모델 동물원 + 손실함수 + 그리드서치 (Phase 3, A100)

### 4-1. 머신러닝 baseline (특징기반)
윈도 → 통계특징(평균·표준편차·기울기·last·min·max, 또는 tsfresh 축약) → **Ridge, RandomForest, XGBoost/LightGBM, SVR**. "DL이 정말 이기나?"를 정량화(DLinear 논쟁과 연결).

### 4-2. 딥러닝 동물원
- **CNN1D**(현) — 그리드: kernel{3,5,7}, 채널{32,64,128}, 깊이, padding{same}, stride{1,2}, **pooling{avg,max,adaptive}**, dilation, activation{ReLU,GELU}, dropout, BN, weight decay, init.
- **LSTM/GRU/BiLSTM** — hidden{64,96,128}, layers{1,2,3}, dropout, bidirectional.
- **CNN-LSTM 하이브리드**(conv 특징→순환).
- **TCN**(dilated causal conv) — receptive field 큼.
- **Transformer / PatchTST 스타일** — patching + channel-independence + RevIN. (조사: 고정윈도→스칼라엔 patching이 유효)
- **DLinear/NLinear** — 선형 강baseline(공정 비교용).
- (선택) **iTransformer/Crossformer식 cross-variate attention** — 센서간 의존.
- (선택) 시계열 **파운데이션 모델** Chronos/MOMENT fine-tune — "최신" 어필.

### 4-3. 손실함수
- **MSE**(기본), **Huber**(이상치 강건), **Asymmetric(NASA-aligned)** — 늦은 예측(과대 RUL)에 큰 벌점 직접 최적화(안전), **Quantile/Pinball** — 예측구간(불확실성).

### 4-4. 그리드서치 인프라 (A100)
- `src/sweep.py` — config dict 레지스트리(모델×하이퍼×손실×윈도×정규화), 멀티프로세스, gpu-guard 연동, 결과 CSV append, 무누수 val 기준 best, resume.
- 수백 config를 밤새. 리더보드 자동 생성 + 모델별 best 비교표.

---

## 5. 최신 시계열 기법 적용 (Phase 4)
- **RevIN**(reversible instance norm), patch embedding, channel-independence를 공통 모듈로.
- 조사 결론: LTSF에서 단순 선형도 강하니 **공정 비교**(동일 윈도·정규화·예산)가 핵심 — Transformer를 무조건 우위로 포장하지 않음(정직).
- 우선순위: PatchTST식 > TCN > iTransformer식 cross-variate > 파운데이션 fine-tune.

---

## 6. MLOps 깊이 (Phase 5)
- **ONNX export**: `torch.onnx.export(dynamo=True)`, dynamic batch/seq, opset 매칭. LSTM은 batch=1 또는 h0/c0 노출 주의.
- **지연 벤치마크**: onnxruntime CPU/GPU(CUDA EP+IOBinding), warmup≥20, **p50/p95/p99** 보고. PyTorch vs ONNX vs int8.
- **양자화**: LSTM/Transformer→`quantize_dynamic`(int8 ~2× CPU), CNN/TCN→`quantize_static`. 정확도-속도 트레이드오프 표.
- **드리프트**: **PSI**(<0.1 안정/0.1~0.25 주의/>0.25 경보), **KS**(대표본 과민 주의), 스트리밍 **ADWIN/Page-Hinkley**(river), 또는 Evidently 리포트. 센서별 모니터 + 재학습 트리거. *주의: 진짜 열화 vs 센서 드리프트 구분 한계 명시.*
- **XAI**: **SHAP DeepExplainer**(배경 100~1000) 또는 **Captum IntegratedGradients**(zero baseline, completeness 체크) → (시간×센서) 기여도 → 시간축 집계로 **센서 중요도**, 부호로 "RUL을 끌어내리는 센서". attention은 보조(‘설명 아님’ 논쟁 명시, gradient와 교차검증).

## 7. 프론트엔드 / 콘솔 (Phase 6)
- **언어/스택 권고: React + Vite (+ TypeScript)**. 이유: 풍부한 인터랙션(드릴다운·실시간·다패널)이 필요한 운영 대시보드엔 Streamlit보다 React가 적합하고, **이미 pro-vision `App/` 디자인 시스템**(라이트테마·SVG 차트·운영/분석 2모드)을 steel-console에서 재사용해봤다 → 그대로 차용. 차트는 외부 라이브러리 없이 SVG/CSS(일관성).
- **App 폴더 참고**: 코드·UI/UX(토큰·카드·KPI·도넛/막대/관리도) 그대로 디자인 언어로.
- **진짜 예지보전 기능**:
  1. **함대(Fleet) 헬스 보드** — 엔진별 헬스인덱스·RUL·상태등급(정상/주의/위험) 정렬.
  2. **엔진 상세** — 사이클별 예측 RUL + **신뢰구간(quantile)** + 실제선 + **경보 임계(정비 리드타임)**.
  3. **센서 패널** — 14센서 곡선 + 이상 하이라이트 + change-point(FPT) 표시.
  4. **XAI 패널** — 이 예측을 끌어내린 센서 기여도(SHAP/IG) 막대 + 시간축 saliency 히트맵.
  5. **드리프트 모니터** — 센서 분포 PSI/KS 추세, 경보, 재학습 권고.
  6. **모델 리더보드/비교** — 동물원 전 모델, 손실함수, 그리드 결과.
  7. **What-if** — 윈도/임계 슬라이더로 경보 시점 변화.
  8. **운영/분석 2모드**(App식).

## 8. 포트폴리오 활용 (Phase 7)
- **PDF 카드 1장**(비전제조 PDF에 Hero로 추가) + Notion + GitHub README(배지·콘솔 GIF·결과표·한계).
- 면접 스토리: "정적 결함검사 → 시간축 고장예측으로 확장 + 무누수·안전지표·**배포/모니터링**까지." 발견(조건정규화 효과·CNN vs LSTM·DLinear 논쟁 공정비교)·한계(드리프트 vs 열화 구분)로 깊이.
- 1줄 요약 지표: "FD001 RMSE 12.x · 모델 10종 무누수 비교 · ONNX p99 지연 · 드리프트·XAI 콘솔".

---

## 9. 단계 실행 순서 (체크리스트)
- [x] P0 baseline(CNN/LSTM, 무누수, NASA score, 콘솔v1) — 완료·push됨
- [~] P1 EDA 고도화 — eda.py 완료, 라벨영문화·VIF/ADF/FPT·리포트 TODO
- [ ] P2 전처리 업그레이드(14센서·조건정규화·smoothing)
- [ ] P3 모델 동물원 + 손실 3종 + A100 그리드서치
- [ ] P4 최신기법(PatchTST/TCN/RevIN, 공정비교)
- [ ] P5 MLOps(ONNX·지연·양자화·드리프트·XAI)
- [ ] P6 React 콘솔 고도화(App 디자인, PdM 8기능)
- [ ] P7 포트폴리오 패키징(PDF/README/GIF)

## 10. SOTA 벤치마크 & 목표 (조사 확정)
공개 문헌 test RMSE(표준 last-cycle 프로토콜):

| 모델 | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| DCNN (Li 2018, 정전형 baseline) | 12.61 | 22.36 | 12.64 | 23.31 |
| BiLSTM (2018) | 13.65 | 23.18 | 13.74 | 24.86 |
| DAST (transformer 2022) | 11.43 | 15.25 | 11.32 | 18.23 |
| TTSNet (TCN+transformer 2025) | 11.02 | 13.25 | 11.06 | 18.26 |
| STAR (best 2024) | **10.61** | **13.47** | **10.71** | **15.87** |
| **현재 우리(baseline)** | 13.17 | 21.67 | 15.86 | 23.63 |
| **목표** | ≤12.0 | **≤16** | ≤13 | **≤19** |

- FD001 SOTA는 ~11에서 포화(10.97). **<10.5 주장은 비표준 프로토콜 의심** — 우리는 표준 last-cycle 유지(정직).
- 최대 레버: **FD002/004 조건별 정규화**(현재 우리 LSTM 43/44 → 목표 16/19). + 윈도 길이 난이도별(FD001=30/FD002=60/FD003=40/FD004=50).
- **계산은 병목 아님**(모델 수만~수십만 파라미터, 학습 ~수십초) → A100으로 수백 config 그리드·다중 seed 평균 가능.

### 차별화(novelty) 포인트
- **시계열 파운데이션 모델(Chronos/MOMENT/TimesFM) × C-MAPSS RUL**은 공개 벤치마크가 거의 없음(few-shot 일부뿐). → frozen 임베딩 + 경량 head로 **few-shot RUL** 비교 = 포트폴리오 신선도.
- **PatchTST/N-BEATS의 C-MAPSS 적용**도 표준화된 결과 부재 → 공정 비교가 기여.
- 단, 정직 프레이밍: "최신=무조건 우위" 아님. DLinear 논쟁대로 **동일 윈도·정규화·예산 공정비교**로 검증.

## 부록 — 핵심 출처(요약)
- 조건정규화/6군집: TowardsDataScience LSTM-PdM, arXiv 2604.27234, 2603.00745, NASA PMC10459474.
- 14센서·piecewise RUL(Heimes 2008)·윈도30: MDPI 13/21/11893, arXiv 2604.27234.
- PHM 지표(mono/trend/progn): Coble&Hines 2009, MathWorks predmaint.
- 최신 TS: PatchTST(ICLR23), iTransformer(ICLR24), DLinear(AAAI23), Informer/Autoformer/FEDformer.
- 드리프트: PSI(fiddler), KS(deepchecks), river ADWIN/DDM, Evidently.
- ONNX/양자화: pytorch onnx docs, onnxruntime quantization.
- XAI: SHAP Deep/Gradient, Captum IG, "Attention is not Explanation"(NAACL19).
