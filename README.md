# 🏦 포르투갈 은행 정기예금 가입 예측

> 정기예금 가입 여부를 예측하는 이진 분류 모델 구축
> DA12기 4팀 전미정 | Codeit Sprint Mission 13

---

## 📌 프로젝트 개요

포르투갈 은행의 텔레마케팅 캠페인 데이터(2008–2010)를 바탕으로, 고객이 정기예금에 가입할지 여부를 예측하는 머신러닝 모델을 구축했다. 단순한 성능 최적화를 넘어, 데이터가 생산된 시대적 맥락(글로벌 금융위기, 포르투갈 재정위기)을 함께 읽는 방식으로 분석을 진행했다.

| 항목 | 내용 |
|---|---|
| 데이터 출처 | UCI Machine Learning Repository |
| 수집 기간 | 2008년 5월 ~ 2010년 11월 |
| 데이터 규모 | 41,188건 / 21개 컬럼 |
| 클래스 불균형 | yes 11.3% : no 88.7% (약 1:8.8) |
| 평가 지표 | F1-score (yes), ROC-AUC |

> **왜 Accuracy가 아닌 F1/AUC인가?**
> 클래스 불균형 데이터에서 Accuracy는 함정이다. 모두 "가입 안 함"으로 예측해도 88% 정확도가 나온다. 실제 가입자를 얼마나 찾아냈는가(Recall)와 예측의 정밀도(Precision)를 동시에 보는 F1-score를 핵심 지표로 삼았다.

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

```
Python 3.10+  |  scikit-learn  |  xgboost  |  pandas  |  numpy  |  matplotlib  |  seaborn
```

---

## 🤖 사용 모델

| 모델 | 역할 |
|---|---|
| Decision Tree (depth=6) | Baseline |
| Random Forest (GridSearchCV 튜닝) | 앙상블 비교 |
| **XGBoost (최종 선택)** | 임계값 조정 유연성 우위 |

---

## 📊 모델 성능 비교

| 모델 | Precision | Recall | F1 (yes) | ROC-AUC | Accuracy |
|---|---|---|---|---|---|
| Decision Tree (depth=6) | 0.34 | 0.69 | 0.45 | 0.8002 | 0.81 |
| Random Forest (tuned) | 0.45 | 0.61 | 0.52 | 0.8047 | 0.87 |
| **XGBoost (tuned)** | **0.42** | **0.63** | **0.51** | **0.8161** | **0.86** |

### Duration 리키지 실험

| 구분 | Precision | Recall | F1 (yes) | ROC-AUC |
|---|---|---|---|---|
| Main 모델 (duration 제외) | 0.45 | 0.63 | 0.51 | 0.8161 |
| Full 모델 (duration 포함) | 0.47 | 0.92 | 0.62 | 0.9500 |

`duration`(통화 시간)은 상담 종료 후에야 확정되는 사후 정보로, 실전에서 사용 불가한 리키지 변수다. Full 모델의 높은 수치는 경고 신호이지 성능이 아니다.

---

## 🔧 주요 전처리 결정

| 항목 | 처리 방식 | 근거 |
|---|---|---|
| `unknown` 처리 | 독립 범주로 유지 | `isnull()`에 잡히지 않는 숨겨진 결측치. `default` 컬럼의 unknown(20.9%)은 침묵도 데이터다 |
| `pdays=999` 재가공 | 이진 변수 `pdays_contacted`로 변환 | 이상치가 아닌 "미접촉" 상태를 의미 |
| `duration` 분리 | `X_main` / `X_full` 두 데이터셋으로 분리 | 리키지 실험용 vs 실전용 명확히 구분 |
| 인코딩 전략 | 순서 없는 범주형 → One-Hot / `education` → Label Encoding | 변수의 순서 의미 반영 |

---

## 💡 핵심 인사이트

### 1. `nr.employed`가 Feature Importance 1위

개인 특성(나이, 직업)보다 **거시 경제 지표**가 가입 결정에 더 강하게 작용했다. 고용 불안이 커질수록 예금 가입이 늘어나는 **손실회피(Loss Aversion)** 패턴.

### 2. October Anomaly — 10월 이상현상

전체 평균 가입률 **11.3%** → 10월 가입률 **43.9%** (평균의 4배)
2008년 10월 **62.7%** (평균의 5.6배)

리먼 브라더스 파산(2008.9.15) 직후, 포르투갈 정부의 예금보험 한도 4배 인상과 BPN 은행 국유화가 동시에 일어난 달이다.

> 모델은 달력을 읽은 게 아니라, 사람들이 가장 두려웠던 달을 기억하고 있는 것이다.

### 3. XGBoost 최종 선택 — 수치보다 유연성

F1-score만 보면 RF(0.52) > XGBoost(0.51). 그러나 임계값 조정 실험에서 XGBoost는 0.3으로 낮추면 **Recall 0.90**까지 가능. RF는 임계값 변화에 둔감.

비즈니스 상황에 따라 **정밀 타겟 모드 / 고감도 모드** 전환이 가능한 유연성을 선택했다.

---

## ⚠️ 한계

- 2008~2010년 금융위기 국면에서만 성립하는 패턴. 정상 경기 사이클에서 역전될 수 있음
- 단일 기관, 단일 시기 데이터 — 다른 시장 환경에 무검증 적용 불가
- 디지털 전환 이전의 채널 구조(텔레마케팅)에 특화된 모델

---

## 📁 폴더 구조

```
mission13/
├── notebook/
│   └── mission13.ipynb
├── data/
│   └── bank-additional-full.csv
├── report/
│   └── mission13_final_report.pdf
└── README.md
```

---

## 📝 분석 노트

단순한 모델 성능 개선 기록이 아니라, 데이터가 생산된 시대와 사람의 이야기를 함께 읽으려 했다.

> *"아직 적히지 않은 페이지를 읽어내는 것이 우리의 일이다."* — 움베르트 에코
