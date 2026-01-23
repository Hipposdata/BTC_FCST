# 🪙 TOBIT: AI-Driven Bitcoin Investment Analysis Platform

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

> **ToBigs Conference TSF 2025 Project**
>
> "단순한 예측(Forecast)을 넘어, 설명(Explain)하고 시뮬레이션(Simulate)합니다."

**TOBIT**은 최신 시계열 모델(PatchTST, iTransformer 등)과 강력한 **XAI(설명 가능한 AI)** 파이프라인을 결합한 비트코인 분석 플랫폼입니다.
단순히 미래 가격을 예측하는 것을 넘어, **TimeSHAP**을 통해 예측의 근거를 설명하고, **Counterfactual Simulator**를 통해 시장 변수 변화에 따른 시나리오를 검증합니다.

---

## 📺 Preview

### Demo Video
[![Video Label](http://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://youtu.be/YOUR_VIDEO_ID)

### Screenshots

| **Market Forecast** | **Deep Insight (XAI)** |
|:---:|:---:|
| <img src="assets/demo_forecast.png" alt="Forecast" width="100%"> | <img src="assets/demo_xai.png" alt="XAI" width="100%"> |
| 실시간 가격 예측 및 경제 지표 대시보드 | TimeSHAP 히트맵 및 Counterfactual 시뮬레이션 |

| **Model Specs** | **Strategy Backtest** |
|:---:|:---:|
| <img src="assets/demo_specs.png" alt="Specs" width="100%"> | <img src="assets/demo_backtest.png" alt="Backtest" width="100%"> |
| SOTA 모델 아키텍처 다이어그램 | 매수/매도 시뮬레이션 및 수익률 검증 |

---

## 💡 Key Features

### 1. Dual-Engine XAI (Explainable AI)
TOBIT은 두 가지 관점에서 모델을 해석합니다.
* **TimeSHAP (Post-hoc Interpretability):** "모델이 **왜** 그런 예측을 했는가?"
    * **Event-Level:** 과거 14~45일 중 예측에 결정적이었던 특정 시점을 포착합니다.
    * **Feature-Level:** 거래량, 금리, 심리지수 중 어떤 변수가 가격 변동을 주도했는지 분석합니다.
    * **Pruning:** 예측에 불필요한 과거 데이터를 가지치기(Pruning)하여 보여줍니다.
* **Counterfactual Simulator (What-If Analysis):** "만약 **변수**가 달라진다면 결과는?"
    * *"만약 오늘의 비트코인 거래량이 20% 급증한다면, 7일 뒤 가격은 어떻게 될까?"*
    * 특정 변수(Feature)의 수치를 조작하여 모델의 민감도(Sensitivity)를 실시간으로 테스트합니다.
    * LLM Analyst와 연동하여 시나리오별 리스크 관리 전략을 제안받습니다.

### 2. SOTA Forecasting Models
* **Transformer-based:** **PatchTST**, **iTransformer** (장기 시계열 및 다변량 상관관계 학습 최적화)
* **NN-based:** **DLinear** (추세/계절성 분해), **TCN** (Dilated Conv), **LSTM**

### 3. Automated Strategy Pipeline
* **Daily Discord Bot:** 매일 아침 시장 데이터를 수집하고 추론(Inference)을 수행하여 리포트를 발송합니다.
* **Signal System:** 예측 수익률이 설정된 임계값(Threshold, 예: ±5%)을 초과할 때만 `STRONG BUY/SELL` 시그널을 생성합니다.

---

## 🛠 System Architecture

데이터 파이프라인은 크게 **자동화된 알림 시스템(Discord)**과 **사용자 대시보드(XAI & LLM)** 두 가지 경로로 나뉩니다.

```mermaid
graph TD
    %% Data Flow
    A[APIs: YFinance / FRED / Alternative.me] --> B(Data Preprocessing)
    B -->|Scaling & Sequence| C{Model Inference}
    
    %% Automated Path
    subgraph "🤖 Automated Pipeline"
    C -->|Daily Cron| D[Daily Bot]
    D -->|Threshold Check| E[Discord Webhook]
    E --> F[User Alert]
    end
    
    %% Dashboard Path
    subgraph "🧠 Analytic Dashboard"
    C --> G[XAI Engine]
    G -->|TimeSHAP / Simulation| H[Upstage Solar API]
    H -->|Natural Language Report| I[User Insight]
    end
