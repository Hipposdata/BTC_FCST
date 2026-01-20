# ⚡ QUANTUM BIT: AI Crypto Intelligence

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/MLOps-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Quantum Bit**는 최신 딥러닝 시계열 모델(SOTA Models)과 설명 가능한 AI(XAI) 기술을 결합하여 비트코인(BTC)의 가격 흐름을 예측하고 분석하는 **AI 기반 투자 분석 플랫폼**입니다.

GitHub Actions를 통한 **MLOps 파이프라인**이 구축되어 있어, 매일 아침 자동으로 최신 데이터를 학습하고 예측 리포트를 Discord로 발송합니다.

---

## 📸 Dashboard Preview
*(여기에 실행 화면 스크린샷을 추가하세요. 예: `![Dashboard](assets/dashboard.png)`)*

---

## 🌟 Key Features (주요 기능)

### 1. 🤖 Advanced AI Models
다양한 아키텍처를 활용하여 시장을 다각도로 분석합니다.
* **Transformer-based:** `PatchTST`, `iTransformer` (SOTA Performance)
* **CNN/RNN-based:** `TCN`, `LSTM`
* **Efficient:** `DLinear`, `MLP`

### 2. 🔄 MLOps & Automation
* **Daily Retraining:** 매일 UTC 00:00 (KST 09:00)에 GitHub Actions가 최신 데이터를 수집하여 모델을 **전체 재학습(Full Retraining)**합니다.
* **Auto Deployment:** 학습된 가중치(`weights/*.pth`)는 자동으로 레포지토리에 커밋되어 Streamlit 앱에 즉시 반영됩니다.
* **Alert System:** 학습 완료 및 예측 결과를 **Discord** 알림으로 실시간 전송합니다.

### 3. 🧠 Explainable AI (XAI)
단순한 예측값을 넘어 "왜" 그런 예측이 나왔는지 설명합니다.
* **Saliency Map:** 예측에 가장 큰 영향을 준 변수(Feature)와 시점(Time Step)을 히트맵으로 시각화.
* **TimeSHAP:** 과거 특정 시점의 데이터가 예측에 미친 기여도를 정량적으로 분석.

### 4. 📊 Professional Dashboard
* Bloomberg Terminal 스타일의 **Dark & Neon UI**.
* 실시간 시장 데이터(Binance, FRED, Yahoo Finance) 연동.
* Interactive Plotly 차트 및 지표 카드 제공.

---

## 🛠️ System Architecture

```mermaid
graph LR
    A[Data Sources] -->|API Fetch| B(Data Preprocessing)
    B -->|Scaling & Sequencing| C{Model Training}
    C -->|MLP, LSTM, TCN...| D[Weights File (.pth)]
    D -->|Git Push| E[GitHub Repo]
    E -->|Auto Load| F[Streamlit Dashboard]
    C -->|Predict| G[Discord Bot]
    
    subgraph "GitHub Actions (Daily)"
    B
    C
    D
    G
    end
