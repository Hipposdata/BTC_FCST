# 🐻 ToBit: From Data to Bitcoin

<div align="center">
    <img src="assets/logo.png" alt="ToBit Logo" width="200">
    <br>
    <h3><b>ToBigs Data Science Club Project</b></h3>
    <p>AI-Driven Crypto Analysis & Auto-Trading Signal Platform</p>
</div>

---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/MLOps-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)
![Discord](https://img.shields.io/badge/Alert-Discord-5865F2?logo=discord&logoColor=white)

**ToBit(투비트)**는 **투빅스(ToBigs)** 데이터 사이언스 동아리에서 개발한 **AI 비트코인 투자 분석 플랫폼**입니다.
최신 시계열 딥러닝 모델(SOTA Models)과 설명 가능한 AI(XAI) 기술을 결합하여, 데이터에 기반한 객관적인 매매 신호를 제공합니다.

---

## 📸 Dashboard Preview
*(실제 실행 화면을 캡처해서 `assets/dashboard.png`로 저장 후 여기에 링크를 거세요)*
`<img src="assets/dashboard.png" width="100%">`

---

## 🌟 Key Features (주요 기능)

### 1. 🤖 Advanced AI Models
최신 연구 트렌드를 반영한 다양한 딥러닝 아키텍처를 비교 분석합니다.
* **Transformer-based:** `PatchTST`, `iTransformer` (Long-term Forecasting SOTA)
* **CNN/RNN-based:** `TCN`, `LSTM`
* **Efficient:** `DLinear`, `MLP`
* **Ensemble:** 다중 모델의 만장일치 및 가중 평균 예측

### 2. 🔄 MLOps & Automation (GitHub Actions)
* **Daily Retraining:** 매일 한국 시간 **오전 09:00**에 최신 데이터를 수집하여 모든 모델을 **전체 재학습(Full Retraining)**합니다.
* **Auto-Update:** 학습된 가중치(`.pth`)는 자동으로 레포지토리에 업데이트되며, 대시보드에 즉시 반영됩니다.
* **Discord Alert:** 학습 완료 후, 앙상블 예측 결과와 매수/매도 신호를 **디스코드**로 실시간 발송합니다.

### 3. 🧠 Explainable AI (XAI)
단순한 예측값을 넘어 모델의 판단 근거를 제시합니다.
* **Attention Heatmap:** 모델이 중요하게 본 변수(Feature)와 시점(Time Step) 시각화.
* **TimeSHAP:** 과거 특정 구간의 데이터가 현재 예측에 미친 영향력 분석.

---

## 🛠️ System Architecture

```mermaid
graph LR
    A[Data Sources\n(Binance, FRED)] -->|Fetch API| B(Data Preprocessing)
    B -->|Scaling & Sequencing| C{Auto Training\n(GitHub Actions)}
    C -->|PatchTST, TCN...| D[Weights File (.pth)]
    D -->|Git Push| E[GitHub Repo]
    E -->|Auto Load| F[Streamlit Dashboard\n(ToBit App)]
    C -->|Ensemble Predict| G[Discord Bot]
    
    subgraph "Daily Routine (09:00 KST)"
    B
    C
    D
    G
    end
