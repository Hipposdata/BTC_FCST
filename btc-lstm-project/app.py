import sys
import os
from pathlib import Path

# --- 1. Python 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN
from data_utils import fetch_multi_data, load_scaler, TICKERS

# 1. 초기 설정
st.set_page_config(page_title="BTC XAI Research Lab", layout="wide")
WEIGHTS_DIR, MODELS_LIST = 'weights', ["LSTM", "DLinear", "PatchTST", "iTransformer", "TCN"]

@st.cache_resource
def get_model(name):
    input_size, seq_len, pred_len = len(TICKERS), 120, 7
    if name == "LSTM": model = LSTMModel(input_size=input_size)
    elif name == "DLinear": model = DLinear(input_size=input_size)
    elif name == "PatchTST": model = PatchTST(input_size=input_size)
    elif name == "iTransformer": model = iTransformer(input_size=input_size)
    elif name == "TCN": model = TCN(input_size=input_size)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, f"{name}.pth"), map_location='cpu'))
    model.eval()
    return model

scaler, df = load_scaler(), fetch_multi_data()
features, btc_idx = list(TICKERS.keys()), list(TICKERS.keys()).index('Bitcoin')

# --- 사이드바 ---
st.sidebar.title("🔍 XAI 분석 엔진")
menu = st.sidebar.radio("이동:", ["📊 통합 예측 비교", "🧠 XAI 분석", "🧪 백테스팅"])
selected_m = st.sidebar.selectbox("주 분석 모델:", MODELS_LIST)

# ---------------------------------------------------------
# 페이지 1: 통합 예측 비교 (기존 유지)
# ---------------------------------------------------------
if menu == "📊 통합 예측 비교":
    st.title("📊 모델별 7일 예측 비교")
    input_tensor = torch.tensor(scaler.transform(df[features].tail(120).values)).float().unsqueeze(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'].tail(30), y=df['Bitcoin'].tail(30), name="Actual", line=dict(color='black', width=3)))
    future_dates = [pd.to_datetime(df['timestamp'].values[-1]) + pd.Timedelta(days=i) for i in range(1, 8)]

    for name in MODELS_LIST:
        with torch.no_grad():
            preds_scaled = get_model(name)(input_tensor).numpy()[0]
        preds = [scaler.inverse_transform(np.array([[0]*btc_idx + [p] + [0]*(len(features)-btc_idx-1)]))[0, btc_idx] for p in preds_scaled]
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name=name))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 페이지 2: 고등 XAI 분석 (2D Heatmap & TimeSHAP)
# ---------------------------------------------------------
elif menu == "🧠 XAI 분석":
    st.title(f"🧠 {selected_m} 모델 정밀 해석 리포트")
    model = get_model(selected_m)
    
    # 데이터 준비
    last_seq_raw = df[features].tail(120).values
    last_seq_scaled = scaler.transform(last_seq_raw)
    input_tensor = torch.tensor(last_seq_scaled).float().unsqueeze(0)
    input_tensor.requires_grad = True
    
    # 1. Saliency 계산
    output = model(input_tensor)
    output[0, 0].backward()
    grads = input_tensor.grad.abs().squeeze().numpy() # [120, 8]
    
    # --- [XAI 1] Time × Feature 2D Heatmap ---
    st.subheader("📍 [Step 1] Time × Feature Saliency Map")
    st.markdown("과거 120일 동안 어떤 지표가 어느 시점에 가장 중요했는지 보여줍니다.")
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=grads.T,
        x=[f"D-{120-i}" for i in range(120)],
        y=features,
        colorscale='YlOrRd',
        colorbar=dict(title="Importance")
    ))
    fig_heat.update_layout(xaxis_title="Time Steps (Past to Present)", yaxis_title="Features")
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- [XAI 2] Simplified TimeSHAP (Temporal Contribution) ---
    st.subheader("⏳ [Step 2] Temporal Feature Contribution (TimeSHAP Style)")
    st.markdown("특정 시간 블록(Cell)을 제외했을 때 예측값의 변화를 측정하여 '시간적 기여도'를 산출합니다.")
    
    # 120일을 10개 블록으로 나누어 SHAP 기여도 계산 (경량화 버전)
    block_size = 12
    temporal_shap = []
    base_pred = output[0, 0].item()
    
    with torch.no_grad():
        for b in range(10):
            perturbed_seq = input_tensor.clone()
            perturbed_seq[0, b*block_size:(b+1)*block_size, :] = 0 # 해당 구간 마스킹
            p_pred = model(perturbed_seq)[0, 0].item()
            temporal_shap.append(abs(base_pred - p_pred)) # 변화량 측정
            
    shap_df = pd.DataFrame({
        'Time Block': [f"Day {b*block_size}~{(b+1)*block_size}" for b in range(10)],
        'Contribution': temporal_shap
    })
    
    fig_shap = px.line(shap_df, x='Time Block', y='Contribution', markers=True, 
                        title="시간 구간별 예측 기여도 (Time-Wise Importance)")
    st.plotly_chart(fig_shap, use_container_width=True)
    
    st.info(f"💡 분석 결과: {selected_m} 모델은 주로 **{shap_df.iloc[shap_df['Contribution'].idxmax()]['Time Block']}** 구간의 데이터에 가장 큰 영향을 받았습니다.")

# ---------------------------------------------------------
# 페이지 3: 백테스팅 (기존 유지)
# ---------------------------------------------------------
elif menu == "🧪 백테스팅":
    st.title("🧪 과거 성과 검증")
    metrics_df = pd.DataFrame({
        "Model": MODELS_LIST,
        "MAE": [1210, 1105, 1090, 1150, 1180],
        "Hit Ratio": ["54.2%", "58.5%", "59.1%", "56.3%", "55.0%"]
    })
    st.table(metrics_df)
