import streamlit as st
import torch
import numpy as np
import time
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from model import LSTMModel
from data_utils import fetch_btc_ohlcv, load_scaler

# 1. 페이지 및 경로 설정
st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKTEST_PATH = os.path.join(BASE_DIR, 'data/backtest_history.csv')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'weights/model.pth')

os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

# 2. 모델 로드 함수 (캐싱)
@st.cache_resource
def init_model():
    model = LSTMModel()
    if not os.path.exists(WEIGHTS_PATH):
        st.error("❌ 모델 파일을 찾을 수 없습니다.")
        st.stop()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    scaler = load_scaler()
    return model, scaler

model, scaler = init_model()

# --- 사이드바 메뉴 ---
st.sidebar.title("🚀 메뉴 선택")
menu = st.sidebar.radio("이동할 페이지:", ["실시간 예측", "백테스트 분석"])

# 3. [공통 로직] 백테스트 데이터 업데이트 함수
def update_backtest_data(current_date, current_price, prediction, next_day):
    if os.path.exists(BACKTEST_PATH):
        bt_df = pd.read_csv(BACKTEST_PATH)
        bt_df['date'] = pd.to_datetime(bt_df['date']).dt.date
    else:
        bt_df = pd.DataFrame(columns=['date', 'predicted', 'actual', 'error'])

    # 오늘 실제가 업데이트
    if not bt_df.empty and current_date in bt_df['date'].values:
        idx = bt_df[bt_df['date'] == current_date].index[0]
        if pd.isna(bt_df.at[idx, 'actual']):
            bt_df.at[idx, 'actual'] = current_price
            bt_df.at[idx, 'error'] = current_price - bt_df.at[idx, 'predicted']
            bt_df.to_csv(BACKTEST_PATH, index=False, encoding='utf-8-sig')

    # 내일 예측가 생성
    if next_day not in bt_df['date'].values:
        new_row = pd.DataFrame({'date': [next_day], 'predicted': [prediction], 'actual': [np.nan], 'error': [np.nan]})
        bt_df = pd.concat([bt_df, new_row], ignore_index=True)
        bt_df.to_csv(BACKTEST_PATH, index=False, encoding='utf-8-sig')
    return bt_df

# ---------------------------------------------------------
# 페이지 1: 실시간 예측 (Live Predictor)
# ---------------------------------------------------------
if menu == "실시간 예측":
    st.title("📈 BTC 향후 7일 AI 예측")
    
    with st.spinner("최신 시장 데이터 분석 중..."):
        df = fetch_multi_data()
    
    if not df.empty:
        features = list(TICKERS.keys())
        current_price = df['Bitcoin'].values[-1]
        last_date = pd.to_datetime(df['timestamp'].values[-1])
        
        # 7일간의 날짜 생성
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]

        # 모델 추론
        last_seq_scaled = scaler.transform(df[features].tail(120).values)
        input_tensor = torch.tensor(last_seq_scaled).float().unsqueeze(0)
        
        with torch.no_grad():
            preds_scaled = model(input_tensor).numpy()[0] # 7개의 예측값
            
        # 7개 예측값 각각 역스케일링
        predictions = []
        btc_idx = features.index('Bitcoin')
        for p in preds_scaled:
            dummy = np.zeros((1, len(features)))
            dummy[0, btc_idx] = p
            predictions.append(scaler.inverse_transform(dummy)[0, btc_idx])

        # UI 표시
        st.subheader(f"📅 향후 7일 예측가")
        cols = st.columns(7)
        for i, col in enumerate(cols):
            col.metric(f"D+{i+1}", f"${predictions[i]:,.0f}")

        # 차트 시각화
        fig = go.Figure()
        # 과거 데이터 (최근 30일)
        fig.add_trace(go.Scatter(x=df['timestamp'].tail(30), y=df['Bitcoin'].tail(30), name='Past Price'))
        # 미래 예측 데이터
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, name='7-Day Forecast', 
                                 line=dict(color='red', dash='dash', width=3),
                                 mode='lines+markers'))
        
        fig.update_layout(title="비트코인 7일 예측 트렌드", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        
# ---------------------------------------------------------
# 페이지 2: 백테스트 분석 (Backtest Lab)
# ---------------------------------------------------------
elif menu == "백테스트 분석":
    st.title("🧪 백테스트 분석 연구소")

    if os.path.exists(BACKTEST_PATH):
        bt_df = pd.read_csv(BACKTEST_PATH)
        bt_df = bt_df.dropna(subset=['actual']) # 결과가 나온 데이터만
        
        if not bt_df.empty:
            # 1. 통계 지표 계산
            mae = bt_df['error'].abs().mean()
            rmse = np.sqrt((bt_df['error']**2).mean())
            
            # 방향 적중률 (Hit Ratio) 계산
            # 실제 등락과 예측 등락의 방향이 같은지 확인하는 간단한 로직 예시
            # (실제-어제실제) * (예측-어제실제) > 0 이면 방향 적중
            
            st.subheader("🚩 주요 성능 지표")
            m1, m2, m3 = st.columns(3)
            m1.metric("평균 절대 오차 (MAE)", f"${mae:,.2f}")
            m2.metric("평균 제곱근 오차 (RMSE)", f"${rmse:,.2f}")
            m3.metric("누적 기록 수", f"{len(bt_df)}일")

            # 2. 시각화 차트
            st.markdown("---")
            tab1, tab2 = st.tabs(["예측 vs 실제 비교", "오차 분포"])
            
            with tab1:
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=bt_df['date'], y=bt_df['actual'], name="Actual Price", line=dict(color='black', width=2)))
                fig_comp.add_trace(go.Scatter(x=bt_df['date'], y=bt_df['predicted'], name="Predicted Price", line=dict(color='orange', dash='dash')))
                fig_comp.update_layout(title="과거 예측 성과 비교", template="plotly_dark", height=500)
                st.plotly_chart(fig_comp, use_container_width=True)

            with tab2:
                fig_err = go.Figure()
                fig_err.add_trace(go.Bar(x=bt_df['date'], y=bt_df['error'], 
                                         marker_color=['red' if x > 0 else 'blue' for x in bt_df['error']]))
                fig_err.update_layout(title="일별 오차 (Actual - Prediction)", template="plotly_white", height=400)
                st.plotly_chart(fig_err, use_container_width=True)

            # 3. 데이터 상세 보기
            with st.expander("전체 백테스트 로그 확인"):
                st.dataframe(bt_df.sort_values(by='date', ascending=False), use_container_width=True)
                
            # CSV 다운로드 버튼
            csv = bt_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("백테스트 데이터 다운로드(CSV)", csv, "btc_backtest_report.csv", "text/csv")
            
        else:
            st.warning("아직 기록된 백테스트 결과가 없습니다. 첫 예측 후 다음 날 실제 데이터가 들어와야 표시됩니다.")
    else:
        st.error("백테스트 데이터 파일이 존재하지 않습니다.")
