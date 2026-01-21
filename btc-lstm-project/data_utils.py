import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import requests

# ------------------------------------------------------------------------------
# 1. Ticker Configuration
# ------------------------------------------------------------------------------
# FREDAPI 대신 yfinance(^TNX)를 사용하여 에러 가능성을 차단합니다.
TICKERS = {
    'BTC_Close': 'BTC-USD',
    'Fear_Greed_Index': 'API',
    'RSI': 'Calc',
    'US_10Y': '^TNX'
}

# ------------------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------------------
def calculate_rsi(series, period=14):
    """RSI(상대강도지수) 계산"""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_fear_greed_index(limit=1000):
    """공포 탐욕 지수 API 호출"""
    url = f"https://api.alternative.me/fng/?limit={limit}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['value'] = df['value'].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.set_index('timestamp')['value']
    except Exception as e:
        print(f"F&G API Error: {e}")
        return pd.Series(dtype=float)

# ------------------------------------------------------------------------------
# 3. Main Data Fetching (Robost Version)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_multi_data():
    # 1. 비트코인 데이터 (BTC-USD)
    df_btc = yf.download('BTC-USD', period='2y', progress=False)
    
    # [Fix] MultiIndex 컬럼 문제 해결 (Price, Ticker) -> Price
    if isinstance(df_btc.columns, pd.MultiIndex):
        try:
            df_btc = df_btc.xs('Close', level=0, axis=1) # Close 컬럼만 추출
        except:
            df_btc = df_btc['Close'] # 일반적인 경우
            
    # 컬럼이 하나만 남았는지 확인 후 이름 변경
    if len(df_btc.columns) > 1:
        # 혹시 여러 티커가 섞여 있으면 첫 번째 것만 사용
        df_btc = df_btc.iloc[:, 0].to_frame()
    
    df_btc.columns = ['BTC_Close']

    # 2. 미국 국채 10년물 (^TNX)
    df_tnx = yf.download('^TNX', period='2y', progress=False)
    
    # [Fix] MultiIndex 컬럼 문제 해결
    if isinstance(df_tnx.columns, pd.MultiIndex):
        try:
            df_tnx = df_tnx.xs('Close', level=0, axis=1)
        except:
            df_tnx = df_tnx['Close']
            
    if len(df_tnx.columns) > 1:
        df_tnx = df_tnx.iloc[:, 0].to_frame()
        
    df_tnx.columns = ['US_10Y']

    # 3. 데이터 병합 (Left Join)
    df = df_btc.join(df_tnx, how='left')

    # [Fix] 결측치 채우기 (주말 금리 NaN 처리)
    df['US_10Y'] = df['US_10Y'].ffill().bfill()

    # 4. 보조지표 추가
    # RSI 계산
    df['RSI'] = calculate_rsi(df['BTC_Close'])
    
    # 공포/탐욕 지수 병합
    fg_series = get_fear_greed_index(limit=len(df) + 100)
    df['date_key'] = df.index.normalize()
    
    if not fg_series.empty:
        fg_series.index = fg_series.index.normalize()
        fg_series = fg_series[~fg_series.index.duplicated(keep='first')]
        df = df.join(fg_series.rename('Fear_Greed_Index'), on='date_key', how='left')
        df['Fear_Greed_Index'] = df['Fear_Greed_Index'].ffill().bfill()
    else:
        df['Fear_Greed_Index'] = 50.0

    # 5. 최종 정리
    df = df.drop(columns=['date_key'], errors='ignore')
    df = df.dropna()
    
    # timestamp 컬럼 추가 (Plotly용)
    df['timestamp'] = df.index
    
    return df

def load_scaler():
    df = fetch_multi_data()
    features = list(TICKERS.keys())
    
    # 데이터프레임에 실제 존재하는 컬럼만 선택
    valid_features = [f for f in features if f in df.columns]
    
    if not valid_features:
        raise ValueError("No valid features found. Check data fetching.")

    scaler = StandardScaler()
    scaler.fit(df[valid_features])
    
    return scaler
