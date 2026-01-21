import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import requests
import joblib
import os
import time
from datetime import datetime
import streamlit as st

# ---------------------------------------------------------
# 설정 및 상수
# ---------------------------------------------------------
# FRED API 키 (없으면 기본값 사용, 필요시 본인 키로 수정)
FRED_API_KEY = os.getenv('FRED_API_KEY', '33f21fe5eacad6f3c9e71ca9ed7d0e1a')
START_DATE = '2017-01-01'

# 학습/예측에서 사용할 최종 변수 목록 (13개)
FEATURE_COLUMNS = [
    'BTC_Close', 'BTC_Volume', 'ETH_Close',   # 시장 활동성
    'US_M2', 'US_CPI', 'US_10Y', 'Nasdaq',    # 거시경제
    'DXY', 'Gold',                            # 대체/안전 자산
    'Fear_Greed_Index', 'VIX', 'RSI', 'MACD'  # 심리 및 기술적 지표
]

# app.py와의 호환성을 위해 TICKERS 딕셔너리 정의
TICKERS = {col: col for col in FEATURE_COLUMNS}

# ---------------------------------------------------------
# 1. 데이터 수집 함수들
# ---------------------------------------------------------
def fetch_binance_price(symbol, name):
    """바이낸스 데이터 수집"""
    print(f"Fetching Binance {name}...")
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)
    
    all_data = []
    current = start_ts
    
    while current < end_ts:
        params = {'symbol': symbol, 'interval': '1d', 'startTime': current, 'limit': 1000}
        try:
            resp = requests.get(url, params=params, timeout=10).json()
            if not resp or isinstance(resp, dict): break
            all_data.extend(resp)
            current = resp[-1][0] + 1
            time.sleep(0.05)
        except: break
    
    if not all_data:
        print(f"⚠️ {name} 데이터가 비어있습니다.")
        return pd.DataFrame()

    columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    
    df = pd.DataFrame(all_data, columns=columns)
    df = df[['Open time', 'Close', 'Volume']]
    df.columns = ['Date', 'Close', 'Volume']
    
    df['Date'] = pd.to_datetime(df['Date'], unit='ms').dt.normalize()
    df.set_index('Date', inplace=True)
    
    cols = ['Close', 'Volume'] if name == 'BTC' else ['Close']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df = df[cols]
    df.columns = [f"{name}_{c}" for c in df.columns]
    return df

def fetch_macro():
    """Yahoo Finance 거시경제 지표"""
    print("Fetching Macro Data...")
    tickers = {
        '^NDX': 'Nasdaq', 'DX-Y.NYB': 'DXY', 'GC=F': 'Gold', 
        '^TNX': 'US_10Y', '^VIX': 'VIX'
    }
    try:
        df = yf.download(list(tickers.keys()), start=START_DATE, progress=False)['Close']
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        df.rename(columns=tickers, inplace=True)
        df.index = df.index.normalize()
        return df
    except Exception as e:
        print(f"⚠️ Yahoo Finance Error: {e}")
        return pd.DataFrame()

def fetch_fred():
    """FRED (CPI, M2)"""
    print("Fetching FRED Data...")
    try:
        fred = Fred(api_key=FRED_API_KEY)
        cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE)
        m2 = fred.get_series('M2SL', observation_start=START_DATE)
        df = pd.DataFrame({'US_CPI': cpi, 'US_M2': m2})
        df.index = pd.to_datetime(df.index).normalize()
        return df.resample('D').ffill()
    except Exception as e:
        print(f"⚠️ FRED API Error: {e}")
        return pd.DataFrame()

def fetch_sentiment():
    """Fear & Greed Index"""
    print("Fetching Sentiment...")
    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        data = requests.get(url).json()['data']
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s').dt.normalize()
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        s = pd.to_numeric(df['value'])
        s.name = 'Fear_Greed_Index'
        return s
    except Exception as e:
        print(f"⚠️ Sentiment Error: {e}")
        return pd.Series(dtype=float)

# ---------------------------------------------------------
# 2. 메인 데이터 처리 함수
# ---------------------------------------------------------
@st.cache_data(ttl=3600)  # Streamlit 캐싱 적용
def fetch_multi_data():
    """모든 데이터를 수집하고 13개 변수로 정리하여 반환"""
    print("🚀 데이터 수집 및 전처리 시작...")
    
    # 1. 수집
    btc = fetch_binance_price('BTCUSDT', 'BTC')
    eth = fetch_binance_price('ETHUSDT', 'ETH')
    macro = fetch_macro()
    econ = fetch_fred()
    sent = fetch_sentiment()
    
    # 2. 병합
    df = btc.join([eth, macro, econ, sent], how='outer').sort_index()
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    # 3. 기술적 지표 계산 (RSI, MACD)
    close = df['BTC_Close']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # 4. 최종 컬럼 필터링 (13개)
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[available_cols].dropna()
    
    # [중요] app.py 시각화를 위해 timestamp 컬럼 생성 (index -> column)
    # reset_index()를 하면 index 이름이 컬럼으로 들어옴
    df_reset = df.reset_index()
    
    # 컬럼명 통일 (Date 또는 index -> timestamp)
    if 'Date' in df_reset.columns:
        df_reset.rename(columns={'Date': 'timestamp'}, inplace=True)
    elif 'index' in df_reset.columns:
        df_reset.rename(columns={'index': 'timestamp'}, inplace=True)
        
    return df_reset

# ---------------------------------------------------------
# 3. 유틸리티 함수 (학습/앱 공용)
# ---------------------------------------------------------
def load_scaler(path='weights/scaler.pkl'):
    """저장된 스케일러 로드 또는 새로 생성"""
    # Streamlit Cloud 경로 호환성
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)
    
    if os.path.exists(full_path):
        return joblib.load(full_path)
    
    # 파일이 없으면 새로 학습해서 반환 (앱 에러 방지)
    from sklearn.preprocessing import StandardScaler
    df = fetch_multi_data()
    feature_data = df[FEATURE_COLUMNS]
    scaler = StandardScaler()
    scaler.fit(feature_data)
    
    # weights 폴더가 없으면 생성 후 저장
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(scaler, full_path)
    
    return scaler
