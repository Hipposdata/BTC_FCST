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
FRED_API_KEY = os.getenv('FRED_API_KEY', '33f21fe5eacad6f3c9e71ca9ed7d0e1a')
START_DATE = '2017-01-01'

# 학습/예측에서 사용할 최종 변수 목록 (13개)
FEATURE_COLUMNS = [
    'BTC_Close', 'BTC_Volume', 'ETH_Close',   # 시장 활동성
    'US_M2', 'US_CPI', 'US_10Y', 'Nasdaq',    # 거시경제
    'DXY', 'Gold',                            # 대체/안전 자산
    'Fear_Greed_Index', 'VIX', 'RSI', 'MACD'  # 심리 및 기술적 지표
]

TICKERS = {col: col for col in FEATURE_COLUMNS}

# ---------------------------------------------------------
# 1. 데이터 수집 함수들
# ---------------------------------------------------------
def fetch_binance_price(symbol, name):
    """바이낸스 데이터 수집 (실패 시 yfinance로 Fallback)"""
    print(f"Fetching {name} data...")
    
    # 1차 시도: Binance API
    try:
        url = "https://api.binance.com/api/v3/klines"
        start_ts = int(pd.Timestamp(START_DATE).timestamp() * 1000)
        end_ts = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        current = start_ts
        
        # 3번 정도만 시도해보고 안되면 바로 포기 (속도 위해)
        retry_count = 0
        while current < end_ts and retry_count < 3:
            params = {'symbol': symbol, 'interval': '1d', 'startTime': current, 'limit': 1000}
            resp = requests.get(url, params=params, timeout=5).json()
            
            if not resp or isinstance(resp, dict): # 에러거나 빈 응답
                break
                
            all_data.extend(resp)
            current = resp[-1][0] + 1
            time.sleep(0.05)
            
        if all_data:
            columns = [
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
            ]
            df = pd.DataFrame(all_data, columns=columns)
            df['Date'] = pd.to_datetime(df['Open time'], unit='ms').dt.normalize()
            df.set_index('Date', inplace=True)
            
            # 필요한 컬럼 선택
            target_cols = ['Close', 'Volume'] if name == 'BTC' else ['Close']
            for c in target_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df = df[target_cols]
            df.columns = [f"{name}_{c}" for c in df.columns]
            return df
            
    except Exception as e:
        print(f"⚠️ Binance API Error: {e}")

    # 2차 시도: YFinance (Fallback)
    print(f"🔄 Switching to YFinance for {name}...")
    try:
        yf_symbol = "BTC-USD" if name == "BTC" else "ETH-USD"
        df = yf.download(yf_symbol, start=START_DATE, progress=False)
        
        # MultiIndex 컬럼 처리 (yfinance 최신버전 이슈)
        # Close 처리
        if 'Close' in df.columns:
            if isinstance(df.columns, pd.MultiIndex):
                try: close = df.xs('Close', level=0, axis=1).iloc[:, 0]
                except: close = df['Close']
            else:
                close = df['Close']
        else:
            return pd.DataFrame() # Close도 없으면 실패

        # Volume 처리 (BTC인 경우)
        if name == 'BTC':
            if 'Volume' in df.columns:
                if isinstance(df.columns, pd.MultiIndex):
                    try: vol = df.xs('Volume', level=0, axis=1).iloc[:, 0]
                    except: vol = df['Volume']
                else:
                    vol = df['Volume']
            else:
                vol = pd.Series(0, index=close.index) # 없으면 0으로 채움
            
            df_final = pd.DataFrame({f"{name}_Close": close, f"{name}_Volume": vol})
        else:
            df_final = pd.DataFrame({f"{name}_Close": close})
            
        df_final.index = df_final.index.normalize()
        return df_final

    except Exception as e:
        print(f"❌ YFinance Error for {name}: {e}")
        return pd.DataFrame()

def fetch_macro():
    """Yahoo Finance 거시경제 지표"""
    print("Fetching Macro Data...")
    tickers = {
        '^NDX': 'Nasdaq', 'DX-Y.NYB': 'DXY', 'GC=F': 'Gold', 
        '^TNX': 'US_10Y', '^VIX': 'VIX'
    }
    try:
        df = yf.download(list(tickers.keys()), start=START_DATE, progress=False)['Close']
        # MultiIndex 처리
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # 컬럼 이름이 티커로 되어있을 수 있으므로 매핑
        # yfinance가 요청한 티커 순서대로 주지 않을 수 있음 -> rename dict 사용
        df.rename(columns=tickers, inplace=True)
        
        # 없는 컬럼 확인 및 채우기 (혹시 다운로드 실패시)
        for code, name in tickers.items():
            if name not in df.columns and code in df.columns:
                 df.rename(columns={code: name}, inplace=True)
                 
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
        
        # 시리즈가 비어있을 경우 대비
        if cpi is None or cpi.empty: cpi = pd.Series(dtype=float)
        if m2 is None or m2.empty: m2 = pd.Series(dtype=float)
            
        df = pd.DataFrame({'US_CPI': cpi, 'US_M2': m2})
        df.index = pd.to_datetime(df.index).normalize()
        return df.resample('D').ffill()
    except Exception as e:
        print(f"⚠️ FRED API Error: {e}")
        # 실패 시 빈 DataFrame 반환하여 merge 단계에서 무시되거나 NaN 처리되게 함
        return pd.DataFrame(columns=['US_CPI', 'US_M2'])

def fetch_sentiment():
    """Fear & Greed Index"""
    print("Fetching Sentiment...")
    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        resp = requests.get(url, timeout=5).json()
        data = resp.get('data', [])
        if not data: return pd.Series(dtype=float, name='Fear_Greed_Index')
            
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s').dt.normalize()
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        s = pd.to_numeric(df['value'], errors='coerce')
        s.name = 'Fear_Greed_Index'
        return s
    except Exception as e:
        print(f"⚠️ Sentiment Error: {e}")
        return pd.Series(dtype=float, name='Fear_Greed_Index')

# ---------------------------------------------------------
# 2. 메인 데이터 처리 함수
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_multi_data():
    """모든 데이터를 수집하고 13개 변수로 정리하여 반환"""
    print("🚀 데이터 수집 및 전처리 시작...")
    
    # 1. 수집
    btc = fetch_binance_price('BTCUSDT', 'BTC')
    
    # [중요] BTC 데이터가 없으면 진행 불가
    if btc.empty or 'BTC_Close' not in btc.columns:
        print("🚨 CRITICAL: BTC 데이터를 가져오지 못했습니다.")
        # 빈 껍데기라도 반환하여 앱이 멈추는 것 방지 (또는 에러 발생)
        return pd.DataFrame(columns=['timestamp'] + FEATURE_COLUMNS)

    eth = fetch_binance_price('ETHUSDT', 'ETH')
    macro = fetch_macro()
    econ = fetch_fred()
    sent = fetch_sentiment()
    
    # 2. 병합
    # outer join으로 최대한 살리고 ffill로 메꿈
    dfs = [d for d in [btc, eth, macro, econ] if not d.empty]
    if not sent.empty: dfs.append(sent)
        
    df = dfs[0]
    for d in dfs[1:]:
        df = df.join(d, how='outer')
        
    df.sort_index(inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True) # 앞부분 결측 제거
    
    # 3. 기술적 지표 계산 (RSI, MACD)
    # 병합 후 BTC_Close가 있는지 재확인
    if 'BTC_Close' not in df.columns:
        return pd.DataFrame(columns=['timestamp'] + FEATURE_COLUMNS)
        
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
    # 실제 존재하는 컬럼만 선택 (혹시 API 실패로 일부 누락되어도 앱이 켜지도록)
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    # 최소한 BTC_Close는 있어야 함
    if 'BTC_Close' not in available_cols:
         return pd.DataFrame(columns=['timestamp'] + FEATURE_COLUMNS)
         
    df = df[available_cols].dropna()
    
    # app.py 시각화를 위해 timestamp 컬럼 생성
    df_reset = df.reset_index()
    if 'Date' in df_reset.columns:
        df_reset.rename(columns={'Date': 'timestamp'}, inplace=True)
    elif 'index' in df_reset.columns:
        df_reset.rename(columns={'index': 'timestamp'}, inplace=True)
        
    return df_reset

# ---------------------------------------------------------
# 3. 유틸리티 함수
# ---------------------------------------------------------
def load_scaler(path='weights/scaler.pkl'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)
    
    if os.path.exists(full_path):
        return joblib.load(full_path)
    
    from sklearn.preprocessing import StandardScaler
    df = fetch_multi_data()
    
    # 데이터가 비어있으면 scaler 학습 불가 -> 임시 반환
    if df.empty:
        return StandardScaler()
        
    # 현재 데이터프레임에 있는 컬럼만 골라서 학습
    valid_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    feature_data = df[valid_features]
    
    scaler = StandardScaler()
    scaler.fit(feature_data)
    
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(scaler, full_path)
    
    return scaler
