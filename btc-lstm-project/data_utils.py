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
if "FRED_API_KEY" in st.secrets:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
else:
    FRED_API_KEY = os.getenv('FRED_API_KEY', '')

if not FRED_API_KEY:
    print("⚠️ 경고: FRED API 키가 설정되지 않았습니다.")

START_DATE = '2017-01-01'

# 13개 최종 변수 목록
FEATURE_COLUMNS = [
    'BTC_Close', 'BTC_Volume', 'ETH_Close',   # [YFinance] Crypto
    'US_M2', 'US_CPI',                        # [FRED] Economy
    'US_10Y', 'Nasdaq', 'DXY', 'Gold', 'VIX', # [YFinance] Macro
    'Fear_Greed_Index',                       # [API] Sentiment
    'RSI', 'MACD'                             # [Calc] 기술적 지표
]

TICKERS = {col: col for col in FEATURE_COLUMNS}

# ---------------------------------------------------------
# 1. 데이터 수집 함수 (YFinance 통합)
# ---------------------------------------------------------
def fetch_market_data():
    """Crypto(BTC, ETH)와 Macro 데이터를 yfinance에서 수집"""
    print("Fetching Market Data from yfinance...")
    
    symbols = {
        'BTC-USD': ['BTC_Close', 'BTC_Volume'],
        'ETH-USD': ['ETH_Close'],
        '^IXIC': ['Nasdaq'],
        'DX-Y.NYB': ['DXY'],
        'GC=F': ['Gold'],
        '^TNX': ['US_10Y'],
        '^VIX': ['VIX']
    }
    
    try:
        tickers_list = list(symbols.keys())
        df = yf.download(tickers_list, start=START_DATE, progress=False)
        
        data_frames = []
        for ticker, cols in symbols.items():
            for target_col in cols:
                measure = 'Volume' if 'Volume' in target_col else 'Close'
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        series = df.xs(measure, level=0, axis=1)[ticker]
                    else:
                        if len(symbols) == 1: series = df[measure]
                        else: series = df[measure][ticker] if ticker in df[measure].columns else pd.Series(dtype=float)
                except KeyError:
                    series = pd.Series(dtype=float)
                
                series.name = target_col
                data_frames.append(series)
        
        df_market = pd.concat(data_frames, axis=1)
        df_market.index = df_market.index.normalize()
        return df_market

    except Exception as e:
        print(f"⚠️ Market Data Error: {e}")
        return pd.DataFrame()

def fetch_fred():
    """FRED (CPI, M2)"""
    print("Fetching FRED Data...")
    
    if not FRED_API_KEY:
        return pd.DataFrame()

    try:
        fred = Fred(api_key=FRED_API_KEY)
        cpi = fred.get_series('CPIAUCSL', observation_start=START_DATE)
        m2 = fred.get_series('M2SL', observation_start=START_DATE)
        
        if cpi is None: cpi = pd.Series(dtype=float)
        if m2 is None: m2 = pd.Series(dtype=float)
            
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
    
    market = fetch_market_data()
    econ = fetch_fred()
    sent = fetch_sentiment()
    
    if 'BTC_Close' not in market.columns:
        print("🚨 Critical: BTC Data missing")
        return pd.DataFrame(columns=['timestamp'] + FEATURE_COLUMNS)

    df = market.join([econ, sent], how='outer')
    df.sort_index(inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
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
    
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    if len(available_cols) < 5:
         return pd.DataFrame(columns=['timestamp'] + FEATURE_COLUMNS)
         
    df = df[available_cols].dropna()
    
    df_reset = df.reset_index()
    if 'Date' in df_reset.columns:
        df_reset.rename(columns={'Date': 'timestamp'}, inplace=True)
    elif 'index' in df_reset.columns:
        df_reset.rename(columns={'index': 'timestamp'}, inplace=True)
        
    return df_reset

# ---------------------------------------------------------
# 3. 유틸리티 함수 (Smart Scaler)
# ---------------------------------------------------------
def load_scaler(path='weights/scaler.pkl'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)
    
    # 1. 기존 파일 로드 시도
    if os.path.exists(full_path):
        try:
            scaler = joblib.load(full_path)
            # [핵심 수정] 저장된 스케일러의 변수 개수와 현재 설정된 변수 개수(13개) 비교
            if hasattr(scaler, 'n_features_in_'):
                if scaler.n_features_in_ == len(FEATURE_COLUMNS):
                    print("✅ 기존 스케일러 로드 성공")
                    return scaler
                else:
                    print(f"⚠️ 스케일러 차원 불일치 (Old: {scaler.n_features_in_} vs New: {len(FEATURE_COLUMNS)}). 재생성합니다.")
            else:
                print("⚠️ 스케일러 정보 손상. 재생성합니다.")
        except Exception as e:
            print(f"⚠️ 스케일러 로드 실패 ({e}). 재생성합니다.")

    # 2. 스케일러 새로 만들기 (차원이 안 맞거나 파일이 없을 때)
    from sklearn.preprocessing import StandardScaler
    df = fetch_multi_data()
    
    if df.empty: 
        return StandardScaler()
        
    # 현재 정의된 13개 컬럼만 학습
    valid_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    feature_data = df[valid_features]
    
    scaler = StandardScaler()
    scaler.fit(feature_data)
    
    # 저장
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    joblib.dump(scaler, full_path)
    print("✅ 새로운 스케일러(13 features) 생성 및 저장 완료")
    
    return scaler

# ---------------------------------------------------------
# 4. 디스코드 알림 기능 (New!)
# ---------------------------------------------------------
def send_discord_message(title, description, fields=None, color=0x58a6ff):
    """
    Discord Webhook을 통해 메시지를 전송합니다.
    """
    if "DISCORD_WEBHOOK_URL" in st.secrets:
        webhook_url = st.secrets["DISCORD_WEBHOOK_URL"]
    else:
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '')
        
    if not webhook_url:
        return False, "Webhook URL이 설정되지 않았습니다."

    # 임베드(Embed) 메시지 포맷
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "footer": {"text": "TOBIT AI Analyst 🐻"},
        "timestamp": datetime.now().isoformat()
    }
    
    if fields:
        embed["fields"] = fields

    payload = {
        "username": "TOBIT Bot",
        "avatar_url": "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Animals/Bear.png",
        "embeds": [embed]
    }

    try:
        response = requests.post(webhook_url, json=payload)
        if 200 <= response.status_code < 300:
            return True, "전송 성공"
        else:
            return False, f"전송 실패 (Code: {response.status_code})"
    except Exception as e:
        return False, f"에러 발생: {e}"
