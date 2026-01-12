import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os

def fetch_btc_ohlcv(limit=2000):
    """
    yfinance의 멀티 인덱스 컬럼 구조를 처리하고 
    데이터 프레임을 표준 형식으로 반환합니다.
    """
    # 1. 데이터 다운로드 (BTC-USD 일봉)
    data = yf.download(tickers="BTC-USD", period="max", interval="1d")
    
    # 2. 인덱스 초기화 (Date를 컬럼으로 변환)
    df = data.reset_index()
    
    # 3. 컬럼명 처리 (멀티 인덱스 대응)
    # yfinance 버전 업데이트로 인해 컬럼이 ('Close', 'BTC-USD') 같은 튜플로 들어오는 경우 해결
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_columns.append(col[0].lower()) # 튜플의 첫 번째 요소(Open, Close 등)만 추출
        else:
            new_columns.append(col.lower())
    df.columns = new_columns
    
    # 4. 컬럼명 표준화 및 필수 컬럼 선택
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
        
    # 데이터 정합성을 위해 필수 컬럼만 슬라이싱
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_cols]
    
    # 최근 데이터만 반환
    return df.tail(limit)

def create_sequences(data, seq_length, prediction_days=7, target_col_idx=0):
    """
    prediction_days: 예측하고 싶은 미래 일수 (7일)
    """
    xs, ys = [], []
    # 데이터를 7일치 더 남겨두고 루프를 돌아야 합니다.
    for i in range(len(data) - seq_length - prediction_days + 1):
        x = data[i:(i + seq_length)]
        # 타겟: i + seq_length 부터 7일간의 종가 데이터
        y = data[i + seq_length : i + seq_length + prediction_days, target_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def save_scaler(scaler, path='weights/scaler.pkl'):
    """스케일러 저장"""
    joblib.dump(scaler, path)

def load_scaler(path='weights/scaler.pkl'):
    """절대 경로를 사용하여 스케일러 로드 (Streamlit Cloud 대응)"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {full_path}")
        
    return joblib.load(full_path)
