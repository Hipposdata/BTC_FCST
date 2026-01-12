import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# 사용할 외생변수 티커 정의
TICKERS = {
    'Bitcoin': 'BTC-USD',
    'DXY': 'DX-Y.NYB',
    'Nasdaq': '^IXIC',
    'S&P500': '^GSPC',
    'US_10Y': '^TNX',
    'Gold': 'GC=F',
    'VIX': '^VIX',
    'WTI_Oil': 'CL=F'
}

def fetch_multi_data(start_date='2015-01-01'):
    """ 여러 지표 데이터를 가져와서 병합하고 결측치를 처리하는 함수"""
    dfs = []
    for name, ticker in TICKERS.items():
        data = yf.download(ticker, start=start_date, interval='1d', progress=False)
        
        # yfinance의 멀티인덱스 컬럼 대응
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        data = data[['Close']].rename(columns={'Close': name})
        dfs.append(data)

    # 데이터 병합 및 시계열 보간
    df = pd.concat(dfs, axis=1)
    full_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_dates)
    # 선형 보간 후 앞뒤 결측치 채움
    df = df.interpolate(method='linear').ffill().bfill()
    
    return df.reset_index().rename(columns={'index': 'timestamp'})

def create_sequences(data, seq_length, prediction_days=7, target_col_idx=0):
    """ 다변수 데이터로부터 미래 7일치를 타겟으로 하는 시퀀스 생성"""
    xs, ys = [], []
    for i in range(len(data) - seq_length - prediction_days + 1):
        x = data[i:(i + seq_length)]
        # 타겟: i + seq_length 시점부터 7일간의 데이터
        y = data[i + seq_length : i + seq_length + prediction_days, target_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def save_scaler(scaler, path='weights/scaler.pkl'):
    """ 스케일러 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path='weights/scaler.pkl'):
    """ 저장된 스케일러 로드"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {full_path}")
    return joblib.load(full_path)
