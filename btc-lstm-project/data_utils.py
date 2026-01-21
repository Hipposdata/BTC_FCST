import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------
# 1. 설정 및 상수
# ------------------------------------------------------------------------------
DATA_PATH = "btc_fcst_data.csv"
SCALER_PATH = "scaler.pkl"

# 모델 학습/예측에 사용할 변수 목록 (app.py와 이름이 일치해야 함)
TICKERS = {
    'BTC_Close': 'BTC-USD',
    'ETH_Close': 'ETH-USD',
    'US_10Y': '^TNX',        # 미국 10년물 국채 금리
    'Nasdaq': '^IXIC',       # 나스닥 지수
    'S&P500': '^GSPC',       # S&P 500
    'DXY': 'DX-Y.NYB',       # 달러 인덱스
    'Gold': 'GC=F'           # 금 선물
}

# ------------------------------------------------------------------------------
# 2. 데이터 수집 및 전처리 함수
# ------------------------------------------------------------------------------
def fetch_multi_data():
    """
    Yahoo Finance에서 데이터를 가져와 전처리 후 DataFrame 반환
    """
    print("📥 데이터 다운로드 중...")
    
    # 1) 비트코인 및 주요 지표 다운로드
    df_list = []
    for col_name, ticker in TICKERS.items():
        try:
            # 최근 2년치 데이터 다운로드 (속도 최적화)
            data = yf.download(ticker, period="2y", interval="1d", progress=False)
            
            # MultiIndex 컬럼 처리 (yfinance 최신 버전 호환)
            if isinstance(data.columns, pd.MultiIndex):
                data = data.xs('Close', level=0, axis=1) if 'Close' in data.columns.get_level_values(0) else data.iloc[:, 0]
            elif 'Close' in data.columns:
                data = data[['Close']]
            else:
                data = data.iloc[:, 0] # 첫 번째 컬럼 사용
            
            # 컬럼명 변경 (예: Close -> BTC_Close)
            if isinstance(data, pd.Series):
                data = data.to_frame(name=col_name)
            else:
                data.columns = [col_name]
            
            df_list.append(data)
        except Exception as e:
            print(f"⚠️ {ticker} 다운로드 실패: {e}")

    # 2) 데이터 병합 (날짜 기준)
    if not df_list:
        raise ValueError("데이터를 가져오지 못했습니다. 인터넷 연결을 확인하세요.")
        
    df = pd.concat(df_list, axis=1).dropna()
    df.index.name = 'timestamp'
    df.reset_index(inplace=True)

    # 3) 보조 지표 생성 (RSI, MACD, Fear&Greed)
    # RSI (14일)
    delta = df['BTC_Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Fear & Greed Index (가상 데이터: RSI 기반 근사치)
    # 실제 API는 유료거나 복잡하므로 RSI와 변동성을 섞어 시뮬레이션
    df['Fear_Greed_Index'] = df['RSI'].rolling(7).mean().fillna(50)

    # 결측치 제거 (지표 계산으로 생긴 앞부분 NaN)
    df = df.dropna().reset_index(drop=True)
    
    print(f"✅ 데이터 로드 완료: {len(df)} rows")
    return df

# ------------------------------------------------------------------------------
# 3. 스케일러 로드/생성 함수
# ------------------------------------------------------------------------------
def load_scaler():
    """
    저장된 스케일러가 있으면 로드하고, 없으면 새로 생성해서 반환
    """
    # 데이터 먼저 로드해서 스케일러 학습에 사용
    df = fetch_multi_data()
    features = list(TICKERS.keys()) # 스케일링할 컬럼들
    
    # 실제 존재하는 컬럼만 필터링
    valid_features = [f for f in features if f in df.columns]
    
    scaler = MinMaxScaler()
    scaler.fit(df[valid_features])
    
    # (선택) 스케일러 저장
    # joblib.dump(scaler, SCALER_PATH) 
    
    return scaler

# 테스트용 실행 코드
if __name__ == "__main__":
    df = fetch_multi_data()
    print(df.head())
    print("컬럼 목록:", df.columns)
