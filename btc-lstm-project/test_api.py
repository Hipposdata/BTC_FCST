import yfinance as yf
import pandas as pd

# 1. 비트코인 데이터만 심플하게 가져와 봅니다.
print("--- [1] 비트코인 데이터 요청 시작 ---")
try:
    btc = yf.download("BTC-USD", period="1mo", progress=False)
    
    if btc.empty:
        print("❌ 실패: 데이터가 비어있습니다. (API 호출은 됐으나 내용 없음)")
    else:
        print(f"✅ 성공: {len(btc)}개의 데이터를 가져왔습니다.")
        print(btc.head()) # 데이터 눈으로 확인
except Exception as e:
    print(f"❌ 에러 발생: {e}")

# 2. 경제 지표(금리 등)도 확인해 봅니다.
print("\n--- [2] 경제 지표(US 10Y) 요청 시작 ---")
try:
    rate = yf.download("^TNX", period="1mo", progress=False) # 미국 10년물 국채
    if rate.empty:
        print("❌ 실패: 경제 지표 데이터가 비어있습니다.")
    else:
        print(f"✅ 성공: {len(rate)}개의 데이터를 가져왔습니다.")
except Exception as e:
    print(f"❌ 에러 발생: {e}")
