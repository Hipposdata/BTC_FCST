import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 필수 라이브러리/모듈 임포트
try:
    from data_utils import fetch_multi_data, load_scaler, send_discord_message, TICKERS
    from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN, MLP
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 설정 (자동 실행용)
# ---------------------------------------------------------
# 매일 아침 사용할 모델 (가장 성능 좋은 걸로 지정하세요)
TARGET_MODEL = "LSTM" 
SEQ_LEN = 14
PRED_LEN = 7

# GitHub Actions 환경변수에서 키 가져오기 (없으면 로컬 테스트용)
FRED_API_KEY = os.getenv("FRED_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# ---------------------------------------------------------
# 모델 로드 함수 (Headless)
# ---------------------------------------------------------
def get_model_headless(name, seq_len):
    input_size = len(TICKERS)
    
    # 모델 초기화
    if name == "MLP": model = MLP(seq_len=seq_len, input_size=input_size, pred_len=PRED_LEN)
    elif name == "DLinear": model = DLinear(seq_len=seq_len, pred_len=PRED_LEN, input_size=input_size, kernel_size=25)
    elif name == "TCN": model = TCN(input_size=input_size, output_size=PRED_LEN, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2)
    elif name == "LSTM": model = LSTMModel(input_size=input_size, output_size=PRED_LEN)
    elif name == "PatchTST": model = PatchTST(input_size=input_size, seq_len=seq_len, pred_len=PRED_LEN, patch_len=7, stride=3, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    elif name == "iTransformer": model = iTransformer(seq_len=seq_len, pred_len=PRED_LEN, input_size=input_size, d_model=256, n_heads=4, n_layers=3, dropout=0.2)
    else: return None

    # 가중치 로드
    weights_path = os.path.join("weights", f"{name}_{seq_len}.pth")
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print(f"✅ {name} 가중치 로드 성공")
        except:
            print(f"⚠️ {name} 가중치 로드 실패 (초기화 상태로 진행)")
    else:
        print(f"⚠️ 가중치 파일 없음: {weights_path}")
        
    model.eval()
    return model

# ---------------------------------------------------------
# 메인 실행 로직
# ---------------------------------------------------------
def run_daily_report():
    print("🚀 Daily Report Bot 시작...")

    if not DISCORD_WEBHOOK_URL:
        print("❌ Discord Webhook URL이 없습니다. 종료합니다.")
        return

    # 1. 데이터 수집
    df = fetch_multi_data()
    if df.empty:
        print("❌ 데이터 수집 실패")
        send_discord_message("🚨 TOBIT Bot Error", "데이터를 가져오지 못했습니다.")
        return

    # 2. 전처리 및 예측
    scaler = load_scaler()
    features = list(TICKERS.keys())
    
    try:
        btc_idx = features.index('BTC_Close')
    except:
        btc_idx = 0

    model = get_model_headless(TARGET_MODEL, SEQ_LEN)
    
    # 예측 수행
    input_raw = df[features].tail(SEQ_LEN).values
    input_tensor = torch.tensor(scaler.transform(input_raw)).float().unsqueeze(0)
    
    with torch.no_grad():
        preds_scaled = model(input_tensor).numpy()[0]
    
    # 역변환 (스케일링 해제)
    preds = []
    for p in preds_scaled:
        dummy = np.zeros(len(features))
        dummy[btc_idx] = p
        preds.append(scaler.inverse_transform(dummy.reshape(1, -1))[0][btc_idx])
    
    target_price_7d = preds[-1]
    current_price = df['BTC_Close'].iloc[-1]
    
    # 3. 메시지 작성
    price_change = ((target_price_7d - current_price) / current_price) * 100
    signal = "BULLISH 🚀" if price_change > 0 else "BEARISH 📉"
    color = 0x3fb950 if price_change > 0 else 0xf85149
    
    description = f"**{TARGET_MODEL}** 모델이 예측한 시장 전망입니다.\n"
    description += f"현재가 대비 7일 후 변동률: **{price_change:+.2f}%**"

    fields = [
        {"name": "💰 Current BTC", "value": f"${current_price:,.0f}", "inline": True},
        {"name": "🎯 Target (7D)", "value": f"${target_price_7d:,.0f}", "inline": True},
        {"name": "🔮 Signal", "value": signal, "inline": True},
        {"name": "😨 Sentiment", "value": f"{df['Fear_Greed_Index'].iloc[-1]:.0f}", "inline": True},
        {"name": "📊 RSI", "value": f"{df['RSI'].iloc[-1]:.1f}", "inline": True},
        {"name": "📈 Nasdaq", "value": f"{df['Nasdaq'].iloc[-1]:,.0f}", "inline": True},
    ]

    # 4. 디스코드 전송
    success, msg = send_discord_message(
        title=f"📅 TOBIT Daily Crypto Report ({datetime.now().strftime('%Y-%m-%d')})",
        description=description,
        fields=fields,
        color=color
    )
    
    if success: print("✅ 리포트 전송 완료!")
    else: print(f"❌ 전송 실패: {msg}")

if __name__ == "__main__":
    run_daily_report()
