import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# 필수 라이브러리 임포트
try:
    from data_utils import fetch_multi_data, load_scaler, send_discord_message, TICKERS
    from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN, MLP
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 설정 (백테스팅 로직 반영)
# ---------------------------------------------------------
TARGET_MODEL = "DLinear"  # 성능 좋은 모델 선택
SEQ_LEN = 14
PRED_LEN = 7

# [중요] 매매 임계값 (백테스팅 기준: 5%)
# 예상 수익률이 이 값보다 커야 '매수' 신호를 보냅니다.
THRESHOLD = 5.0 

# 환경변수 로드
FRED_API_KEY = os.getenv("FRED_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# ---------------------------------------------------------
# 모델 로드 함수 (Headless)
# ---------------------------------------------------------
def get_model_headless(name, seq_len):
    input_size = len(TICKERS)
    
    if name == "MLP": model = MLP(seq_len=seq_len, input_size=input_size, pred_len=PRED_LEN)
    elif name == "DLinear": model = DLinear(seq_len=seq_len, pred_len=PRED_LEN, input_size=input_size, kernel_size=25)
    elif name == "TCN": model = TCN(input_size=input_size, output_size=PRED_LEN, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2)
    elif name == "LSTM": model = LSTMModel(input_size=input_size, output_size=PRED_LEN)
    elif name == "PatchTST": model = PatchTST(input_size=input_size, seq_len=seq_len, pred_len=PRED_LEN, patch_len=7, stride=3, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    elif name == "iTransformer": model = iTransformer(seq_len=seq_len, pred_len=PRED_LEN, input_size=input_size, d_model=256, n_heads=4, n_layers=3, dropout=0.2)
    else: return None

    weights_path = os.path.join("weights", f"{name}_{seq_len}.pth")
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print(f"✅ {name} 가중치 로드 성공")
        except:
            print(f"⚠️ {name} 가중치 로드 실패 (초기화 상태)")
    else:
        print(f"⚠️ 가중치 파일 없음: {weights_path}")
        
    model.eval()
    return model

# ---------------------------------------------------------
# 메인 실행 로직
# ---------------------------------------------------------
def run_daily_report():
    print("🚀 Daily Report Bot 시작 (백테스팅 로직 적용)...")

    if not DISCORD_WEBHOOK_URL:
        print("❌ Discord Webhook URL 없음.")
        return

    # 1. 데이터 수집
    df = fetch_multi_data()
    if df.empty:
        send_discord_message("🚨 TOBIT Error", "데이터 수집 실패")
        return

    # 2. 전처리
    scaler = load_scaler()
    features = list(TICKERS.keys())
    try: btc_idx = features.index('BTC_Close')
    except: btc_idx = 0

    # 3. 모델 예측
    model = get_model_headless(TARGET_MODEL, SEQ_LEN)
    input_raw = df[features].tail(SEQ_LEN).values
    input_tensor = torch.tensor(scaler.transform(input_raw)).float().unsqueeze(0)
    
    with torch.no_grad():
        preds_scaled = model(input_tensor).numpy()[0]
    
    # 7일 뒤 가격 역변환
    dummy = np.zeros(len(features))
    dummy[btc_idx] = preds_scaled[-1]
    target_price_7d = scaler.inverse_transform(dummy.reshape(1, -1))[0][btc_idx]
    
    current_price = df['BTC_Close'].iloc[-1]
    
    # [핵심] 수익률 계산 및 신호 결정 (5% 룰 적용)
    roi_pct = ((target_price_7d - current_price) / current_price) * 100
    
    if roi_pct >= THRESHOLD:
        signal = "🔥 STRONG BUY (적극 매수)"
        action_msg = f"예상 수익률({roi_pct:.2f}%)이 기준({THRESHOLD}%)을 초과했습니다."
        color = 0x3fb950 # 초록색
    elif roi_pct <= -THRESHOLD:
        signal = "📉 STRONG SELL (적극 매도)"
        action_msg = f"예상 손실률({roi_pct:.2f}%)이 기준(-{THRESHOLD}%)을 초과했습니다."
        color = 0xf85149 # 빨간색
    else:
        signal = "✋ HOLD (관망)"
        action_msg = f"변동폭({roi_pct:.2f}%)이 기준({THRESHOLD}%) 이내입니다. 관망을 추천합니다."
        color = 0xffeb3b # 노란색

    # 4. 디스코드 메시지 전송
    description = f"**[{TARGET_MODEL}] 모델 분석 리포트**\n"
    description += f"현재가: **${current_price:,.0f}** → 7일 뒤: **${target_price_7d:,.0f}**\n\n"
    description += f"💡 **투자 판단:** {action_msg}"

    fields = [
        {"name": "🔮 Signal", "value": signal, "inline": False},
        {"name": "📈 Exp. Return (7D)", "value": f"{roi_pct:+.2f}%", "inline": True},
        {"name": "⚙️ Threshold", "value": f"±{THRESHOLD}%", "inline": True},
        {"name": "😨 Sentiment", "value": f"{df['Fear_Greed_Index'].iloc[-1]:.0f}", "inline": True},
        {"name": "📊 RSI (14)", "value": f"{df['RSI'].iloc[-1]:.1f}", "inline": True},
    ]

    success, msg = send_discord_message(
        title=f"📅 TOBIT Daily Strategy ({datetime.now().strftime('%Y-%m-%d')})",
        description=description,
        fields=fields,
        color=color
    )
    
    if success: print("✅ 리포트 전송 완료!")
    else: print(f"❌ 전송 실패: {msg}")

if __name__ == "__main__":
    run_daily_report()
