import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_utils import fetch_multi_data, create_sequences, save_scaler, TICKERS
from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN, MLP

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ 학습 장치: {device}")

# =============================================================================
# 1. 데이터 준비 및 전처리 (Data Leakage 방지 적용)
# =============================================================================
print("데이터 다운로드 및 전처리 중...")
df = fetch_multi_data()
features = list(TICKERS.keys())
btc_idx = features.index('BTC_Close') # 또는 'Bitcoin' (data_utils.py 설정에 따름)

# [핵심 수정] 스케일링 데이터 누수 방지 로직
# 전체 데이터로 fit을 하면 미래 정보(평균, 분산)가 반영되므로,
# 과거 데이터(앞쪽 90%)로만 기준을 잡고(fit), 변환(transform)은 전체에 대해 수행합니다.
train_split_idx = int(len(df) * 0.9)
train_data_for_scaler = df[features].iloc[:train_split_idx]

scaler = StandardScaler()
scaler.fit(train_data_for_scaler) # ⚠️ 과거 데이터로만 학습!
scaled_data = scaler.transform(df[features]) # 변환은 전체 데이터 적용
save_scaler(scaler)

input_dim = len(features)
prediction_days = 7

# =============================================================================
# 2. 모델 설정 (14, 21, 45일 멀티 시퀀스)
# =============================================================================
# 다양한 관점(단기/중기/장기)을 학습하기 위한 시퀀스 길이 목록
SEQ_LENGTHS = [14, 21, 45]
model_names = ["MLP", "DLinear", "TCN", "LSTM", "PatchTST", "iTransformer"]

# 모델 인스턴스 생성 팩토리 함수
def get_model_instance(name, seq_len):
    if name == "MLP": 
        return MLP(seq_len=seq_len, input_size=input_dim, pred_len=prediction_days, hidden_sizes=[256, 128], dropout=0.1)
    elif name == "DLinear": 
        return DLinear(seq_len=seq_len, pred_len=prediction_days, input_size=input_dim, kernel_size=25)
    elif name == "TCN": 
        return TCN(input_size=input_dim, output_size=prediction_days, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2)
    elif name == "LSTM": 
        return LSTMModel(input_size=input_dim, output_size=prediction_days)
    elif name == "PatchTST": 
        return PatchTST(input_size=input_dim, seq_len=seq_len, pred_len=prediction_days,
                        patch_len=7, stride=3, d_model=64, n_heads=4, n_layers=2, dropout=0.2)
    elif name == "iTransformer": 
        return iTransformer(seq_len=seq_len, pred_len=prediction_days, input_size=input_dim,
                            d_model=256, n_heads=4, n_layers=3, dropout=0.2)
    return None

# =============================================================================
# 3. 학습 루프 (시퀀스 길이 x 모델 종류)
# =============================================================================
os.makedirs('weights', exist_ok=True)
batch_size = 64

print(f"🚀 총 {len(SEQ_LENGTHS) * len(model_names)}개의 모델 학습이 진행됩니다.")

for seq_len in SEQ_LENGTHS:
    print(f"\n{'='*50}")
    print(f"📅 Lookback Window: {seq_len}일 데이터셋 생성")
    print(f"{'='*50}")
    
    # 해당 길이에 맞는 시퀀스 데이터 생성
    # (주의: 학습은 '전체 데이터'를 사용하여 최신 경향까지 반영합니다)
    X, y = create_sequences(scaled_data, seq_len, prediction_days=prediction_days, target_col_idx=btc_idx)
    X_train, y_train = torch.tensor(X).float(), torch.tensor(y).float()
    
    for name in model_names:
        # 모델별 최적 하이퍼파라미터 적용 (Epochs, LR)
        if name == "TCN":
            epochs, lr = 200, 0.005
        elif name in ["PatchTST", "iTransformer"]:
            epochs, lr = 150, 0.001
        else:
            epochs, lr = 100, 0.005 # MLP, LSTM, DLinear 등

        print(f"🚀 [{name}] (Seq: {seq_len}) 학습 시작... (Epochs: {epochs})")
        
        # 모델 초기화 & GPU 이동
        model = get_model_instance(name, seq_len)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 학습 수행
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(X_train.size()[0])
            epoch_loss = 0
            
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices].to(device), y_train[indices].to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 기울기 폭발 방지
                optimizer.step()
                epoch_loss += loss.item()
                
            # 로그 출력 (진행 상황 확인용)
            if (epoch + 1) % 50 == 0:
                avg_loss = epoch_loss / (len(X_train) / batch_size)
                print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

        # 가중치 파일 저장 (파일명 형식: 모델명_시퀀스길이.pth)
        save_path = f'weights/{name}_{seq_len}.pth'
        torch.save(model.cpu().state_dict(), save_path)
        print(f"✅ 저장 완료: {save_path}")

print("\n🎉 모든 학습이 완료되었습니다!")
