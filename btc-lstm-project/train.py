import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_utils import fetch_multi_data, create_sequences, save_scaler, TICKERS
from model import LSTMModel, DLinear, PatchTST, iTransformer, TCN

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ 학습 장치: {device}")

# 1. 데이터 준비
df = fetch_multi_data()
features = list(TICKERS.keys())
btc_idx = features.index('Bitcoin')
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])
save_scaler(scaler)

seq_length, prediction_days = 120, 7
X, y = create_sequences(scaled_data, seq_length, prediction_days=prediction_days, target_col_idx=btc_idx)
X_train, y_train = torch.tensor(X).float(), torch.tensor(y).float()

# 2. 모델 리스트
models = {
    "LSTM": LSTMModel(input_size=len(features), output_size=prediction_days),
    "DLinear": DLinear(seq_len=seq_length, pred_len=prediction_days, input_size=len(features)),
    "PatchTST": PatchTST(input_size=len(features), seq_len=seq_length, pred_len=prediction_days),
    "iTransformer": iTransformer(seq_len=seq_length, pred_len=prediction_days, input_size=len(features)),
    "TCN": TCN(input_size=len(features), output_size=prediction_days)
}

# 3. 학습 루프
os.makedirs('weights', exist_ok=True)
epochs, batch_size = 50, 64

for name, model in models.items():
    print(f"\n🚀 {name} 학습 시작...")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 안정성 확보
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"[{name}] Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/(len(X_train)/batch_size):.6f}")

    torch.save(model.cpu().state_dict(), f'weights/{name}.pth')
    print(f"✅ {name} 저장 완료")
