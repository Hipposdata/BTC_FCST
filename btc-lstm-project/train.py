import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import fetch_btc_ohlcv, create_sequences, save_scaler
from sklearn.preprocessing import MinMaxScaler



import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LSTMModel  # 기존 코드


# 1. 데이터 준비
df = fetch_btc_ohlcv()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['close']])
save_scaler(scaler) # 중요: 추론에서 사용

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
X_train = torch.tensor(X).float()
y_train = torch.tensor(y).float()

# 2. 모델 및 학습 설정
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 학습 루프
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# 4. 저장
torch.save(model.state_dict(), 'weights/model.pth')
print("Model and Scaler saved!")