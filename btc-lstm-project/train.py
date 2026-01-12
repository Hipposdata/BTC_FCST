import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.preprocessing import StandardScaler

# data_utils에서 TICKERS와 필요한 함수들을 가져옵니다.
from data_utils import fetch_multi_data, create_sequences, save_scaler, TICKERS 
from model import LSTMModel

# 1. 데이터 준비 및 피처 정의
df = fetch_multi_data()

# TICKERS의 키값들을 리스트로 변환하여 'features'를 정의합니다.
# ['Bitcoin', 'DXY', 'Nasdaq', 'S&P500', 'US_10Y', 'Gold', 'VIX', 'WTI_Oil']
features = list(TICKERS.keys()) 

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features]) # 정의된 features 사용
save_scaler(scaler)

# 2. 하이퍼파라미터 설정
seq_length = 120 # 과거 120일 데이터 사용
prediction_days = 7 # 미래 7일 가격 예측
btc_index = features.index('Bitcoin') # 비트코인 열 번호 찾기

# 시퀀스 생성
X, y = create_sequences(
    scaled_data, 
    seq_length, 
    prediction_days=prediction_days, 
    target_col_idx=btc_index
)

X_train = torch.tensor(X).float()
y_train = torch.tensor(y).float() # (샘플 수, 7) 형태

# 3. 모델 설정
# len(features)는 8이 됩니다.
model = LSTMModel(input_size=len(features), output_size=prediction_days)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 학습 루프
epochs = 50
batch_size = 64
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# 5. 저장
os.makedirs('weights', exist_ok=True)
torch.save(model.state_dict(), 'weights/model.pth')
print("모델 및 스케일러 저장 완료!")
