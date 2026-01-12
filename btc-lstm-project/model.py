import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    # output_size를 7로 기본값 설정
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, output_size=7, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size) # 최종 출력이 7이 됨

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # 마지막 시점의 hidden state에서 7일치 예측
        return out
