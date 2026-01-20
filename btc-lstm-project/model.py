import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. MLP
class MLP(nn.Module):
    def __init__(self, seq_len, input_size, pred_len=7, hidden_sizes=[256, 128], dropout=0.1):
        super(MLP, self).__init__()
        flatten_dim = seq_len * input_size
        self.net = nn.Sequential(
            nn.Linear(flatten_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], pred_len)
        )

    def forward(self, x):
        # [Batch, Seq, Feature] -> [Batch, Seq * Feature]
        x = x.reshape(x.size(0), -1)
        return self.net(x)

# 2. DLinear
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len=14, pred_len=7, input_size=8, kernel_size=25):
        super(DLinear, self).__init__()
        self.decomposition = SeriesDecomp(kernel_size=kernel_size)
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq, Feature] -> [Batch, Feature, Seq]
        x = x.permute(0, 2, 1)
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_output = self.seasonal(seasonal_init)
        trend_output = self.trend(trend_init)
        out = seasonal_output + trend_output
        return out[:, 0, :]

# 3. TCN (요청 구조 반영: channels=[64,64,64], dropout=0.2)
class TCN(nn.Module):
    def __init__(self, input_size=8, output_size=7, num_channels=[64, 64, 64], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers += [
                nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                               stride=1, padding=padding, dilation=dilation_size)),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: [Batch, Seq, Feature] -> [Batch, Feature, Seq]
        x = x.permute(0, 2, 1)
        y = self.network(x)
        if self.network[0].padding[0] > 0:
             y = y[:, :, :-self.network[0].padding[0]]
        return self.fc(y[:, :, -1])

# 4. LSTM
# 4. Stacked LSTM (단방향, 여러 층)
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, output_size=7, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)         # out: [B, T, H]
        out = out[:, -1, :]           # 마지막 시점
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    
# 5. PatchTST (요청 구조 반영: patch=7, stride=3, d_model=64, heads=4, layers=2)
class PatchTST(nn.Module):
    def __init__(self, input_size=8, seq_len=14, pred_len=7, 
                 patch_len=7, stride=3, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super(PatchTST, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_size = input_size
        
        # 패치 개수 계산
        self.patch_num = int((seq_len - patch_len) / stride) + 1
        
        # Patch Embedding: [Batch, Var, Patch_Num, Patch_Len] -> [Batch, Var, Patch_Num, d_model]
        self.patch_embedding = nn.Linear(patch_len, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                   dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Final Projection
        self.head = nn.Linear(d_model * self.patch_num, pred_len)

    def forward(self, x):
        # x: [Batch, Seq, Feature]
        # 1. Unfold for Patching
        # [Batch, Seq, Feature] -> [Batch, Feature, Seq]
        x = x.permute(0, 2, 1)
        
        # Unfold: [Batch, Feature, Patch_Num, Patch_Len]
        # unfold dimension is 2 (Seq axis)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # 2. Embedding
        # [Batch, Feature, Patch_Num, d_model]
        enc_out = self.patch_embedding(patches)
        
        # 3. Transformer Encoder (Channel Independent)
        # Reshape to [Batch * Feature, Patch_Num, d_model] for transformer
        B, F, P, D = enc_out.shape
        enc_out = enc_out.reshape(B * F, P, D)
        
        # Encoder Output: [Batch * Feature, Patch_Num, d_model]
        enc_out = self.encoder(enc_out)
        
        # 4. Flatten & Projection
        # [Batch * Feature, Patch_Num * d_model]
        enc_out = enc_out.reshape(B, F, -1)
        
        # [Batch, Feature, Pred_Len]
        out = self.head(enc_out)
        
        # BTC (idx 0) 예측값만 반환 [Batch, Pred_Len]
        return out[:, 0, :]

# 6. iTransformer (요청 구조 반영: d_model=256, heads=4, layers=3)
class iTransformer(nn.Module):
    def __init__(self, seq_len=14, pred_len=7, input_size=8, 
                 d_model=256, n_heads=4, n_layers=3, dropout=0.2):
        super(iTransformer, self).__init__()
        # Inverted: 임베딩이 Time Step 전체를 하나의 토큰으로 봄
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                   dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [Batch, Seq, Feature] -> [Batch, Feature, Seq]
        x = x.permute(0, 2, 1)
        
        # Embedding: [Batch, Feature, d_model]
        enc_out = self.enc_embedding(x)
        
        # Transformer: [Batch, Feature, d_model]
        enc_out = self.encoder(enc_out)
        
        # Projection: [Batch, Feature, Pred_Len]
        out = self.projection(enc_out)
        
        # BTC (idx 0)만 반환
        return out[:, 0, :]
