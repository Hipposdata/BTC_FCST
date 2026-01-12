import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=3, output_size=7):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# 2. DLinear (Decomposition Linear)
class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len=120, pred_len=7, input_size=8):
        super(DLinear, self).__init__()
        self.decompsition = series_decomp(kernel_size=25)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        return (seasonal_output + trend_output)[:, 0, :]

# 3. PatchTST (Simplified)
class PatchTST(nn.Module):
    def __init__(self, input_size=8, seq_len=120, pred_len=7, patch_len=16, stride=8):
        super(PatchTST, self).__init__()
        self.patch_num = int((seq_len - patch_len) / stride + 1)
        self.linear_patch = nn.Linear(patch_len, 64)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True), num_layers=2)
        self.fc = nn.Linear(64 * self.patch_num, pred_len)

    def forward(self, x):
        x = x[:, :, 0]
        patches = x.unfold(dimension=1, size=16, step=8)
        x = self.linear_patch(patches)
        x = self.transformer_encoder(x)
        return self.fc(x.view(x.size(0), -1))

# 4. iTransformer
class iTransformer(nn.Module):
    def __init__(self, seq_len=120, pred_len=7, input_size=8, d_model=128):
        super(iTransformer, self).__init__()
        self.enc_embedding = nn.Linear(seq_len, d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True), num_layers=2)
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        enc_out = self.enc_embedding(x)
        return self.projection(self.encoder(enc_out))[:, 0, :]

# 5. TCN (Temporal Convolutional Network)
class TCN(nn.Module):
    def __init__(self, input_size=8, output_size=7, num_channels=[64, 64], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                                     stride=1, padding=(kernel_size-1) * dilation_size, dilation=dilation_size)),
                       nn.ReLU(), nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: [Batch, Seq, Feature] -> [Batch, Feature, Seq]
        x = x.permute(0, 2, 1)
        y1 = self.network(x)
        return self.fc(y1[:, :, -1])
