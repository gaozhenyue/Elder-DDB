import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, num_class=1):
        """
        LSTM模型初始化
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(LSTMModel, self).__init__()
        self.num_class = num_class
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, self.num_class)
        
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 只取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        
        # Dropout和全连接层
        out = self.dropout(last_out)
        out = self.fc(out)

        if self.num_class == 1:
            out = torch.sigmoid(out)
        
        return out