import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PositionalEmbedding
import numpy as np


class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(LayerNormLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        return lstm_out


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, lstm_layers = 2):
        super(DataEmbedding_inverted, self).__init__()
        self.lstm = LayerNormLSTM(input_size=1, hidden_size=d_model, num_layers=lstm_layers, dropout=dropout)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        batchsize, time, c_in = x.size()
        # x_mark = (batch_size, max_len, 1) 
        
        # 将输入reshape成(batchsize * c_in, time, 1)
        x = x.permute(0, 2, 1).contiguous().view(batchsize * c_in, time, 1)

        lstm_out = self.lstm(x)

        if x_mark is None:
            # 取最后一个时间步的输出
            lstm_out = lstm_out[:, -1, :]  # [batchsize * c_in, d_model]
        else:
            x_mark = x_mark.repeat(1, 1, c_in)
            x_mark = x_mark.permute(0, 2, 1).contiguous().view(batchsize * c_in, time, 1)
            # 取每个序列最后一个非padding位置的输出
            lengths = x_mark.squeeze(-1).sum(dim=1).long()  # 各序列实际长度
            lstm_out = lstm_out[torch.arange(batchsize* c_in), lengths-1, :]  # (batch_size, hidden_size)

        # reshape回(batchsize, c_in, d_model)
        lstm_out = lstm_out.view(batchsize, c_in, -1)
        lstm_out = lstm_out + self.position_embedding(lstm_out)
        return self.dropout(lstm_out)



class lstm_itransformerModel(nn.Module):
    """
    seq_len: 输入序列长度，默认24
    output_attention：是否在编码中输出attention，默认Fasle
    d_model：模型的嵌入维度
    embed：时间特征编码, options:[timeF, fixed, learned]，默认 timeF
    freq： freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]，默认 h
    factor：attn factor，默认是1
    enc_in：encoder input size， 默认是7
    n_heads：num of heads，默认是8
    d_ff：dimension of fcn，默认是2048
    e_layers：num of encoder layers，默认是2
    """

    def __init__(self, seq_len=24, output_attention=0, d_model=512, embed='timeF', freq='h', dropout=0.1, 
                 factor=1, n_heads=8, d_ff=2048, activation='gelu', e_layers=2, enc_in=53, lstm_layers=1, num_class=1):
        super(lstm_itransformerModel, self).__init__()
        self.seq_len = seq_len
        self.output_attention = output_attention
        self.num_class = num_class
        # Embedding  数据嵌入
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout, lstm_layers)  # seq_len 时间长度；d_model 时间维度嵌入后的维度
        # Encoder 构建编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        # self.projection = nn.Linear(d_model * enc_in, num_class)

        if isinstance(self.num_class, list):
            self.projection = nn.ModuleList()
            for i in range(len(self.num_class)):
                branch = nn.Linear(d_model * enc_in, self.num_class[i])
                self.projection.append(branch)
        else:   
            self.projection = nn.Linear(d_model * enc_in, self.num_class)


    
    def classification(self, x_enc, mask=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, mask)  # 数据嵌入
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # 嵌入数据送入transformer编码器

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        # output = self.projection(output)  # (batch_size, num_classes)
        # output = torch.sigmoid(output)

        if isinstance(self.num_class, list):
            outputs = []
            for branch in self.projection:
                outputs.append(branch(output))
        else:
            outputs = self.projection(output)  # (batch_size, num_classes)
        return outputs


    
    def forward(self, x_enc, mask=None):
        dec_out = self.classification(x_enc, mask)
        return dec_out  # [B, N]
