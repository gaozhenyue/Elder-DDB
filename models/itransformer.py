import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np



class itransformerModel(nn.Module):
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
                 factor=1, n_heads=8, d_ff=2048, activation='gelu', e_layers=2, enc_in=53, num_class=1):
        super(itransformerModel, self).__init__()
        self.seq_len = seq_len
        self.output_attention = output_attention
        # Embedding  数据嵌入
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)  # seq_len 时间长度；d_model 时间维度嵌入后的维度
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
        self.projection = nn.Linear(d_model * enc_in, num_class)


    
    def classification(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)  # 数据嵌入
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # 嵌入数据送入transformer编码器

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        # output = torch.sigmoid(output)
        return output


    
    def forward(self, x_enc, mask=None):
        dec_out = self.classification(x_enc)
        return dec_out  # [B, N]
