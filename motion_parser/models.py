import os
import numpy as np
import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe.view([-1, self.max_len, self.d_model])[0]
        x = x + pe[:x.size(0)]
        return self.dropout(x)

class SoftAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super(SoftAttention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.query_proj = nn.Linear(d_in, d_out, bias=False)
        self.key_proj = nn.Linear(d_in, d_out, bias=False)
        self.value_proj = nn.Linear(d_in, d_out, bias=False)
        self.d_model = d_out

    def forward(self, query, key, value):
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.d_model)
        key = self.key_proj(key).view(batch_size, -1, self.d_model)
        value = self.value_proj(value).view(batch_size, -1, self.d_model)

        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.d_out)
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        context = context.view(1, batch_size, self.d_model)
        context = context.permute(1, 0, 2).contiguous().view(batch_size, -1)

        return context, attn
    
    def gen_beta(self, x):
        context, attn = self.forward(x, x, x)
        return context

class HardAttention(nn.Module):
    def __init__(self, d_in, d_out, keep_percent = 1/2):
        super(HardAttention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.query_proj = nn.Linear(d_in, d_out, bias=False)
        self.key_proj = nn.Linear(d_in, d_out, bias=False)
        self.value_proj = nn.Linear(d_in, d_out, bias=False)
        self.d_model = d_out
        self.keep_percent = keep_percent
        self.score_dropout = nn.Dropout(p=(1-keep_percent))

    def forward(self, query, key, value):
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.d_model)
        key = self.key_proj(key).view(batch_size, -1, self.d_model)
        value = self.value_proj(value).view(batch_size, -1, self.d_model)

        score = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(self.d_out)
        # filter top percentage of the score
        keep_num = int(batch_size * self.keep_percent)
        filter_threshhold = torch.topk(score.view(-1), keep_num).values.min()
        score = torch.where(score>filter_threshhold, score, 0)
        # filter by reandom
        # score = self.score_dropout(score)

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        context = context.view(1, batch_size, self.d_model)
        context = context.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return context, attn

    def gen_beta(self, x):
        context, attn = self.forward(x, x, x)
        return context

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads=1, dropout=0.0, keep_percent=1/2, device=None):
        super(SelfAttention, self).__init__()

        self.W = nn.Linear(d_in, d_out, bias=False)
        # self.attention = nn.MultiheadAttention(embed_dim = d_out, num_heads = num_heads, device = device)
        # self.attention = SoftAttention(d_out, d_out)
        self.attention = HardAttention(d_out, d_out, keep_percent=keep_percent)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.attention(self.W(x), self.W(x), self.W(x))[0]
        out = x
        return out

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers=6, nhead=8, device=None):
        super(Encoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, d_out, num_layers=6, nhead=8, device=None):
        super(Decoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, d_out)
    
    def forward(self, x, state):
        x = self.transformer_decoder(x, state)
        x = self.decoder(x)
        return x

class MotionParser(nn.Module):
    def __init__(self, d_motion, d_control, d_model=64, num_heads=1, dropout=0.0, device=None):
        super(MotionParser, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_motion, dropout)
        self.self_attention = SelfAttention(d_motion, d_model, keep_percent=1/2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3)
        self.decoder = Decoder(d_model, d_control, num_layers=3)
        self.dropout = nn.Dropout(dropout)

        self.device = device

    def parse_data(self, x):
        x = self.positional_encoding(x)
        x = self.self_attention(x)
        return x
    
    def get_beta(self, x, i):
        return self.parse_beta(x)[i, :]

    def forward(self, x):
        x = self.parse_data(x)
        x1 = self.encoder(x)
        nn.Dropout(p=0.5)
        x1 = self.decoder(x, x1)
        return x1