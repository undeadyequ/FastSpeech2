import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols

class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()

        txt_dim = config["txt_dim"]
        nhead = config["nhead"]
        dropout = config["dropout"]
        style_dim = config["style_dim"]
        self.self_attn = nn.MultiheadAttention(txt_dim, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(style_dim, nhead, dropout=dropout)

    def forward(self, phoneme, style_emb):
        """
        Args:
            phoneme:  (N. L, E1)
            style_emb: (N, E2)
        Returns:

        """
        phoneme = phoneme.transpose(1, 0)

        phone2, txt_attn = self.self_attn(phoneme, phoneme, phoneme)
        phoneme = phoneme + phone2
        style_emb = style_emb.unsqueeze(1)
        style_emb = style_emb.transpose(1, 0)
        style_emb2, style_attn = self.cross_attn(phone2, style_emb, style_emb)
        phoneme = phoneme + style_emb2

        phoneme = phoneme.transpose(1, 0)
        return phoneme

