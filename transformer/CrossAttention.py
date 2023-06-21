import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from .Layers import FFTBlock
from text.symbols import symbols

import torch.nn as nn


class CrossAttention(nn.Module):
    def __int__(self, config):
        self(CrossAttention, self).__init__()
        txt_dim = config[""]
        nhead = config[""]
        dropout = config[""]
        style_dim = config[""]
        self.self_attn = nn.MultiheadAttention(txt_dim, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(style_dim, nhead, dropout=dropout)

    def froward(self, phoneme, style_emb):
        """
        Args:
            phoneme:  (L, N, E1)
            style_emb: (N, E2)
        Returns:

        """
        phone2, txt_attn = self.self_attn(phoneme, phoneme, phoneme)
        phoneme = phoneme + phone2

        ## attn_mask and key_padding_mask ???
        style_emb = style_emb.unsqueeze(0).expand(phoneme.size(0), -1, -1)
        style_emb2, style_attn = self.cross_attn(phone2, style_emb, style_emb)
        phoneme = phoneme + style_emb2
        return phoneme