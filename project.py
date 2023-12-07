# LSH implementation for python

import math
import os
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import dataset

#TODO Custom Transformer Encoding layer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LSHSelfAttention(nn.Module):
    # TODO: implement this
    
    pass

class LSHEncoderLayer(nn.Module):
    def __init__(d_model: int, nhead: int, d_forward: int, bucket_size: int, num_hashes: int, dropout: float=0.5):
        super().__init__()
        self.lsh_attention = LSHSelfAttention(d_model, nhead, dropout, bucket_size=5, num_hashes=5)
 
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_forward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_forward, d_model)
        ) 

    def forward(self, src, src_mask=None):
        
        # apply the LSH Attention
        src = self.lsh_attention(src, mask=src_mask)
        
        # apply feedforward
        src = self.feed_forward(src)
        
        # TODO: other norms and residual conns?
        
        return src
      
# Use Pretrained word embeddings instead??
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
       super().__init__()
       self.dropout = nn.Dropout(p=dropout)
       
       position = torch.arange(max_len).unsqueeze(1)
       div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
       pe = torch.zeros(max_len, 1, d_model)
       pe[:, 0, 0::2] = torch.sin(position * div_term)
       pe[:, 0, 1::2] = torch.cos(position * div_term)

       # TODO: not learnable pe, but defined as self here
       self.register_buffer('pe', pe)
       
    def forward(self, x: Tensor):
        x = x + self.pe[:x.size(0)]
        return self.dropoput(x)
        
class LSHModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_forward: int, nlayers: int, dropout: float=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # TODO: custom LSH Encoder Layer
        self.encoder_layers = nn.ModuleList([
            LSHEncoderLayer(nhead, d_model, d_forward, dropout)
            for _ in range(nlayers)
        ])
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, nlayers)
        
        self.d_model = d_model
        self.linear = nn.Linear(ntoken, d_model)
        self.init_weights()
       
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: Tensor, src_mask: Tensor=None):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """
            Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
            
        # TODO, edit these encoder layers
        output = self.transformer_encoder(src, src_mask)
        
        output = self.linear(output)
        return output
  
       