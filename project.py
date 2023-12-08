# LSH implementation for python

import math
import os
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import dataset

#TODO Custom Transformer Encoding layer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
        
class LSHSelfAttention(nn.Module):
    # TODO: implement this
    def __init__(self, d_model: int, num_heads: int, dropout: int, bucket_size: int=5, num_hashes: int=5):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.bucket_size = bucket_size 
        self.num_hashes = num_hashes
        self.d_k = d_model // num_heads
        
        self.query_projection = nn.Linear(d_model, self.d_k * num_heads)
        self.key_projection = nn.Linear(d_model, self.d_k * num_heads)
        self.value_projection = nn.Linear(d_model, self.d_k * num_heads)
        
        #output
        self.out_projection = nn.Linear(self.d_k * num_heads, d_model)

        
        self.attention_dropout = nn.Dropout(dropout)
       
    def forward(self, query, key, value, mask=None):
        # Step 1: Apply query/key/value projections
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        # Step 2: Hash queries and keys into buckets
        # Implement the hashing function here
        # ...

        # Step 3: Perform attention within each bucket
        # You might need to sort or reorder queries/keys/values based on the hash
        # ...

        # Step 4: Apply attention to the values
        # This will include calculating attention scores, applying dropout, and combining the results
        # ...

        # Step 5: Concatenate results from different heads (if multiple rounds of hashing)
        # ...

        # Step 6: Apply final output projection
        output = self.out_projection(attention_output)

        return output
         
        

class LSHEncoderLayer(nn.Module):
    def __init__(d_model: int, num_heads: int, d_forward: int, bucket_size: int, num_hashes: int, dropout: float=0.5):
        super().__init__()
        self.lsh_attention = LSHSelfAttention(d_model, num_heads, dropout) 
 
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_forward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_forward, d_model)
        )

    def forward(self, src, src_mask=None):
        src = self.lsh_attention(src, mask=src_mask)
        src = self.feed_forward(src)
        
        # TODO: other norms and residual conns?
        
        return src
class LSHModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, num_heads: int, d_forward: int, nlayers: int, dropout: float=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # TODO: custom LSH Encoder Layer
        self.encoder_layers = nn.ModuleList([
            LSHEncoderLayer(num_heads, d_model, d_forward, dropout)
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
  
       