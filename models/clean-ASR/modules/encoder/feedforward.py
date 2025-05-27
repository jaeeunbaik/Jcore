import torch
import torch.nn as nn

from .swish import Swish


class FeedForward(nn.Module):
    def __init__(
        self,
        output_size,
        ff_expansion_factor=4, 
        activation=Swish(),
        dropout=0.1,
    ):
        super().__init__()
        self.output_size = output_size
        self.ff_expansion_factor = 4
        self.activation = activation
        self.Layernorm = nn.LayerNorm(self.output_size)
        self.linear1 = nn.Linear(output_size, output_size * ff_expansion_factor)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(output_size * ff_expansion_factor, output_size)
        
    def forward(self, x):
        res = x
        x = self.Layernorm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        # print(f"[DEBUG] FeedForward inside: shape={x.shape}, dtype={x.type}, device={x.device}")
        x = self.linear2(x)
        x = self.dropout(x)
        return x + res