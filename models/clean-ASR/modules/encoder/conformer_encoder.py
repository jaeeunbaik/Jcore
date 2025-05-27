import logging
import math

import torch
import torch.nn as nn

from modules.encoder.subsampling import Conv2dSubsampling 
from modules.transformer.attention import RelPositionMultiHeadedAttention
from modules.transformer.embedding import PositionalEncoding, RelPositionalEncoding
from util.utils_module import get_activation, make_non_pad_mask
from .feedforward import FeedForward
from .convolution_module import ConvolutionModule



class ConformerEncoder(nn.Module):
    def __init__(
        self,
        config,  # Accept a single config object
        **kwargs  # Allow additional kwargs for flexibility
    ):
        super().__init__()
        
        # Extract parameters from config
        # Subsampling Module
        self.sampling_in = config.sampling_in
        self.subsampling = config.subsampling
        self.subsampling_factor = config.subsampling_factor
        self.subsampling_channels = config.subsampling_channels
        self.dropout_before_conformer = 0.1
        # Encoder Module
        self.num_blocks = config.num_blocks
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.pos_embed_type = config.pos_embed_type
        self.dropout_rate = config.dropout_rate
        self.ff_expansion_factor = config.ff_expansion_factor
        self.attention_heads = config.attention_heads
        self.linear_units = config.linear_units
        self.cnn_module_kernel = config.cnn_module_kernel
        self.conv_batchnorm = config.conv_batchnorm
        self.normalize_before = config.normalize_before
        self.return_intermediates = config.return_intermediates

        # Rest of your init code
        ff_activation_type = config.activation_type if config.ff_activation_type is None else config.ff_activation_type
        activation = get_activation(ff_activation_type)
        
        self.sampling = Conv2dSubsampling(
            subsampling=self.subsampling, 
            subsampling_factor=self.subsampling_factor,
            feat_in=self.sampling_in,
            feat_out=self.output_size,
            conv_channels=self.subsampling_channels,
            activation=get_activation("relu"),
            is_causal=False,
        )
        self.linear = nn.Linear(self.subsampling_channels, self.output_size)
        if self.pos_embed_type == "relpos":
            self.embed = RelPositionalEncoding(self.output_size, self.dropout_rate)
        elif self.pos_embed_type == "abspos":
            self.embed = PositionalEncoding(self.output_size, self.dropout_rate)
        else:
            assert self.pos_embed_type != "relpos" or "abspos", 'not supported positional embedding'
        self.dropout = nn.Dropout(self.dropout_before_conformer)
        self.return_intermediates = self.return_intermediates
        self.blocks = nn.ModuleList([
            ConformerLayer(
                output_size=self.output_size,
                feedforward=FeedForward(self.output_size, self.ff_expansion_factor, activation, self.dropout_rate),
                mhsa=RelPositionMultiHeadedAttention(
                    self.attention_heads, self.output_size, self.dropout_rate, zero_triu=False),
                convmodule=ConvolutionModule(self.output_size, self.cnn_module_kernel, activation, conv_batchnorm=self.conv_batchnorm)
            )
            for _ in range(self.num_blocks)
        ])
        
        self.layernorm = nn.LayerNorm(self.output_size)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 논문에서 사용한 초기화 방법에 가깝게
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                # 표준 초기화에서 안정성 보장
                fan_in = m.in_channels * m.kernel_size[0]
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    
    def forward(self, xs_pad, lengths):
        """
        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            lengths (torch.Tensor)
        """
        intermediates = []
        # print(f"[DEBUG] Encoder input: shape={xs_pad.shape}, dtype={xs_pad.type}, device={xs_pad.device}, lengths={lengths}")
        xs_pad, lengths = self.sampling(xs_pad, lengths)
        # print(f"[DEBUG] After sampling: shape={xs_pad.shape}, dtype={xs_pad.type}, device={xs_pad.device}")
        # xs_pad = self.linear(xs_pad)  # 이미 sampling layer에 있음 ..
        # print(f"[DEBUG] After pos emb: shape={xs_pad.shape}, dtype={xs_pad.type}, device={xs_pad.device}")
        masks = (make_non_pad_mask(lengths, length_dim=len(lengths))[:, None, :]).to(xs_pad.device)
        xs_pad, pos_emb = self.embed(xs_pad)
        xs_pad = self.dropout(xs_pad)
        for layer_idx, encoder_layer in enumerate(self.blocks):
            xs_pad, masks = encoder_layer(xs_pad, masks, pos_emb)  # xs_pad : [batch, T, output_size(256)]
            # print(f"[DEBUG] {layer_idx + 1}th layer output: shape={xs_pad.shape}, dtype={xs_pad.type}, device={xs_pad.device}")
            if self.return_intermediates:
                intermediates.append(xs_pad)
            
        # print(f"[DEBUG] conformer Just After blocks, shape={xs_pad.shape}")
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        # print(f"[DEBUG] conformer Before layernorm shape={xs_pad.shape}")
        # if self.normalize_before:
        # print(f"[DEBUG] conformer After layernorm shape={xs_pad.shape}")
        # olens = masks.squeeze(1).sum(1)
        # print(f"[DEBUG] conformer encoder output shape={xs_pad.shape}, length = {olens}")
        return xs_pad, masks
    
    
    
class ConformerLayer(nn.Module):
    def __init__(self, output_size, feedforward, mhsa, convmodule):
        super().__init__()
        self.feed_forward = feedforward
        self.mhsa = mhsa
        self.conv_module = convmodule
        self.layernorm = nn.LayerNorm(output_size)
        
    def forward(self, x, mask, pos_emb):
        residual = x
        x = residual + 0.5 * self.feed_forward(x)
        # print(f"[DEBUG] Encoder layer FirstFF: shape={x.shape}, dtype={x.type}, device={x.device}")
        residual = x
        x = self.layernorm(x)
        x = residual + self.mhsa(x, x, x, pos_emb, mask)
        # print(f"[DEBUG] Encoder layer MHSA: shape={x.shape}, dtype={x.type}, device={x.device}")
        residual = x
        x = residual + self.conv_module(x)
        # print(f"[DEBUG] Encoder layer ConvModule: shape={x.shape}, dtype={x.type}, device={x.device}")
        residual = x
        x = residual + 0.5 * self.feed_forward(x)
        x = self.layernorm(x) 
        # print(f"[DEBUG] Encoder layer LastFF: shape={x.shape}, dtype={x.type}, device={x.device}")
        
        return x, mask
