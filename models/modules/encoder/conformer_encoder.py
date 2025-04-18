import logging
import torch
import torch.nn as nn

from modules.encoder.subsampling import Conv2dSubsampling 
from modules.transformer.attention import RelPositionMultiHeadedAttention
from modules.transformer.embedding import PositionalEncoding, RelPositionalEncoding
from util.utils_module import get_activation, make_pad_mask
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
        
        if self.pos_embed_type == "relpos":
            self.embed = RelPositionalEncoding(self.output_size, self.dropout_rate)
        elif self.pos_embed_type == "abspos":
            self.embed = PositionalEncoding(self.output_size, self.dropout_rate)
        else:
            assert self.pos_embed_type != "relpos" or "abspos", 'not supported positional embedding'
        
        self.return_intermediates = self.return_intermediates
        self.blocks = nn.ModuleList([
            ConformerLayer(
                output_size=self.output_size,
                feedforward=FeedForward(self.output_size, self.ff_expansion_factor, activation, self.dropout_rate),
                mhsa=RelPositionMultiHeadedAttention(
                    self.attention_heads, self.output_size, self.dropout_rate, zero_triu=False),
                convmodule=ConvolutionModule(self.output_size, self.cnn_module_kernel, activation)
            )
            for _ in range(self.num_blocks)
        ])
        
        self.layernorm = nn.LayerNorm(self.output_size)
    
    
    def forward(self, xs_pad, lengths):
        """
        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            lengths (torch.Tensor)
        """
        intermediates = []
        xs_pad, lengths = self.sampling(xs_pad, lengths)
        xs_pad, pos_emb = self.embed(xs_pad)
        masks = (~make_pad_mask(lengths)[:, None, :]).to(xs_pad.device)
        
        for layer_idx, encoder_layer in enumerate(self.blocks):
            xs_pad, masks = encoder_layer(xs_pad, masks, pos_emb)
            if self.return_intermediates:
                intermediates.append(xs_pad)
            
            
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        
        if self.normalize_before:
            xs_pad = self.layernorm(xs_pad)
        
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens
    
    
    
class ConformerLayer(nn.Module):
    def __init__(self, output_size, feedforward, mhsa, convmodule):
        super().__init__()
        self.feed_forward = feedforward
        self.mhsa = mhsa
        self.conv_module = convmodule
        self.layernorm = nn.LayerNorm(output_size)
        
    def forward(self, x, mask, pos_emb):
        residual = x
        x = self.layernorm(x)
        x = residual + 0.5 * self.feed_forward(x)
        residual = x
        x = self.layernorm(x)
        x = residual + self.mhsa(x, x, x, pos_emb, mask)
        
        residual = x
        x = self.layernorm(x)
        x = residual + self.conv_module(x)
        
        residual = x
        x = self.layernorm(x)
        x = residual + 0.5 * self.feed_forward(x)
        
        
        return x, mask
