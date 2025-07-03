"""
🖤🐰 JaeEun Baik, 2025
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from typing import Union


class Predictor(torch.nn.Module):
    '''
    Args
        n_layers: num of rnn layer
        embed_dim: hidden size of rnn layer
        output_dim: output size coincident with encoder output dimension
    '''
    def __init__(self, 
                 n_layers=1, 
                 embed_dim=256,
                 hidden_dim=640,
                 output_dim=256,
                 num_embeddings=5000,
                 layer_type='lstm',
                 embed_dropout_rate=0.1,
                 rnn_dropout_rate=0.1):
        super(Predictor, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layer_type = layer_type
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=output_dim
        )
        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        if self.layer_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.embed_dim,
                hidden_size=self.hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_dropout_rate,
            )
        elif self.layer_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.embed_dim,
                hidden_size=self.hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_dropout_rate
            )
        self.output_linear = nn.Linear(self.hidden_dim, output_dim)
        
    def forward(
        self,
        y: torch.Tensor,
        y_lengths: Optional[torch.Tensor] = None,
        states: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None, 
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the Predictor network.
        """
        embedded = self.embed(y)
        embedded = self.embed_dropout(embedded)
        
        if y_lengths is not None:
            y_lengths_cpu = y_lengths.to("cpu", dtype=torch.int64)
            
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, y_lengths_cpu, batch_first=True, enforce_sorted=False
            )
            
            if self.layer_type == 'lstm':
                outputs, new_states = self.rnn(packed_embedded, states) # (h, c)
            elif self.layer_type == 'gru':
                outputs, new_states = self.rnn(packed_embedded, states) # (h, )
            else:
                raise ValueError(f"Unsupported layer_type: {self.layer_type}")

            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = self.output_linear(outputs)
            
            return outputs, new_states
        else: 
            if self.layer_type == 'lstm':
                outputs, new_states = self.rnn(embedded, states) 
            elif self.layer_type == 'gru':
                outputs, new_states = self.rnn(embedded, states)
            else:
                raise ValueError(f"Unsupported layer_type: {self.layer_type}")
            
            outputs = self.output_linear(outputs)
            
            return outputs, new_states
    
class Joiner(torch.nn.Module):
    '''
    Args
        input_dim : encoder output dimension
        output_dim : vocab size
    '''
    def __init__(self, input_dim, output_dim):
        super(Joiner, self).__init__()
        self.input_dim = input_dim  # 256
        self.output_dim = output_dim  # 2000
        # self.output_linear = nn.Linear(self.input_dim, self.output_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim << 1, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, output_dim, bias=False),
        )
    def forward(self, enc_out, dec_out):
        '''
            Input
                enc_out : (B, T, c)
                dec_out : (B, U, C)
            
            return 
                output  # (B, T, U, C)
        '''
        if enc_out.dim() == 3 and dec_out.dim() == 3:
            input_length = enc_out.size(1)
            target_length = dec_out.size(1)

            enc_out = enc_out.unsqueeze(2)
            dec_out = dec_out.unsqueeze(1)

            enc_out = enc_out.repeat([1, 1, target_length, 1])
            dec_out = dec_out.repeat([1, input_length, 1, 1])

        outputs = torch.cat((enc_out, dec_out), dim=-1)
        outputs = self.fc(outputs)

        return outputs

