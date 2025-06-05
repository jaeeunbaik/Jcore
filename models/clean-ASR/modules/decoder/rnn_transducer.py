# 🐰🖤
# 2025, JaeEunBaik

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional


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
            embedding_dim=output_dim,
            padding_idx=0,
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
                hidden_size=2*self.embed_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=rnn_dropout_rate
            )
        self.output_linear = nn.Linear(self.hidden_dim, output_dim)
        
    def forward(
        self,
        y: torch.Tensor,
        y_lengths: Optional[Tuple[torch.Tensor]] = None,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
            input
                y:
                state: a tuple of two tensors containing the states information of LSTM layers in this decoder.
            return
            
        """
        # embed_out = self.embed(y)
        # embed_out = self.embed_dropout(embed_out)
        # rnn_out, (h, c) = self.rnn(embed_out, states)
        # out = self.output_linear(rnn_out)
        
        # return out, (h, c)
        embedded = self.embed(y)

        if y_lengths is not None:
            y_lengths_cpu = y_lengths.to("cpu", dtype=torch.int64)
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded.transpose(0, 1), y_lengths_cpu, enforce_sorted=False
            )
            outputs, states = self.rnn(embedded, states)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = self.output_linear(outputs.transpose(0, 1))
        else:
            outputs, states = self.rnn(embedded, states)
            outputs = self.output_linear(outputs)

        return outputs, states
    
    
    
class Joiner(torch.nn.Module):
    '''
    Args
        input_dim : encoder output dimension
        output_dim : vocab size
    '''
    def __init__(self, input_dim, output_dim):
        super(Joiner, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_linear = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, enc_out, dec_out):
        '''
            enc_out : (B, T, c)
            dec_out : (B, U, C)
        '''
        enc_out = enc_out.unsqueeze(2)
        dec_out = dec_out.unsqueeze(1)
        logit = enc_out + dec_out
        logit = torch.tanh(logit)
        
        output = self.output_linear(logit)
        
        return output
    
    
# class Joiner(torch.nn.Module):
#     def __init__(self, encoder_output_dim, predictor_output_dim, output_dim, joiner_hidden_dim=512):
#         super(Joiner, self).__init__()
#         self.encoder_output_dim = encoder_output_dim # C
#         self.predictor_output_dim = predictor_output_dim # D_pred
#         self.output_dim = output_dim # Vocab_size (odim)
#         self.joiner_hidden_dim = joiner_hidden_dim # 중간 결합 차원 (e.g., 256 or 512)

#         # self.linear_enc = nn.Linear(self.encoder_output_dim, self.joiner_hidden_dim)
#         # self.linear_pred = nn.Linear(self.predictor_output_dim, self.joiner_hidden_dim)
#         # self.output_linear = nn.Linear(self.joiner_hidden_dim, self.output_dim)
#         self.fc = nn.Sequential(
#             nn.Linear(encoder_output_dim << 1, encoder_output_dim),
#             nn.Tanh(),
#             nn.Linear(encoder_output_dim, output_dim, bias=False),
#         )
        
        
#     def forward(self, enc_out, dec_out):
#         if enc_out.dim() == 3 and dec_out.dim() == 3:
#             input_length = enc_out.size(1)
#             target_length = dec_out.size(1)

#             enc_out = enc_out.unsqueeze(2)
#             dec_out = dec_out.unsqueeze(1)

#             enc_out = enc_out.repeat([1, 1, target_length, 1])
#             dec_out = dec_out.repeat([1, input_length, 1, 1])

#         outputs = torch.cat((enc_out, dec_out), dim=-1)
#         outputs = self.fc(outputs)

#         return outputs