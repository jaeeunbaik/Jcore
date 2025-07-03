"""
🖤🐰 JaeEun Baik, 2025
"""

import numpy as np
import os
import logging
import torch
import torchaudio
import torch.nn as nn
import sentencepiece as spm
import kenlm
from warprnnt_pytorch import RNNTLoss as WarpRNNTLoss
from typing import Tuple

from argparse import Namespace
from torchaudio.models.decoder import ctc_decoder

from modules.conformer.model import Conformer
from modules.decoder.transformer_decoder import TransformerDecoder
from modules.decoder.ctc import CTC
from modules.decoder.rnn_transducer import Predictor, Joiner
from modules.decoder.label_smoothing_loss import LabelSmoothingLoss

from util.utils_text import ErrorCalculator, add_sos_eos, get_lm_file_paths
from util.utils_module import target_mask, th_accuracy
from util.utils_decoding import rnnt_greedy_search, rnnt_beam_search, transformer_beam_search, transformer_greedy_decode, CTCPrefixScore, CTCPrefixScorer, GreedyCTCDecoder
from util.utils_lm import KenLMWrapper

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5



class e2eASR(nn.Module):
    def __init__(
        self,
        model_config,
        tokenizer_path,
        ignore_id = 0,
    ):
        super().__init__()
        self.encoder_config = model_config.encoder
        self.decoder_config = model_config.decoder
        self.odim = self.decoder_config.odim 
        self.tokenizer_path = tokenizer_path
        
        self.blank = 0
        self.sos = 2
        self.eos = 3
        self.ignore_id = ignore_id
        
        self.decoder = None
        self.criterion = None
        self.ctc = None
        self.predictor = None
        self.joiner = None
        # Encoder
        if self.encoder_config.type == "Conformer":
            self.encoder = Conformer(
                input_dim=self.encoder_config.input_dim,
                encoder_dim=self.encoder_config.encoder_dim,
                num_encoder_layers=self.encoder_config.num_blocks,
                num_attention_heads=self.encoder_config.attention_heads,
                feed_forward_expansion_factor=self.encoder_config.ff_expansion_factor,
                conv_expansion_factor=self.encoder_config.conv_expansion_factor,
                input_dropout_p=self.encoder_config.input_dropout_p,
                feed_forward_dropout_p=self.encoder_config.ff_dropout_p,
                attention_dropout_p=self.encoder_config.att_dropout_p,
                conv_dropout_p=self.encoder_config.conv_dropout_p,
                conv_kernel_size=self.encoder_config.cnn_module_kernel,
                half_step_residual=self.encoder_config.half_step_residual
            )
        
        # Decoder
        self.decoder_type = self.decoder_config.type
        self.decoding_type = self.decoder_config.decoding_method
        self.ctc_weight = self.decoder_config.ctc_weight        
        
        if self.decoder_type == "rnnt":
            self.predictor = Predictor(n_layers=self.decoder_config.dlayers,
                                       embed_dim=self.encoder_config.encoder_dim,
                                       hidden_dim=self.decoder_config.dunits,
                                       output_dim=self.encoder_config.encoder_dim,
                                       num_embeddings=self.odim,
                                       layer_type=self.decoder_config.dtype)
            self.joiner = Joiner(input_dim=self.encoder_config.encoder_dim,
                                 output_dim=self.decoder_config.odim)
            self.ctc_weight = 0.0
            self.rnnt_loss = WarpRNNTLoss(blank=self.blank, reduction='mean') 
            
        elif self.decoder_type == "transformer":
            self.decoder = TransformerDecoder(self.odim)
            self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, 0.1, False)
            self.ctc_weight = 0.0
            
        elif self.decoder_type =="ctc":
            self.ctc = CTC(self.decoder_config)
            self.ctc_weight = 0.0

        self.rnnlm = None
        self.kenlm_path = None
        if self.decoder_config.lm is not None and self.decoder_config.lm.endswith('.arpa'):
            self.kenlm_path = self.decoder_config.lm
        self._init_parameters()


    def _init_parameters(self):
        """Initializes model parameters."""
        for p in self.parameters():
            if p.dim() > 1: # 1D 텐서 (bias)는 초기화하지 않음
                nn.init.xavier_uniform_(p)

        if self.decoder_type == "rnnt" and hasattr(self.predictor, 'rnn'):
            for name, param in self.predictor.rnn.named_parameters():
                if 'weight_ih' in name: # input-hidden weights
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name: # hidden-hidden weights
                    nn.init.orthogonal_(param.data) # 직교 초기화가 LSTM에 효과적일 수 있음
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
                    if 'bias_ih_l' in name: 
                        if 'lstm' in self.predictor.layer_type:
                           num_gates = param.data.shape[0] // 4
                           param.data[num_gates : 2 * num_gates].fill_(1.0) # Forget gate bias to 1.0
                    elif 'bias_hh_l' in name: # hidden-hidden bias
                         if 'lstm' in self.predictor.layer_type:
                           num_gates = param.data.shape[0] // 4
                           param.data[num_gates : 2 * num_gates].fill_(1.0) # Forget gate bias to 1.0


    def forward(self, xs_pad, ilens, ys_pad, ylens):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        hs_pad, hs_lengths = self.encoder(xs_pad, ilens)
        hs_lengths = hs_lengths.to(torch.int32)
        
        max_hs_length = hs_pad.size(1)
        hs_mask = (torch.arange(max_hs_length, device=hs_pad.device).unsqueeze(0) < hs_lengths.unsqueeze(1)).unsqueeze(1)

        # loss_rnnt = None
        # if self.decoder_type == 'rnnt':
        #     batch_size = xs_pad.size(0)
        #     current_device = xs_pad.device
            
        #     zeros_for_predictor = torch.full((batch_size, 1), self.blank, dtype=torch.int32, device=current_device)
        #     predictor_input_y = torch.cat((zeros_for_predictor, ys_pad), dim=1)
        #     predictor_input_y_lengths = ylens + 1 

        #     # Predictor 호출
        #     pred, _ = self.predictor(y=predictor_input_y, y_lengths=predictor_input_y_lengths)
        #     # pred의 차원: (B, max_target_len_in_batch + 1, D_pred)
            
        #     # Joiner 호출
        #     logits = self.joiner(hs_pad, pred)
        #     # logit의 차원: (B, T, U+1, V) 
        #     # where T is max_hs_length, U+1 is max_predictor_input_len

        #     # RNNT Loss의 targets (pure labels) 준비: ys_pad에서 특수 토큰 제거
        #     special_ids_for_rnnt_target = [self.ignore_id, self.sos, self.eos, self.blank] 
        #     rnnt_targets_list = []  
        #     rnnt_target_lengths_list = []

        #     for y_seq_tensor, current_ylen in zip(ys_pad, ylens):
        #         actual_seq = [
        #             token_id for token_id in y_seq_tensor[:current_ylen].tolist()
        #             if token_id not in special_ids_for_rnnt_target
        #         ]
        #         rnnt_targets_list.append(actual_seq)
        #         rnnt_target_lengths_list.append(len(actual_seq))
            
        #     max_rnnt_target_len = max(rnnt_target_lengths_list) if rnnt_target_lengths_list else 0

        #     targets_for_rnnt_loss = torch.full(
        #         (batch_size, max_rnnt_target_len),
        #         fill_value=self.blank, # 여기서는 blank가 아니라 self.ignore_id (패딩용)을 사용하거나, 아니면 0이 아닌 다른 명확한 패딩 ID를 쓰는 것이 좋습니다.
        #                                 # torchaudio.functional.rnnt_loss가 targets에서 blank를 제외하므로 blank(0)으로 패딩해도 문제 없음.
        #         dtype=torch.int32,
        #         device=current_device
        #     )
        #     for i, seq in enumerate(rnnt_targets_list):
        #         if len(seq) > 0:
        #             targets_for_rnnt_loss[i, :len(seq)] = torch.tensor(
        #                 seq,
        #                 dtype=torch.int32,
        #                 device=current_device
        #             )
                    
        #     target_lengths_for_rnnt_loss = torch.tensor(rnnt_target_lengths_list, dtype=torch.int32, device=current_device)

        #     max_hs_length_in_batch = hs_lengths.max().item()
        #     if logits.shape[1] > max_hs_length_in_batch:
        #         logits = logits[:, :max_hs_length_in_batch, :, :]
            
        #     # logit의 세 번째 차원 (U+1)을 (rnnt_target_lengths + 1)에 맞춤
        #     max_predictor_output_len = predictor_input_y_lengths.max().item()
        #     if logits.shape[2] > max_predictor_output_len: # max_rnnt_target_len + 1
        #         logits = logits[:, :, :max_predictor_output_len, :]

        #     logits = logits.contiguous() 
            
        #     loss_rnnt = self.rnnt_loss(logits, targets_for_rnnt_loss, hs_lengths, target_lengths_for_rnnt_loss)

        # 2. rnnt loss
        loss_rnnt = None
        if self.decoder_type == 'rnnt':
            batch_size = xs_pad.size(0)
            current_device = xs_pad.device
            
            # Predictor 입력 준비: ys_pad 앞에 blank 토큰 추가
            # ys_pad는 (B, U) 형태이며, RNN-T Predictor는 (B, U+1) 형태의 입력을 받습니다.
            # 첫 번째 토큰은 항상 blank (0)으로 시작합니다.
            zeros_for_predictor = torch.full((batch_size, 1), self.blank, dtype=torch.int32, device=current_device)
            predictor_input_y = torch.cat((zeros_for_predictor, ys_pad), dim=1)
            predictor_input_y_lengths = ylens + 1 # Predictor 입력 길이는 레이블 길이 + 1 (blank 토큰 포함)
            # Predictor 호출
            pred, _ = self.predictor(y=predictor_input_y, y_lengths=predictor_input_y_lengths)
            # pred의 차원: (B, max_target_len_in_batch + 1, D_pred)
            
            # Joiner 호출
            # hs_pad: (B, T, D_enc)
            # pred: (B, U+1, D_pred)
            logits = self.joiner(hs_pad, pred)
            logits = logits.to(torch.float32)
            
            # logit의 차원: (B, T, U+1, V)
            # where T is max_hs_length, U+1 is max_predictor_input_len
            
            # logits의 T 차원 (인코더 출력 길이)을 hs_lengths에 맞게 자릅니다.
            # hs_lengths는 인코더의 실제 출력 길이이므로, logits의 T 차원도 이에 맞춰야 합니다.
            special_ids_for_rnnt_target = [self.ignore_id, self.sos, self.eos, self.blank] 
            rnnt_targets_list = []  
            rnnt_target_lengths_list = []

            for y_seq_tensor, current_ylen in zip(ys_pad, ylens):
                actual_seq = [
                    token_id for token_id in y_seq_tensor[:current_ylen].tolist()
                    if token_id not in special_ids_for_rnnt_target
                ]
                rnnt_targets_list.append(actual_seq)
                rnnt_target_lengths_list.append(len(actual_seq))
            
            max_hs_length_in_batch = hs_lengths.max().item()
            if logits.shape[1] > max_hs_length_in_batch:
                logits = logits[:, :max_hs_length_in_batch, :, :]
            
            # logits의 U+1 차원 (Predictor 출력 길이)을 predictor_input_y_lengths에 맞게 자릅니다.
            max_predictor_output_len = predictor_input_y_lengths.max().item()
            if logits.shape[2] > max_predictor_output_len:
                logits = logits[:, :, :max_predictor_output_len, :]
            logits = logits.contiguous() # RNNT Loss 계산을 위해 contiguous 메모리 할당 보장
            max_rnnt_target_len = max(rnnt_target_lengths_list) if rnnt_target_lengths_list else 0

            targets_for_rnnt_loss = torch.full(
                (batch_size, max_rnnt_target_len),
                fill_value=self.blank, # 여기서는 blank가 아니라 self.ignore_id (패딩용)을 사용하거나, 아니면 0이 아닌 다른 명확한 패딩 ID를 쓰는 것이 좋습니다.
                                        # torchaudio.functional.rnnt_loss가 targets에서 blank를 제외하므로 blank(0)으로 패딩해도 문제 없음.
                dtype=torch.int32,
                device=current_device
            )
            for i, seq in enumerate(rnnt_targets_list):
                if len(seq) > 0:
                    targets_for_rnnt_loss[i, :len(seq)] = torch.tensor(
                        seq,
                        dtype=torch.int32,
                        device=current_device
                    )
                    
            target_lengths_for_rnnt_loss = torch.tensor(rnnt_target_lengths_list, dtype=torch.int32, device=current_device)

            ys_pad = ys_pad.to(torch.int32)
            ylens = ylens.to(torch.int32)

            loss_rnnt = self.rnnt_loss(logits, targets_for_rnnt_loss, hs_lengths, target_lengths_for_rnnt_loss).mean()

        # 3. decoder loss   
        if self.decoder_type in ["transformer", "hybrid"]:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        # 3. ctc loss
        if self.decoder_type in ["transformer", "rnnt"]:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.encoder_config.encoder_dim), hs_lengths, ys_pad)
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(self.hs_pad.view(batch_size, -1, self.encoder_config.encoder_dim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

            
            
        # 5. compute cer/wer
        # if self.training or self.error_calculator is None or self.decoder is None:
            # cer, wer = None, None
        # else:
            # ys_hat = pred_pad.argmax(dim=-1)
            # cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # 6. return loss
        if self.decoder_type == "hybrid":
            loss = (1 - self.ctc_weight) * loss_att + self.ctc_weight * loss_ctc
        elif self.decoder_type == "ctc":
            loss = loss_ctc
        elif self.decoder_type == "transformer":
            loss = loss_att
        elif self.decoder_type == "rnnt":
            loss = loss_rnnt

        return {
            "loss": loss,
            "loss_ctc": loss_ctc.item() if loss_ctc is not None else None, 
            "loss_att": loss_att.item() if loss_att is not None else None
        }
    
    

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x, ilens):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x)
        enc_output, *_ = self.encoder(x, ilens)
        return enc_output.squeeze(0)

    def recognize(self, x, ilens, y, ylens, recog_args, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        sp = spm.SentencePieceProcessor()
        sp.load(self.tokenizer_path)
        
        labels = [sp.id_to_piece(i) for i in range(self.odim)]
        
        if self.kenlm_path:
            lm = KenLMWrapper(self.kenlm_path, sp)
        
        enc_output = self.encode(x, ilens)
        if enc_output.dim() == 2:
            enc_output = enc_output.unsqueeze(0)
        
        if self.decoder_type == 'ctc':
            if self.decoding_type == 'greedy':
                decoder = GreedyCTCDecoder(labels=labels, blank=self.blank)
                lpz = self.ctc.log_softmax(enc_output)
                all_greedy_results = []
                for i in range(lpz.size(0)):
                    single_seq_emission = lpz[i]
                    greedy_result_single_seq = decoder(single_seq_emission)
                    all_greedy_results.append(" ".join(greedy_result_single_seq))
                return all_greedy_results
            
            
            elif self.decoding_type == 'beamsearch':
                beam_size = self.decoder_config.beam_size
                lm_weight = 3.23
                word_score = -0.26
                files = get_lm_file_paths(self.decoder_config.lm_fld)
                
                lpz = self.ctc.log_softmax(enc_output)
                lpz_cpu = lpz.cpu()
                
                decoder = ctc_decoder(
                    lexicon=files.lexicon,
                    tokens=files.tokens,
                    lm=None,  # files.lm
                    nbest=self.decoder_config.beam_size,
                    beam_size=beam_size,
                    lm_weight=lm_weight,
                    word_score=word_score
                )
                
                beam_search_result = decoder(lpz_cpu)
                transcripts = []
                for i in range(lpz_cpu.size(0)):
                    beam_search_transcript = " ".join(beam_search_result[i][0].words).strip()
                    transcripts.append(beam_search_transcript)
                return transcripts
                   
            
        elif self.decoder_type == 'rnnt':
            batch_size = enc_output.size(0)
            hyps = []
            for i in range(batch_size):
                encoder_out_i = enc_output[i:i+1]
                current_device = encoder_out_i.device

                if self.decoder_config.decoding_method == 'greedy':
                    hyp = rnnt_greedy_search(self.predictor, self.joiner, encoder_out_i, self.blank, current_device)
                elif self.decoder_config.decoding_method == 'beamsearch':
                    hyp = rnnt_beam_search(ylens, 
                                           self.predictor, 
                                           self.joiner, 
                                           encoder_out_i, 
                                           self.decoder_config.beam_size, 
                                           self.blank, 
                                           current_device, 
                                           lm=lm, 
                                           lm_weight=self.decoder_config.lm_weight)
                else:
                    raise ValueError(f'Unsupported decoding method: {self.decoder_config.decoding_method}')
                
                decoded_text = sp.decode(hyp).replace(" ", " ").strip()
                hyps.append(decoded_text.split()) # 단어 단위로 분리하여 List[List[str]] 유지
            # 결과 반환
            if self.decoder_config.decoding_method == 'greedy_search':
                return {'greedy_search': hyps}
            
            else:
                return {f'beam_{self.decoder_config.beam_size}': hyps}
        
        elif self.decoder_type == 'transformer':
            current_device = enc_output.device
            if self.decoding_type == 'greedy':
                decoded_token_ids = transformer_greedy_decode(
                    model=self.decoder,
                    encoder_output=enc_output,
                    sos_id=self.sos,
                    eos_id=self.eos,
                    max_len=int(recog_args.maxlenratio*max(ylens).item()), # `maxlenratio`가 실제 max_len으로 사용된다고 가정
                    device=current_device
                )
                transcript = sp.decode(decoded_token_ids)
                return [transcript.replace("|", " ").strip()] # 결과를 리스트로 반환
            
            elif self.decoding_type == 'beamsearch':
                decoded_token_ids = transformer_beam_search(
                    model=self.decoder, # 마찬가지로 self.decoder를 전달
                    encoder_output=enc_output,
                    sos_id=self.sos,
                    eos_id=self.eos,
                    max_len=int(recog_args.maxlenratio*max(ylens).item()),
                    beam_size=recog_args.beam_size, # `recog_args`에 빔 크기 파라미터가 있어야 합니다.
                    device=current_device
                )
                # 토큰 ID를 실제 텍스트로 변환
                transcript = sp.decode(decoded_token_ids)
                return [transcript.replace("|", " ").strip()] 
            else:
                raise ValueError(f'Unsupported decoding method for Transformer: {self.decoding_type}')
        return []


