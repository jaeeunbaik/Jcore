"""
🖤🐰 Jaeeun Baik, 2025
"""

import numpy
import logging
import torch
import torchaudio
import torch.nn as nn
import sentencepiece as spm

from argparse import Namespace
from torchaudio.models.decoder import ctc_decoder

from modules.conformer.model import Conformer
from modules.decoder.transformer_decoder import TransformerDecoder
from modules.decoder.ctc import CTC
from modules.decoder.rnn_transducer import Predictor, Joiner
from modules.decoder.label_smoothing_loss import LabelSmoothingLoss

from util.utils_text import ErrorCalculator, add_sos_eos, get_lm_file_paths
from util.utils_module import subsequent_mask, target_mask, end_detect, th_accuracy
from util.utils_decoding import rnnt_greedy_search, rnnt_beam_search, transformer_beam_search, transformer_greedy_decode, CTCPrefixScore, CTCPrefixScorer, GreedyCTCDecoder

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5

class e2eASR(nn.Module):
    def __init__(
        self,
        model_config,
        tokenizer_path,
        ignore_id = -1,
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
        self.cal_loss = None
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
                                 output_dim=self.encoder_config.encoder_dim)
            self.cal_loss = torchaudio.transforms.RNNTLoss(blank=self.blank, reduction="sum")
            self.ctc_weight = 0.0
            
        elif self.decoder_type == "transformer":
            self.decoder = TransformerDecoder(self.odim)
            self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, 0.1, False)
            self.ctc_weight = 0.0
            
        elif self.decoder_type =="ctc":
            self.ctc = CTC(self.decoder_config)
            self.ctc_weight = 0.0

        if model_config.report_cer or model_config.report_wer:
            self.error_calculator = ErrorCalculator(
                model_config.char_list,
                model_config.sym_space,
                model_config.sym_blank,
                model_config.report_cer,
                model_config.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None


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
        # xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        hs_pad, hs_lengths = self.encoder(xs_pad, ilens)
        hs_lengths = hs_lengths.to(torch.int32)
        
        max_hs_length = hs_pad.size(1)
        hs_mask = (torch.arange(max_hs_length, device=hs_pad.device).unsqueeze(0) < hs_lengths.unsqueeze(1)).unsqueeze(1)

        # 2. decoder loss
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
        
        # 4. rnnt loss
        if self.decoder_type == 'rnnt':
            batch_size = xs_pad.size(0)
            current_device = xs_pad.device
            
            zeros = torch.zeros((batch_size, 1)).to(device=current_device)
            # rnnt loss 계산을 위한 padding 되지 않은 ys
            special_ids = [self.ignore_id, self.sos, self.eos, self.blank] # self.blank 포함
            ys_actual_sequences = []
            for y_seq_tensor in ys_pad: 
                current_actual_seq = []
                for token_id in y_seq_tensor.tolist():
                    if token_id not in special_ids: # 특수 토큰 필터링 (blank, sos, eos, ignore_id 모두 제거)
                        current_actual_seq.append(token_id)
                ys_actual_sequences.append(current_actual_seq)
                
            targets = torch.cat((zeros, ys_pad), dim=1).to(
                device=current_device, dtype=torch.int
            )
            target_lengths = (ylens + 1).to(device=current_device)
            pred, _ = self.predictor(y=targets, y_lengths=target_lengths)
            logits = self.joiner(hs_pad, pred)
            
            max_effective_hs_length = hs_lengths.max().item()
            if logits.shape[1] > max_effective_hs_length:
                logits = logits[:, :max_effective_hs_length, :, :]
            special_ids = [self.ignore_id, self.sos, self.eos, self.blank] 
            
            ys_actual_sequences = [] # List of lists, each containing non-special tokens
            actual_target_lengths = [] # Corresponding lengths for rnnt_loss targets

            # `ys_pad`의 각 시퀀스를 `ylens`에 주어진 길이까지 탐색
            for y_seq_tensor, current_ylen in zip(ys_pad, ylens): 
                current_actual_seq = []
                # 현재 시퀀스의 실제 길이까지만 순회하며 특수 토큰 제거
                for token_id in y_seq_tensor[:current_ylen].tolist(): 
                    if token_id not in special_ids: 
                        current_actual_seq.append(token_id)
                ys_actual_sequences.append(current_actual_seq)
                actual_target_lengths.append(len(current_actual_seq)) # 특수 토큰 제외 후 실제 길이 저장

            max_len_for_rnnt_targets = max(actual_target_lengths) if actual_target_lengths else 0

            targets_for_rnnt_loss_input = torch.full(
                (batch_size, max_len_for_rnnt_targets), 
                fill_value=self.blank, 
                dtype=torch.int32,
                device=current_device
            )
            for i, actual_seq in enumerate(ys_actual_sequences):
                if len(actual_seq) > 0: # 빈 시퀀스인 경우 슬라이싱 오류 방지
                    targets_for_rnnt_loss_input[i, :len(actual_seq)] = torch.tensor(
                        actual_seq, 
                        dtype=torch.int32, 
                        device=current_device
                    )
            logits = logits.contiguous()
            # --- rnnt_loss 호출 ---
            loss_rnnt = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=targets_for_rnnt_loss_input, # `ys_actual_sequences`로 채워진 텐서
            logit_lengths=hs_lengths, # 인코더 실제 길이
            target_lengths=torch.tensor(actual_target_lengths, dtype=torch.int32, device=current_device), # `targets`에 해당하는 실제 길이
            blank=self.blank,
            reduction="sum"
            )

            
        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

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
            "loss_att": loss_att.item() if loss_att is not None else None, 
            "cer": cer,
            "wer": wer
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

    def recognize(self, x, ilens, y, ylens, recog_args, rnnlm=None, use_jit=False):
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
                    nbest=3,
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
                    hyp = rnnt_beam_search(ylens, self.predictor, self.joiner, encoder_out_i, self.decoder_config.beam_size, self.blank, current_device)
                else:
                    raise ValueError(f'Unsupported decoding method: {self.decoder_config.decoding_method}')
                
                # 디코딩 결과 저장
                hyps.append(sp.decode(hyp).split())
            
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
