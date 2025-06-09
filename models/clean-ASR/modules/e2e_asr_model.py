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
            
            # RNN-T loss의 targets는 blank를 제외한 실제 레이블 시퀀스입니다.
            # Predictor의 입력 y는 (blank) + target_tokens 형태입니다.
            # 하지만 torchaudio.functional.rnnt_loss는 내부적으로 이를 처리합니다.
            # 따라서 Predictor에 전달할 targets는 그대로 ys_pad를 사용하고,
            # ylens는 원래 ys_pad의 길이를 사용합니다.
            # RNNT loss의 targets는 blank, sos, eos, ignore_id 모두를 제거한 순수 레이블 시퀀스여야 합니다.

            special_ids_for_rnnt_target = [self.ignore_id, self.sos, self.eos, self.blank] 
            
            rnnt_targets_list = []  # rnnt 에는 특수 토큰을 제거한 순수 레이블 시퀀스여야함.
            rnnt_target_lengths_list = []

            for y_seq_tensor, current_ylen in zip(ys_pad, ylens):
                actual_seq = [
                    token_id for token_id in y_seq_tensor[:current_ylen].tolist()
                    if token_id not in special_ids_for_rnnt_target
                ]
                rnnt_targets_list.append(actual_seq)
                rnnt_target_lengths_list.append(len(actual_seq))
            
            max_rnnt_target_len = max(rnnt_target_lengths_list) if rnnt_target_lengths_list else 0

            targets_for_rnnt_loss = torch.full(
                (batch_size, max_rnnt_target_len),
                fill_value=self.blank, # rnnt_loss의 targets에는 blank가 포함되지 않지만, 패딩값은 blank로 해도 무방
                                        # 중요한 것은 rnnt_loss가 blank 인덱스를 알고 있다는 것.
                                        # 다른 패딩 ID를 쓰는 것이 더 명확할 수 있습니다.
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


            # Step 2: Predictor와 Joiner를 사용하여 logits 생성
            # Predictor는 일반적으로 target 시퀀스의 시작에 blank를 추가하여 (U+1) 길이를 입력받는 구조입니다.
            # rnnt_loss 내부에서는 이 부분을 처리해주지만, 모델의 Predictor와 Joiner는 명시적으로 (U+1) 차원을 만들어야 합니다.
            # Predictor의 입력은 (B, U) 형태의 targets_for_rnnt_loss 이고, 출력은 (B, U, D_pred)입니다.
            # Joiner는 (B, T, D_enc)와 (B, U, D_pred)를 받아서 (B, T, U, V) 형태를 반환합니다.
            # 이 (B, T, U, V) 형태는 torchaudio.functional.rnnt_loss의 logit 입력과 일치하지 않습니다.
            # torchaudio의 rnnt_loss는 logit의 세 번째 차원이 U+1이어야 합니다.
            # 즉, Predictor가 (B, U+1, D_pred) 형태의 출력을 생성해야 합니다.

            # 이를 위해, Predictor에 입력할 타겟 시퀀스 앞에 blank 토큰을 추가합니다.
            # Predictor는 이 blank 토큰을 포함하여 (U+1) 길이의 시퀀스에 대한 예측을 수행해야 합니다.
            # rnnt_loss의 관점에서, Predictor의 입력은 (blank, y_1, ..., y_U) 입니다.
            # 따라서 Predictor에 전달할 `y`는 `targets_for_rnnt_loss` 앞에 `blank`를 추가한 형태여야 합니다.
            
            # Predictor의 입력 y: targets_for_rnnt_loss 앞에 self.blank 토큰 추가
            # `targets_for_rnnt_loss`는 이미 max_rnnt_target_len으로 패딩되어 있음.
            # 각 시퀀스의 시작에 blank를 추가합니다.
            zeros_for_predictor = torch.full((batch_size, 1), self.blank, dtype=torch.int32, device=current_device)
            predictor_input_y = torch.cat((zeros_for_predictor, targets_for_rnnt_loss), dim=1)
            predictor_input_y_lengths = target_lengths_for_rnnt_loss + 1 # 길이도 1 증가

            pred, _ = self.predictor(y=predictor_input_y, y_lengths=predictor_input_y_lengths)
            # pred의 차원: (B, max_rnnt_target_len + 1, D_pred)
            
            # Joiner는 (B, T, D_enc)와 (B, U+1, D_pred)를 받아서
            # (B, T, U+1, V) 형태의 logit을 생성해야 합니다.
            logits = self.joiner(hs_pad, pred) 
            # hs_pad (B, T, D_enc) -> unsqueeze(2) -> (B, T, 1, D_enc)
            # pred (B, U+1, D_pred) -> unsqueeze(1) -> (B, 1, U+1, D_pred)
            # logit = (B, T, U+1, D_join)

            # rnnt_loss의 logit_lengths는 hs_lengths를 사용.
            # rnnt_loss의 target_lengths는 target_lengths_for_rnnt_loss (U)를 사용.
            # logits의 T 차원과 U+1 차원이 각각 logit_lengths와 target_lengths에 맞춰져야 합니다.
            
            # logit의 T 차원 자르기 (logit_lengths에 맞춰)
            max_hs_length_in_batch = hs_lengths.max().item()
            if logits.shape[1] > max_hs_length_in_batch:
                logits = logits[:, :max_hs_length_in_batch, :, :]
            
            # logit의 U+1 차원 자르기 (target_lengths + 1에 맞춰)
            # rnnt_loss는 targets의 최대 길이 U에 1을 더한 차원 U+1을 기대합니다.
            max_predictor_output_len = predictor_input_y_lengths.max().item()
            if logits.shape[2] > max_predictor_output_len:
                logits = logits[:, :, :max_predictor_output_len, :]

            logits = logits.contiguous() 
            
            loss_rnnt = torchaudio.functional.rnnt_loss(
                logits=logits,
                targets=targets_for_rnnt_loss, # blank, sos, eos, ignore_id 제거된 실제 토큰 ID 시퀀스 (U 길이)
                logit_lengths=hs_lengths, # 인코더 출력의 실제 길이 (T)
                target_lengths=target_lengths_for_rnnt_loss, # RNN-T targets의 실제 길이 (U)
                blank=self.blank,
                reduction="sum"
            )
            loss_att = None
            self.acc = None
            
            
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
                    hyp = rnnt_beam_search(ylens, self.predictor, self.joiner, encoder_out_i, self.decoder_config.beam_size, self.blank, current_device)
                else:
                    raise ValueError(f'Unsupported decoding method: {self.decoder_config.decoding_method}')
                
                # 디코딩 결과 저장
                # hyps.append(sp.decode(hyp).split())
                # ... 이전 코드 ...
                # 디코딩 결과 저장
                # sp.decode(hyp)는 토큰 ID 리스트를 원래 문자열로 변환합니다.
                # SentencePiece의 default prefix가 ' ' (U+2581)이므로,
                # 이를 일반 공백으로 대체하고 .strip()으로 앞뒤 공백을 제거합니다.
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
