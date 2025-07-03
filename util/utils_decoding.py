import torch
import numpy as np
import logging
import os
from typing import List, Tuple, Optional, Union
import sentencepiece as spm
import torch.nn as nn
import kenlm

from distutils.version import LooseVersion
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union


is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")
# LooseVersion('1.2.0') == LooseVersion(torch.__version__) can't include e.g. 1.2.0+aaa
is_torch_1_2 = (
    LooseVersion("1.3") > LooseVersion(torch.__version__) >= LooseVersion("1.2")
)
datatype = torch.bool if is_torch_1_2_plus else torch.uint85


@dataclass
class CustomFilePaths:
    lexicon: Optional[str] = None
    tokens: Optional[str] = None
    lm: Optional[str] = None

def get_lm_file_paths(base_dir: str) -> CustomFilePaths:
    """
    주어진 디렉토리에서 언어 모델 관련 파일(lexicon, tokens, lm)의 경로를 찾습니다.

    Args:
        base_dir (str): 언어 모델 관련 파일들이 존재하는 상위 폴더 경로.

    Returns:
        CustomFilePaths: lm, lexicon, tokens 파일 경로를 담은 데이터 클래스.
                         파일이 없으면 해당 필드는 None으로 유지됩니다.
    """
    found_lexicon = None
    found_tokens = None
    found_lm = None

    # 디렉토리 내의 파일들을 탐색
    for root, _, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file == "lexicon.txt":
                found_lexicon = file_path
            elif file == "tokens.txt":
                found_tokens = file_path
            elif file.endswith(".bin") and "lm" in file:
                found_lm = file_path

    return CustomFilePaths(lexicon=found_lexicon, tokens=found_tokens, lm=found_lm)

class KenLMWrapper:
    def __init__(self, lm_path: str, sp_processor: spm.SentencePieceProcessor):
        if not os.path.exists(lm_path):
            raise FileNotFoundError(f"KenLM model not found at {lm_path}")
        
        # logging.info(f"Loading KenLM model from {lm_path}...")
        self.lm = kenlm.LanguageModel(lm_path) # KenLM 모델 로드
        self.sp = sp_processor
        self.vocab_size = sp_processor.get_piece_size() # vocab_size 추가
        # logging.info("KenLM model loaded.")

        self.unk_id = self.sp.unk_id() 
        
    def get_initial_lm_state(self) -> kenlm.State:
        state = kenlm.State()
        self.lm.BeginSentenceWrite(state) # 문장 시작 컨텍스트로 상태 초기화
        return state 
    

def rnnt_greedy_search(decoder, joiner, encoder_out: torch.Tensor, blank_id, device) -> List[int]:
    """
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
    Returns:
      Return the decoded result.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)

    sos = torch.tensor([blank_id], device=device, dtype=torch.int64).reshape(1, 1)
    
    if isinstance(decoder.rnn, torch.nn.LSTM):
        decoder_out, (h, c) = decoder(y=sos)
    elif isinstance(decoder.rnn, torch.nn.GRU):
        decoder_out, h = decoder(y=sos)
        
    T = encoder_out.size(1)
    t = 0
    hyp = []

    sym_per_frame = 0
    sym_per_utt = 0

    max_sym_per_utt = 1000
    max_sym_per_frame = 3

    while t < T and sym_per_utt < max_sym_per_utt:
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # fmt: on
        logits = joiner(current_encoder_out, decoder_out)
        # logits is (1, 1, 1, vocab_size)

        log_prob = logits.log_softmax(dim=-1)
        # log_prob is (1, 1, 1, vocab_size)
        # TODO: Use logits.argmax()
        y = log_prob.argmax()
        if y != blank_id:
            hyp.append(y.item())
            y = y.reshape(1, 1)
            # decoder_out, (h, c) = decoder(y=y, states=(h, c))
            decoder_out, h = decoder(y=y, states=h)

            sym_per_utt += 1
            sym_per_frame += 1

        if y == blank_id or sym_per_frame > max_sym_per_frame:
            sym_per_frame = 0
            t += 1

    return hyp


@dataclass
class TransducerHypothesis:
    ys: List[int]  # the predicted sequences so far
    log_prob: float  # The log prob of ys

    # Optional decoder state. We assume it is LSTM for now,
    # so the state is a tuple (h, c)
    decoder_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
def rnnt_beam_search(
    ylens, 
    predictor, 
    joiner, 
    encoder_out: torch.Tensor, 
    beam_size: int, 
    blank_id: int, 
    device: torch.device, 
    lm: Optional[Union[nn.Module, KenLMWrapper]]=None,
    lm_weight: float=0.0
) -> List[int]:
    
    # Initialize beam candidates
    initial_tokens = torch.tensor([blank_id], device=device, dtype=torch.int32)
    beams = [] 
    for _ in range(beam_size): 
        initial_lm_state = None
        if lm is not None and isinstance(lm, KenLMWrapper):
            initial_lm_state = lm.get_initial_lm_state()
        
        beams.append({
            "score": 0.0, 
            "tokens": initial_tokens, 
            "predictor_states": None, 
            "lm_states": initial_lm_state
        })

    time_steps = encoder_out.size(1)
    max_decode_steps = int(time_steps) * 2 

    is_lstm = isinstance(predictor.rnn, torch.nn.LSTM)
    is_gru = isinstance(predictor.rnn, torch.nn.GRU) 

    for t in range(time_steps):
        new_beams_at_t = [] 

        current_tokens_batch = torch.cat([b["tokens"][-1:].unsqueeze(0) for b in beams], dim=0)
        
        num_layers_directions = predictor.rnn.num_layers * (2 if predictor.rnn.bidirectional else 1) 
        hidden_size = predictor.hidden_dim

        predictor_input_states = None
        if beams[0]["predictor_states"] is not None: 
            if is_lstm: # LSTM인 경우 (h, c) 튜플 상태
                h_states_to_concat = [b["predictor_states"][0] for b in beams]
                c_states_to_concat = [b["predictor_states"][1] for b in beams]
                predictor_input_states = (torch.cat(h_states_to_concat, dim=1), torch.cat(c_states_to_concat, dim=1))
            elif is_gru: # GRU인 경우 단일 h 텐서 상태
                h_states_to_concat = [b["predictor_states"] for b in beams]
                predictor_input_states = torch.cat(h_states_to_concat, dim=1)
            else:
                raise TypeError(f"Unsupported RNN type in Predictor: {type(predictor.rnn)}")
            
        pred_out_batch, new_predictor_states_batch = predictor(
            y=current_tokens_batch, 
            y_lengths=torch.ones(current_tokens_batch.size(0), dtype=torch.long, device=device), 
            states=predictor_input_states
        )

        lm_pred_out_batch = None

        if lm is not None and isinstance(lm, nn.Module): 
            lm_input_states_h = []
            lm_input_states_c = []
            for b in beams:
                if b["lm_states"] is None:
                    lm_input_states_h.append(torch.zeros(lm.num_layers, 1, lm.hidden_dim, device=device))
                    lm_input_states_c.append(torch.zeros(lm.num_layers, 1, lm.hidden_dim, device=device))
                else:
                    lm_input_states_h.append(b["lm_states"][0])
                    lm_input_states_c.append(b["lm_states"][1])
            lm_input_states = (torch.cat(lm_input_states_h, dim=1), torch.cat(lm_input_states_c, dim=1))

            lm_logits, new_lm_states_batch = lm(input_tokens=current_tokens_batch, states=lm_input_states)
            lm_pred_out_batch = torch.log_softmax(lm_logits, dim=-1) # LM log probs
            
        elif lm is not None and not isinstance(lm, KenLMWrapper):
            logging.warning(f"Unsupported LM type encountered: {type(lm)}. LM will be skipped.")
            lm = None 
            
        for i, beam in enumerate(beams):
            extracted_predictor_states_i = None
            if is_lstm:
                extracted_predictor_states_i = (new_predictor_states_batch[0][:, i:i+1, :], new_predictor_states_batch[1][:, i:i+1, :])
            elif is_gru:
                extracted_predictor_states_i = new_predictor_states_batch[:, i:i+1, :]
            
            extracted_lm_states_i = None
            if lm is not None and isinstance(lm, KenLMWrapper):
                extracted_lm_states_i = beam["lm_states"] 
            elif lm is not None and isinstance(lm, nn.Module): # RNNLM/Transformer-LM
                 extracted_lm_states_i = (new_lm_states_batch[0][:, i:i+1, :], new_lm_states_batch[1][:, i:i+1, :])

            pred_out_i = pred_out_batch[i:i+1] # (1, 1, D_pred)
            enc_out_t = encoder_out[:, t:t+1, :] # (1, 1, C_enc)

            logits = joiner(enc_out_t, pred_out_i)
            rnnt_log_probs = torch.log_softmax(logits, dim=-1).squeeze() # RNN-T의 로그 확률

            final_log_probs = rnnt_log_probs
            if lm is not None:
                if isinstance(lm, KenLMWrapper):
                    pass 
                elif lm_pred_out_batch is not None: 
                    lm_log_probs_i_tensor = lm_pred_out_batch[i, :] # 해당 빔의 LM 로그 확률 (1D 텐서)
                    final_log_probs = rnnt_log_probs + lm_weight * lm_log_probs_i_tensor # RNN-T + LM 점수

            blank_score = final_log_probs[blank_id].item()

            new_beams_at_t.append({
                "score": beam["score"] + blank_score,
                "tokens": beam["tokens"], # 토큰 시퀀스 변화 없음
                "predictor_states": beam["predictor_states"], # Predictor 상태 유지
                "lm_states": beam["lm_states"], # LM 상태도 유지
            })

            vocab_size = final_log_probs.size(-1)
            non_blank_indices = torch.arange(vocab_size, device=device)
            non_blank_indices = non_blank_indices[non_blank_indices != blank_id]

            non_blank_log_probs_filtered = final_log_probs[non_blank_indices]

            topk_non_blank_size = min(beam_size - 1, non_blank_log_probs_filtered.size(-1))

            if topk_non_blank_size > 0:
                topk_scores_rnnt, topk_relative_indices = torch.topk(non_blank_log_probs_filtered, topk_non_blank_size, dim=-1)
                topk_tokens_non_blank = non_blank_indices[topk_relative_indices]

                for rnnt_score, token_id_tensor in zip(topk_scores_rnnt.tolist(), topk_tokens_non_blank):
                    token_id = token_id_tensor.item()
                    
                    new_lm_state_for_this_path = None
                    lm_score_for_this_token = 0.0 # 기본값

                    if lm is not None and isinstance(lm, KenLMWrapper):
                        token_str_to_score = lm.sp.id_to_piece(token_id)
                        current_lm_state_for_path = beam["lm_states"] 
                        new_kenlm_state = kenlm.State() 
                        
                        score_from_kenlm = lm.lm.BaseScore(current_lm_state_for_path, token_str_to_score, new_kenlm_state)
                        
                        lm_score_for_this_token = score_from_kenlm
                        new_lm_state_for_this_path = new_kenlm_state
                        
                    elif lm is not None and isinstance(lm, nn.Module): 
                        new_lm_state_for_this_path = extracted_lm_states_i
                        if lm_pred_out_batch is not None:
                            lm_score_for_this_token = lm_pred_out_batch[i, token_id].item()

                    combined_score = rnnt_score + lm_weight * lm_score_for_this_token

                    new_beams_at_t.append({
                        "score": beam["score"] + combined_score,
                        "tokens": torch.cat((beam["tokens"], token_id_tensor.unsqueeze(0))), 
                        "predictor_states": extracted_predictor_states_i, 
                        "lm_states": new_lm_state_for_this_path, 
                    })
                    
        beams = sorted(new_beams_at_t, key=lambda x: x["score"], reverse=True)[:beam_size]
        
        if all(b["tokens"][-1].item() == blank_id for b in beams) or t == max_decode_steps - 1:
            break

    best_beam = max(beams, key=lambda x: x["score"])
    final_tokens = [token.item() for token in best_beam["tokens"] if token.item() != blank_id]

    return final_tokens

class TMHypothesis:
    def __init__(self, tokens: List[int], log_prob: float, decoder_input_tensor: torch.Tensor, decoder_state: List[Any] = None):
        self.tokens = tokens
        self.log_prob = log_prob
        self.decoder_input_tensor = decoder_input_tensor
        self.decoder_state = decoder_state # 각 디코더 레이어의 캐시를 저장할 필드

    def __lt__(self, other):
        return self.log_prob < other.log_prob

    def __repr__(self):
        return f"Hyp(tokens={self.tokens}, log_prob={self.log_prob:.2f})"


def transformer_greedy_decode(
    model, 
    encoder_output: torch.Tensor,  # (1, T_enc, D_model) - 인코더 최종 출력
    sos_id: int, 
    eos_id: int, 
    max_len: int, 
    device: torch.device,
) -> List[int]:
    """
    Args:
        model: Transformer Decoder를 포함하는 전체 ASR 모델 또는 Decoder 모듈.
               여기서는 `model.decoder.score`를 호출할 수 있다고 가정합니다.
        encoder_output: 인코더에서 나온 출력 텐서. (1, T_enc, D_model) 형태.
        sos_id: 시작 토큰 ID.
        eos_id: 종료 토큰 ID.
        max_len: 생성할 시퀀스의 최대 길이.
        device: 디코딩을 수행할 디바이스 (CPU 또는 CUDA).

    Returns:
        디코딩된 토큰 ID 리스트 (SOS 및 EOS 제외).
    """
    current_token = torch.tensor([sos_id], dtype=torch.long, device=device)
    
    decoded_tokens = []
    decoder_states = None

    for _ in range(max_len):
        log_probs_next_token, new_decoder_states = model.decoder.score(
            current_token, 
            decoder_states, 
            encoder_output
        )
        
        next_token_id = log_probs_next_token.argmax(dim=-1).item()
        
        if next_token_id == eos_id:
            break 
        
        decoded_tokens.append(next_token_id)
        
        current_token = torch.tensor([next_token_id], dtype=torch.long, device=device)
        decoder_states = new_decoder_states
        
    return decoded_tokens
    
    
    
def transformer_beam_search(
    model, 
    encoder_output: torch.Tensor,  
    sos_id: int, 
    eos_id: int, 
    max_len: int,
    beam_size: int,
    device: torch.device,
) -> List[int]:
    """
    Transformer Decoder를 위한 Beam Search Decoding 함수.
    forward_one_step 메서드를 활용하여 효율적인 디코딩을 수행합니다.

    Args:
        model: Transformer Decoder를 포함하는 전체 ASR 모델 또는 Decoder 모듈.
               여기서는 `model.decoder.batch_score` 또는 `model.decoder.forward_one_step`
               을 호출할 수 있다고 가정합니다.
        encoder_output: 인코더에서 나온 출력 텐서. (1, T_enc, D_model) 형태.
        sos_id: 시작 토큰 ID.
        eos_id: 종료 토큰 ID.
        max_len: 생성할 시퀀스의 최대 길이.
        beam_size: 빔 크기.
        device: 디코딩을 수행할 디바이스 (CPU 또는 CUDA).

    Returns:
        가장 높은 점수를 가진 디코딩된 토큰 ID 리스트 (EOS 제외).
    """
    expanded_encoder_output = encoder_output.repeat(beam_size, 1, 1)

    initial_hypothesis = TMHypothesis(
        tokens=[sos_id],
        log_prob=0.0,
        decoder_input_tensor=torch.tensor([[sos_id]], dtype=torch.long, device=device),
        decoder_state=None
    )
    beams = [initial_hypothesis]
    
    completed_hypotheses = []

    for _ in range(max_len):
        if not beams:
            break

        current_decoder_inputs = []
        current_decoder_states = []
        
        num_current_beams = len(beams)

        for hyp in beams:
            current_decoder_inputs.append(hyp.decoder_input_tensor[:, -1:]) # (1, 1) 
            current_decoder_states.append(hyp.decoder_state)
        # (num_current_beams, 1)
        batched_decoder_inputs = torch.cat(current_decoder_inputs, dim=0)

        encoder_output_for_batch = encoder_output.repeat(num_current_beams, 1, 1)

        if current_decoder_states[0] is None:
            states_for_batch_score = [None] * num_current_beams
        else:
            states_for_batch_score = current_decoder_states

        log_probs_next_tokens_batch, new_decoder_states_batch = model.decoder.batch_score(
            batched_decoder_inputs,  # (num_current_beams, 1)
            states_for_batch_score,  # List[Any] of decoder states (num_current_beams)
            encoder_output_for_batch # (num_current_beams, T_enc, D_model)
        )
        
        new_beams_candidates = []

        for i, hyp in enumerate(beams):
            log_probs = log_probs_next_tokens_batch[i] # (vocab_size,)
            
            topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_size, dim=-1)
            
            current_hyp_new_state = new_decoder_states_batch[i] 

            for log_prob_val, token_id in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                new_log_prob = hyp.log_prob + log_prob_val
                new_tokens = hyp.tokens + [token_id]
                next_decoder_input_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                
                new_hyp = TMHypothesis(
                    tokens=new_tokens,
                    log_prob=new_log_prob,
                    decoder_input_tensor=next_decoder_input_token,
                    decoder_state=current_hyp_new_state
                )
                
                new_beams_candidates.append(new_hyp)
        new_beams_candidates.sort(key=lambda x: x.log_prob, reverse=True)
        beams = new_beams_candidates[:beam_size]
        
        next_beams = []
        for hyp in beams:
            if hyp.tokens and hyp.tokens[-1] == eos_id:
                completed_hypotheses.append(hyp)
            else:
                next_beams.append(hyp)
        beams = next_beams

        if not beams and completed_hypotheses:
            break
            
    if not completed_hypotheses and beams:
        completed_hypotheses = beams

    if not completed_hypotheses:
        return [] # 디코딩 실패

    best_hypothesis = max(completed_hypotheses, key=lambda x: x.log_prob / (len(x.tokens) - 1 if len(x.tokens) > 1 else 1)) 
    
    final_tokens = best_hypothesis.tokens
    if final_tokens and final_tokens[0] == sos_id:
        final_tokens = final_tokens[1:] 
    if final_tokens and final_tokens[-1] == eos_id:
        final_tokens = final_tokens[:-1]

    return final_tokens



class CTCPrefixScoreTH(object):
    """Batch processing of CTCPrefixScore

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probablities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x, xlens, blank, eos, margin=0):
        """Construct CTC prefix scorer

        :param torch.Tensor x: input label posterior sequences (B, T, O)
        :param torch.Tensor xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        # In the comment lines,
        # we assume T: input_length, B: batch size, W: beam width, O: output dim.
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.size(0)
        self.input_length = x.size(1)
        self.odim = x.size(2)
        self.dtype = x.dtype
        self.device = (
            torch.device("cuda:%d" % x.get_device())
            if x.is_cuda
            else torch.device("cpu")
        )
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Reshape input x
        xn = x.transpose(0, 1)  # (B, T, O) -> (T, B, O)
        xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
        self.x = torch.stack([xn, xb])  # (2, T, B, O)
        self.end_frames = torch.as_tensor(xlens) - 1

        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = torch.arange(
                self.input_length, dtype=self.dtype, device=self.device
            )
        # Base indices for index conversion
        self.idx_bh = None
        self.idx_b = torch.arange(self.batch, device=self.device)
        self.idx_bo = (self.idx_b * self.odim).unsqueeze(1)

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels

        :param list y: prefix label sequences
        :param tuple state: previous CTC state
        :param torch.Tensor pre_scores: scores for pre-selection of hypotheses (BW, O)
        :param torch.Tensor att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """
        output_length = len(y[0]) - 1  # ignore sos
        last_ids = [yi[-1] for yi in y]  # last output label ids
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch  # assuming each utterance has the same # of hyps
        self.scoring_num = scoring_ids.size(-1) if scoring_ids is not None else 0
        # prepare state info
        if state is None:
            r_prev = torch.full(
                (self.input_length, 2, self.batch, n_hyps),
                self.logzero,
                dtype=self.dtype,
                device=self.device,
            )
            r_prev[:, 1] = torch.cumsum(self.x[0, :, :, self.blank], 0).unsqueeze(2)
            r_prev = r_prev.view(-1, 2, n_bh)
            s_prev = 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

        # select input dimensions for scoring
        if self.scoring_num > 0:
            scoring_idmap = torch.full(
                (n_bh, self.odim), -1, dtype=torch.long, device=self.device
            )
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = torch.arange(n_bh, device=self.device).view(-1, 1)
            scoring_idmap[self.idx_bh[:n_bh], scoring_ids] = torch.arange(
                snum, device=self.device
            )
            scoring_idx = (
                scoring_ids + self.idx_bo.repeat(1, n_hyps).view(-1, 1)
            ).view(-1)
            x_ = torch.index_select(
                self.x.view(2, -1, self.batch * self.odim), 2, scoring_idx
            ).view(2, -1, n_bh, snum)
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = self.x.unsqueeze(3).repeat(1, 1, 1, n_hyps, 1).view(2, -1, n_bh, snum)

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = torch.full(
            (self.input_length, 2, n_bh, snum),
            self.logzero,
            dtype=self.dtype,
            device=self.device,
        )
        if output_length == 0:
            r[0, 0] = x_[0, 0]

        r_sum = torch.logsumexp(r_prev, 1)
        log_phi = r_sum.unsqueeze(2).repeat(1, 1, snum)
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, pos] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx]] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = torch.matmul(att_w, self.frame_ids)
            f_min = max(int(f_arg.min().cpu()), f_min_prev)
            f_max = max(int(f_arg.max().cpu()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]
            rr = torch.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]).view(
                2, 2, n_bh, snum
            )
            r[t] = torch.logsumexp(rr, 1) + x_[:, t]

        # compute log prefix probabilities log(psi)
        log_phi_x = torch.cat((log_phi[0].unsqueeze(0), log_phi[:-1]), dim=0) + x_[0]
        if scoring_ids is not None:
            log_psi = torch.full(
                (n_bh, self.odim), self.logzero, dtype=self.dtype, device=self.device
            )
            log_psi_ = torch.logsumexp(
                torch.cat((log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0),
                dim=0,
            )
            for si in range(n_bh):
                log_psi[si, scoring_ids[si]] = log_psi_[si]
        else:
            log_psi = torch.logsumexp(
                torch.cat((log_phi_x[start:end], r[start - 1, 0].unsqueeze(0)), dim=0),
                dim=0,
            )

        for si in range(n_bh):
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids

        :param state    : CTC state
        :param best_ids : index numbers selected by beam pruning (B, W)
        :return selected_state
        """
        r, s, f_min, f_max, scoring_idmap = state
        # convert ids to BHO space
        n_bh = len(s)
        n_hyps = n_bh // self.batch
        vidx = (best_ids + (self.idx_b * (n_hyps * self.odim)).view(-1, 1)).view(-1)
        # select hypothesis scores
        s_new = torch.index_select(s.view(-1), 0, vidx)
        s_new = s_new.view(-1, 1).repeat(1, self.odim).view(n_bh, self.odim)
        # convert ids to BHS space (S: scoring_num)
        if scoring_idmap is not None:
            snum = self.scoring_num
            hyp_idx = (best_ids // self.odim + (self.idx_b * n_hyps).view(-1, 1)).view(
                -1
            )
            label_ids = torch.fmod(best_ids, self.odim).view(-1)
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim
        # select forward probabilities
        r_new = torch.index_select(r.view(-1, 2, n_bh * snum), 2, vidx).view(
            -1, 2, n_bh
        )
        return r_new, s_new, f_min, f_max

    def extend_prob(self, x):
        """Extend CTC prob.

        :param torch.Tensor x: input label posterior sequences (B, T, O)
        """

        if self.x.shape[1] < x.shape[1]:  # self.x (2,T,B,O); x (B,T,O)
            # Pad the rest of posteriors in the batch
            # TODO(takaaki-hori): need a better way without for-loops
            xlens = [x.size(1)]
            for i, l in enumerate(xlens):
                if l < self.input_length:
                    x[i, l:, :] = self.logzero
                    x[i, l:, self.blank] = 0
            tmp_x = self.x
            xn = x.transpose(0, 1)  # (B, T, O) -> (T, B, O)
            xb = xn[:, :, self.blank].unsqueeze(2).expand(-1, -1, self.odim)
            self.x = torch.stack([xn, xb])  # (2, T, B, O)
            self.x[:, : tmp_x.shape[1], :, :] = tmp_x
            self.input_length = x.size(1)
            self.end_frames = torch.as_tensor(xlens) - 1

    def extend_state(self, state):
        """Compute CTC prefix state.


        :param state    : CTC state
        :return ctc_state
        """

        if state is None:
            # nothing to do
            return state
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state

            r_prev_new = torch.full(
                (self.input_length, 2),
                self.logzero,
                dtype=self.dtype,
                device=self.device,
            )
            start = max(r_prev.shape[0], 1)
            r_prev_new[0:start] = r_prev
            for t in range(start, self.input_length):
                r_prev_new[t, 1] = r_prev_new[t - 1, 1] + self.x[0, t, :, self.blank]

            return (r_prev_new, s_prev, f_min_prev, f_max_prev)


class CTCPrefixScore(object):
    """Compute CTC label sequence scores

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, x, blank, eos, xp):
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state

        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = self.xp.logaddexp(
            r_prev[:, 0], r_prev[:, 1]
        )  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = self.xp.ndarray((self.input_length, len(cs)), dtype=np.float32)
            for i in range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilities log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in range(start, self.input_length):
            r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = (
                self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
            )
            log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = self.xp.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = self.xp.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.xp.rollaxis(r, 2)



class CTCPrefixScorer(torch.nn.Module):
    """Decoder interface wrapper for CTCPrefixScore."""

    def __init__(self, ctc: torch.nn.Module, eos: int):
        """Initialize class.

        Args:
            ctc (torch.nn.Module): The CTC implementation.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.

        """
        self.ctc = ctc
        self.eos = eos
        self.impl = None

    def init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        # TODO(karita): use CTCPrefixScoreTH
        self.impl = CTCPrefixScore(logp, 0, self.eos, np)
        return 0, self.impl.initial_state()

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary

        Returns:
            state: pruned state

        """
        if type(state) == tuple:
            if len(state) == 2:  # for CTCPrefixScore
                sc, st = state
                return sc[i], st[i]
            else:  # for CTCPrefixScoreTH (need new_id > 0)
                r, log_psi, f_min, f_max, scoring_idmap = state
                s = log_psi[i, new_id].expand(log_psi.size(1))
                if scoring_idmap is not None:
                    return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
                else:
                    return r[:, :, i, new_id], s, f_min, f_max
        return None if state is None else state[i]

    def score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        prev_score, state = state
        presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
        tscore = torch.as_tensor(
            presub_score - prev_score, device=x.device, dtype=x.dtype
        )
        return tscore, (presub_score, new_st)

    def batch_init_state(self, x: torch.Tensor):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))  # assuming batch_size = 1
        xlen = torch.tensor([logp.size(1)])
        self.impl = CTCPrefixScoreTH(logp, xlen, 0, self.eos)
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            ids (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        batch_state = (
            (
                torch.stack([s[0] for s in state], dim=2),
                torch.stack([s[1] for s in state]),
                state[0][2],
                state[0][3],
            )
            if state[0] is not None
            else None
        )
        return self.impl(y, batch_state, ids)

    def extend_prob(self, x: torch.Tensor):
        """Extend probs for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            x (torch.Tensor): The encoded feature tensor

        """
        logp = self.ctc.log_softmax(x.unsqueeze(0))
        self.impl.extend_prob(logp)

    def extend_state(self, state):
        """Extend state for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            state: The states of hyps

        Returns: exteded state

        """
        new_state = []
        for s in state:
            new_state.append(self.impl.extend_state(s))

        return new_state


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices_list = indices.cpu().tolist()
        
        indices_filtered = [i for i in indices_list if i != self.blank]
        joined = "".join([self.labels[i] for i in indices_filtered])
        return joined.replace("|", " ").strip().split()