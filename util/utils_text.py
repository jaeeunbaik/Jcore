import re
import os
import json
import logging
import sys
from itertools import groupby

import torch
import numpy as np
import editdistance
import sentencepiece as spm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass




class TokenProcessor:
    """Process text to BPE token ids using SentencePiece"""
    
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def __call__(self, text: str):
        ids = self.sp.encode(text, out_type=int)  # text2id
        return torch.tensor(ids, dtype=torch.long)
    
    def id2text(self, tokens: List[int], filter_blank=False) -> str:
        """
        토큰 ID 목록을 텍스트로 변환하는 함수
        범위를 벗어난 토큰은 자동으로 필터링합니다.
        """
        try:
            # 입력이 텐서인 경우 리스트로 변환
            if torch.is_tensor(tokens):
                tokens = tokens.tolist()
            
            max_token_id = self.sp.get_piece_size() - 1
            valid_tokens = []
            
            for t in tokens:
                if filter_blank and t == 0:
                    continue
                elif t == -1:
                    continue
                elif 0 <= t <= max_token_id:
                    valid_tokens.append(t)
                    
            if not valid_tokens:
                return ""
                
            return self.sp.decode(valid_tokens)
        
        except Exception as e:
            print(f"[ERROR] id2text 함수에서 오류 발생: {str(e)}")
            print(f"[DEBUG] 원본 토큰: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            return ""



class ErrorCalculator(object):
    """Calculate CER and WER for E2E_ASR and CTC models during training.

    :param y_hats: numpy array with predicted text
    :param y_pads: numpy array with true (target) text
    :param char_list:
    :param sym_space:
    :param sym_blank:
    :return:
    """

    def __init__(
        self, char_list, sym_space, sym_blank, report_cer=False, report_wer=False
    ):
        """Construct an ErrorCalculator object."""
        super(ErrorCalculator, self).__init__()

        self.report_cer = report_cer
        self.report_wer = report_wer

        self.char_list = char_list
        self.space = sym_space
        self.blank = sym_blank
        # NOTE (Shih-Lun): else case is for OpenAI Whisper ASR model,
        #                  which doesn't use <blank> token
        if self.blank in self.char_list:
            self.idx_blank = self.char_list.index(self.blank)
        else:
            self.idx_blank = None
        if self.space in self.char_list:
            self.idx_space = self.char_list.index(self.space)
        else:
            self.idx_space = None

    def __call__(self, ys_hat, ys_pad, is_ctc=False):
        """Calculate sentence-level WER/CER score.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :param bool is_ctc: calculate CER score for CTC
        :return: sentence-level WER score
        :rtype float
        :return: sentence-level CER score
        :rtype float
        """
        cer, wer = None, None
        if is_ctc:
            return self.calculate_cer_ctc(ys_hat, ys_pad)
        elif not self.report_cer and not self.report_wer:
            return cer, wer

        seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad)
        if self.report_cer:
            cer = self.calculate_cer(seqs_hat, seqs_true)

        if self.report_wer:
            wer = self.calculate_wer(seqs_hat, seqs_true)
        return cer, wer

    def calculate_cer_ctc(self, ys_hat, ys_pad):
        """Calculate sentence-level CER score for CTC.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: average sentence-level CER score
        :rtype float
        """

        cers, char_ref_lens = [], []
        for i, y in enumerate(ys_hat):
            y_hat = [x[0] for x in groupby(y)]
            y_true = ys_pad[i]
            seq_hat, seq_true = [], []
            for idx in y_hat:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_hat.append(self.char_list[int(idx)])

            for idx in y_true:
                idx = int(idx)
                if idx != -1 and idx != self.idx_blank and idx != self.idx_space:
                    seq_true.append(self.char_list[int(idx)])

            hyp_chars = "".join(seq_hat)
            ref_chars = "".join(seq_true)
            if len(ref_chars) > 0:
                cers.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

        cer_ctc = float(sum(cers)) / sum(char_ref_lens) if cers else None
        return cer_ctc

    def convert_to_char(self, ys_hat, ys_pad):
        """Convert index to character.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: token list of prediction
        :rtype list
        :return: token list of reference
        :rtype list
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            ymax = eos_true[0] if len(eos_true) > 0 else len(y_true)
            # NOTE: padding index (-1) in y_true is used to pad y_hat
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:ymax]]
            seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
            seq_hat_text = "".join(seq_hat).replace(self.space, " ")
            seq_hat_text = seq_hat_text.replace(self.blank, "")
            seq_true_text = "".join(seq_true).replace(self.space, " ")
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)
        return seqs_hat, seqs_true

    def calculate_cer(self, seqs_hat, seqs_true):
        """Calculate sentence-level CER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level CER score
        :rtype float
        """

        char_eds, char_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_chars = seq_hat_text.replace(" ", "")
            ref_chars = seq_true_text.replace(" ", "")
            char_eds.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))
        return float(sum(char_eds)) / sum(char_ref_lens)

    def calculate_wer(self, seqs_hat, seqs_true):
        """Calculate sentence-level WER score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level WER score
        :rtype float
        """

        word_eds, word_ref_lens = [], []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            hyp_words = seq_hat_text.split()
            ref_words = seq_true_text.split()
            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))
        return float(sum(word_eds)) / sum(word_ref_lens)



def preprocess_text(text):
    # 모든 문장 부호 제거
    text = re.sub(r'[,.!?:;\(\)\[\]"\'`]', '', text)
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip().lower()
    return text


def add_sos_eos(ys_pad, sos, eos, ignore_id):
    """Add <sos> and <eos> labels.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int sos: index of <sos>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from util.utils_module import pad_list

    _sos = ys_pad.new([sos])
    _eos = ys_pad.new([eos])
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


# torchaudio 내부의 FilePaths 클래스를 그대로 사용할 수 없으므로,
# 동일한 구조의 dataclass를 직접 정의합니다.
# 실제 torchaudio.models.decoder.ctc_decoder 함수는 이 구조를 기대합니다.
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

            # 파일명 또는 확장자를 기준으로 파일을 식별합니다.
            # 실제 사용되는 파일명/확장자에 따라 조건을 조정해야 합니다.
            if file == "lexicon.txt": # 예시 파일명: lexicon.txt
                found_lexicon = file_path
            elif file == "tokens.txt": # 예시 파일명: tokens.txt
                found_tokens = file_path
            elif file.endswith(".bin") and "lm" in file: # 예시 확장자: .bin (KenLM 바이너리)
                found_lm = file_path
            # 다른 언어 모델 파일 형식 (예: .arpa)이 있다면 추가

    return CustomFilePaths(lexicon=found_lexicon, tokens=found_tokens, lm=found_lm)

# 사용 예시:
if __name__ == "__main__":
    from torchaudio.models.decoder import download_pretrained_files
    
    files = download_pretrained_files("librispeech-4-gram")
    # 테스트를 위한 가상 디렉토리 생성
    # test_dir = "lm_files_test"
    # os.makedirs(test_dir, exist_ok=True)
    # with open(os.path.join(test_dir, "lexicon.txt"), "w") as f: f.write("test lexicon")
    # with open(os.path.join(test_dir, "tokens.txt"), "w") as f: f.write("test tokens")
    # with open(os.path.join(test_dir, "kenlm.bin"), "w") as f: f.write("test lm binary")
    # os.makedirs(os.path.join(test_dir, "subdir"), exist_ok=True)
    # with open(os.path.join(test_dir, "subdir", "another_lm.bin"), "w") as f: f.write("another lm")


    # # 함수 호출
    # file_paths = get_lm_file_paths(test_dir)

    # print(f"Lexicon Path: {file_paths.lexicon}")
    # print(f"Tokens Path: {file_paths.tokens}")
    # print(f"LM Path: {file_paths.lm}")

    # # 모든 파일이 존재하는지 확인 (간단한 테스트)
    # assert file_paths.lexicon is not None
    # assert file_paths.tokens is not None
    # assert file_paths.lm is not None

    # # 존재하지 않는 파일에 대한 테스트 (예: tokens.txt 삭제 후)
    # os.remove(os.path.join(test_dir, "tokens.txt"))
    # file_paths_missing = get_lm_file_paths(test_dir)
    # print(f"\nAfter deleting tokens.txt:")
    # print(f"Tokens Path: {file_paths_missing.tokens}")
    # assert file_paths_missing.tokens is None

    # # 생성된 디렉토리 삭제
    # import shutil
    # shutil.rmtree(test_dir)