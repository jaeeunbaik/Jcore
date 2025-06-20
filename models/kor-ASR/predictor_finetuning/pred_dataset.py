"""
🖤🐰 JaeEun Baik, 2025
"""
import torch
from torch.utils.data import Dataset
import logging # 로그 출력을 위해 추가
import sentencepiece as spm # SentencePiece 모델 로드를 위해 추가
import os # 파일 존재 여부 확인을 위해 추가
from typing import List, Tuple # 타입 힌트 추가


class PredictorDataset(Dataset):
    def __init__(self, data_config, subset, sos_id: int = 2, eos_id: int = 3): # SOS/EOS ID를 인자로 받음
        """
        Dataset for Next Token Prediction (Predictor Finetuning).
        
        Args:
            data_config: Configuration dictionary for data loading (expected to have scp_dir, tokenizer path)
            subset (str): Subset name (e.g., 'train', 'dev')
            sos_id (int): ID for Start-of-Sentence token.
            eos_id (int): ID for End-of-Sentence token.
        """
        super().__init__()
        self.data_config = data_config
        self.scp_path = self.data_config.scp_dir + f"{subset}_token.scp" # 텍스트 토큰 SCP 파일 경로
        self.items = self._load_scp(self.scp_path) # 토큰 ID 시퀀스들을 로드

        self.sos_id = sos_id
        self.eos_id = eos_id
        

    def _load_scp(self, scp_path: str) -> List[torch.Tensor]:
        """
        Load dataset manifest file where each line contains space-separated token IDs.
        
        Example line in scp_path: "10 20 30 40 50" (representing a sentence)
        """
        items = []
        if not os.path.exists(scp_path):
            logging.error(f"SCP file not found: {scp_path}")
            raise FileNotFoundError(f"SCP file not found: {scp_path}")

        with open(scp_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    # 토큰 ID 문자열을 공백으로 분리하여 정수 리스트로 변환
                    token_ids = list(map(int, line.split(' ')))
                    token_tensors = torch.tensor(token_ids, dtype=torch.int64) # LongTensor가 더 일반적
                    items.append(token_tensors) # 'token' 키 없이 바로 텐서를 리스트에 추가
                except ValueError as ve:
                    logging.warning(f"Skipping line {line_num + 1} due to ValueError (invalid token ID format): {line} - {ve}")
                except Exception as e:
                    logging.warning(f"Skipping line {line_num + 1} due to unexpected error: {line} - {e}")
        
        logging.info(f"Loaded {len(items)} token sequences from {scp_path}")
        return items
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns input and target sequences for next token prediction.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - input_tokens (torch.Tensor): Sequence for the Predictor input (e.g., [<s>, t1, t2, ..., tn-1])
            - input_lengths (torch.Tensor): Length of input_tokens
            - target_tokens (torch.Tensor): Target sequence for loss calculation (e.g., [t1, t2, ..., tn, </s>])
            - target_lengths (torch.Tensor): Length of target_tokens
        """
        original_tokens = self.items[idx] # 이미 torch.Tensor임

        # SOS와 EOS 토큰을 추가합니다.
        # Predictor 입력은 [<s>, t1, t2, ..., tn-1]
        # Loss 타겟은 [t1, t2, ..., tn, </s>]
        
        # 1. Predictor 입력 시퀀스: SOS + original_tokens
        input_for_predictor = torch.cat(
            (torch.tensor([self.sos_id], dtype=torch.int64), original_tokens), dim=0
        )
        input_for_predictor_len = len(input_for_predictor)

        # 2. Loss 타겟 시퀀스: original_tokens + EOS
        target_for_loss = torch.cat(
            (original_tokens, torch.tensor([self.eos_id], dtype=torch.int64)), dim=0
        )
        target_for_loss_len = len(target_for_loss)

        
        return input_for_predictor, input_for_predictor_len, target_for_loss, target_for_loss_len