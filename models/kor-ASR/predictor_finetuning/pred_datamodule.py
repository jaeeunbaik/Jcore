"""
🖤🐰 JaeEun Baik, 2025
"""

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import logging
from typing import Tuple, List

from pred_dataset import PredictorDataset # 새로 정의한 Dataset 클래스 임포트

class PredictorDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_config = config.data
        self.dataloader_config = config.dataloader
        self.config = config

        self.batch_size = self.dataloader_config.batch_size
        self.num_workers = self.dataloader_config.num_workers
        self.pin_memory = self.dataloader_config.pin_memory

        self.sos_id = config.asr.sos_id if hasattr(config.asr, 'sos_id') else 2
        self.eos_id = config.asr.eos_id if hasattr(config.asr, 'eos_id') else 3
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PredictorDataset(
                self.data_config, 'train', sos_id=self.sos_id, eos_id=self.eos_id
            )
            self.val_dataset = PredictorDataset(
                self.data_config, 'dev', sos_id=self.sos_id, eos_id=self.eos_id
            )

    def _collate_fn(self, batch_list: List[Tuple[torch.Tensor, int, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collate function for Predictor finetuning.
        Pads sequences and returns only input_tokens and target_tokens.
        
        Expected input from Dataset.__getitem__:
        (input_for_predictor, input_for_predictor_len, target_for_loss, target_for_loss_len)
        """
        # 길이에 따라 정렬 (pack_padded_sequence 사용을 위해)
        batch_list.sort(key=lambda x: x[1], reverse=True)
        
        inputs, input_lengths, targets, target_lengths = zip(*batch_list)
        
        # Predictor 입력 (padded_inputs) 패딩
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs,
            batch_first=True,
            padding_value=self.sos_id # Predictor 입력 패딩은 SOS나 Blank로 하는 경우가 많음.
                                      # 여기서는 <s>로 패딩
        )
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
        
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=self.eos_id # Target 패딩은 EOS로 하는 경우가 많음
        )
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)

        return padded_inputs, input_lengths, padded_targets, target_lengths


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )