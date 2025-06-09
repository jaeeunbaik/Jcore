import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .dataset import Dataset


class ASRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        PyTorch Lightning DataModule for ASR data.
        
        Args:
            data_config: Configuration dictionary for data loading
        """
        super(ASRDataModule, self).__init__()
        self.data_config = config.data
        self.dataloader_config = config.dataloader
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_clean = self.data_config.test_clean
        
        # Dataloader parameters
        self.batch_size = self.dataloader_config.batch_size
        self.num_workers = self.dataloader_config.num_workers
        self.pin_memory = self.dataloader_config.pin_memory

    def setup(self, stage=None):
        """Setup datasets for each stage"""
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(self.data_config, 'train')
            self.val_dataset = Dataset(self.data_config, 'dev')
        
        if stage == 'test' or stage is None:
            if self.test_clean:
                self.test_dataset = Dataset(self.data_config, 'testclean')
            else: 
                self.test_datset = Dataset(self.data_config, 'testother')
        if stage == 'validate':
            self.val_dataset = Dataset(self.data_config, 'dev')
        
        
    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True
        )

    def test_dataloader(self):
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            persistent_workers=True
        )
    
    def _collate_fn(self, batch):
        """
        Custom collate function to handle variable length sequences
        
        Returns:
            tuple: (padded_features, feature_lengths, padded_targets)
        """
        # Sort batch by feature length (descending)
        batch.sort(key=lambda x: x[1], reverse=True)
        
        # Separate features, lengths, targets, target_lengths
        features, feature_lengths, targets, target_lengths = zip(*batch)
        
        # Get max lengths
        max_feat_len = max(feature_lengths)
        
        # Prepare padded batch
        batch_size = len(features)
        feature_dim = features[0].shape[1]
        
        # Initialize padded tensors
        padded_features = torch.nn.utils.rnn.pad_sequence(
            [f for f, _, _, _ in batch], # features만 추출
            batch_first=True,
            padding_value=0.0 # 멜 스펙트로그램의 패딩 값
        )
        
        feature_lengths = torch.tensor(feature_lengths) # zip에서 나온 튜플을 텐서로 변환

        # targets 패딩 (기존 코드와 동일)
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=-1 # 0 또는 모델의 ignore_id에 맞는 값 (토크나이저의 pad_id)
        )
        target_lengths = torch.tensor(target_lengths)

        return padded_features, feature_lengths, padded_targets, target_lengths