import os
import torch
from tqdm import tqdm
import logging
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
        
        self.train_mean = self.data_config.get('audio_feature_mean', None) # config의 키 이름과 맞춤
        self.train_std = self.data_config.get('audio_feature_std', None)   # config의 키 이름과 맞춤 (오타 수정)
        
        if isinstance(self.train_mean, str) and os.path.exists(self.train_mean):
            logging.info(f"Loading precomputed mean from {self.train_mean}")
            self.train_mean = torch.load(self.train_mean)
        if isinstance(self.train_std, str) and os.path.exists(self.train_std):
            logging.info(f"Loading precomputed std from {self.train_std}")
            self.train_std = torch.load(self.train_std)
    
    def setup(self, stage=None):
        precomputed_mean_val = self.train_mean # __init__에서 config에서 로드되었거나, 파일에서 로드된 값
        precomputed_std_val = self.train_std   # __init__에서 config에서 로드되었거나, 파일에서 로드된 값

        # 1. 만약 config에도 없고, 이전에 계산된 적도 없다면 (첫 실행 시)
        if (stage == 'fit' or stage is None) and \
           (not isinstance(precomputed_mean_val, torch.Tensor) or precomputed_mean_val is None): # mean이 텐서가 아니거나 None일 때
            
            # 통계 계산을 위해 임시 Dataset 생성 (증강 비활성화)
            temp_train_dataset_for_stats = Dataset(self.data_config, 'train', compute_stats_only=True)
            self._compute_mean_std(temp_train_dataset_for_stats) # mean/std 계산 및 self.train_mean/std에 저장
            
            # 이제 self.train_mean/std에 계산된 텐서가 저장됨
            precomputed_mean_val = self.train_mean
            precomputed_std_val = self.train_std
            
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(self.data_config, 'train',
                                         precomputed_mean=precomputed_mean_val, precomputed_std=precomputed_std_val)
            self.val_dataset = Dataset(self.data_config, 'dev',
                                       precomputed_mean=precomputed_mean_val, precomputed_std=precomputed_std_val)
        
        if stage == 'test' or stage is None:
            if self.test_clean:
                self.test_dataset = Dataset(self.data_config, 'testclean',
                                            precomputed_mean=precomputed_mean_val, precomputed_std=precomputed_std_val)
            else: 
                self.test_datset = Dataset(self.data_config, 'testother',
                                           precomputed_mean=precomputed_mean_val, precomputed_std=precomputed_std_val)
        if stage == 'validate':
            self.val_dataset = Dataset(self.data_config, 'dev')
            
            
    def _compute_mean_std(self, dataset: Dataset):
        """Computes mean and std deviation of the dataset features from a given dataset."""
        logging.info("Calculating mean and std of training features for normalization...")
        
        # 임시 DataLoader를 사용하여 데이터셋을 순회합니다.
        stats_dataloader = DataLoader(
            dataset, # compute_stats_only=True로 초기화된 Dataset
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False, 
            collate_fn=self._collate_fn, # 기존 _collate_fn 사용
            persistent_workers=True if self.num_workers > 0 else False
        )

        sums = 0.0
        sum_sqs = 0.0
        count = 0

        # tqdm으로 진행률 표시
        for features, feature_lengths, _, _ in tqdm(stats_dataloader, desc="Calculating Mean/Std"):
            # features: (B, T, D), feature_lengths: (B,)
            
            # 각 샘플의 실제 길이까지만 사용하여 평균/분산 계산
            current_batch_size = features.size(0)
            
            for i in range(current_batch_size):
                length = feature_lengths[i].item() # 실제 Feature 길이 가져오기
                if length == 0: # 길이가 0인 샘플은 건너뜁니다.
                    continue
                valid_features = features[i, :length, :] # (length, feature_dim)
                sums += valid_features.sum(dim=0) # (feature_dim,)
                sum_sqs += (valid_features ** 2).sum(dim=0) # (feature_dim,)
                count += length # 총 유효 시간 스텝 수

        # 최종 평균과 표준편차 계산
        if count == 0:
            logging.warning("No valid features found for mean/std calculation. Setting mean/std to 0/1.")
            mean = torch.zeros(dataset.n_mels, dtype=torch.float32)
            std = torch.ones(dataset.n_mels, dtype=torch.float32)
        else:
            mean = sums / count
            std = torch.sqrt((sum_sqs / count) - (mean ** 2))
            
            # NaN 방지 (std가 0에 가까울 때 또는 일부 차원의 분산이 0일 때)
            std = torch.max(std, torch.tensor(1e-5, device=std.device))

        self.train_mean = mean.tolist() # 리스트로 저장하여 Dataset에 전달하기 위함 (float32 텐서에서)
        self.train_std = std.tolist() # 리스트로 저장

        logging.info(f"Computed training feature mean (first 5 elements): {[f'{x:.4f}' for x in self.train_mean[:5]]}")
        logging.info(f"Computed training feature std (first 5 elements): {[f'{x:.4f}' for x in self.train_std[:5]]}")
        # 전체를 출력하면 너무 길어질 수 있으므로 일부만 출력
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../stats") # 예: stats 폴더에 저장    
        os.makedirs(output_dir, exist_ok=True)
        mean_file = os.path.join(output_dir, "train_mean.pt")
        std_file = os.path.join(output_dir, "train_std.pt")
        
        torch.save(torch.tensor(self.train_mean, dtype=torch.float32), mean_file)
        torch.save(torch.tensor(self.train_std, dtype=torch.float32), std_file)
        logging.info(f"Mean saved to: {mean_file}")
        logging.info(f"Std saved to: {std_file}")
        
        
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
        
        # Initialize padded tensors
        padded_features = torch.nn.utils.rnn.pad_sequence(
            features,
            batch_first=True,
            padding_value=0.0 # 멜 스펙트로그램의 패딩 값
        )
        
        feature_lengths = torch.tensor(feature_lengths) # zip에서 나온 튜플을 텐서로 변환

        # targets 패딩 (기존 코드와 동일)
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=0 # 0 또는 모델의 ignore_id에 맞는 값 (토크나이저의 pad_id)
        )
        target_lengths = torch.tensor(target_lengths)

        return padded_features, feature_lengths, padded_targets, target_lengths