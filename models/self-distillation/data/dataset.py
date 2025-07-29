# 🐰🖤
import os
import random
import logging

import torch
import torchaudio
import numpy as np
import librosa
import scipy.signal as ss

from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio.functional as AF
from torchaudio.transforms import MelSpectrogram, Resample, SpeedPerturbation
from torchaudio.transforms import TimeMasking, FrequencyMasking
from typing import Dict, List, Optional, Tuple

# nnAudio 임포트 (올바른 이름으로 수정)
from nnAudio.features import Gammatonegram

from util.utils_text import TokenProcessor


class Dataset(Dataset):
    def __init__(self, data_config, subset, precomputed_mean=None, precomputed_std=None, compute_stats_only=False):
        """
        Dataset for ASR with various augmentation options.
        
        Args:
            config_path: Path to the dataset configuration file
        """
        super().__init__()
        self.data_config = data_config
        self.subset = subset
        self.scp_path = self.data_config.scp_dir + f"{subset}_token.scp"
        self.items = self._load_scp(self.scp_path)
        
        self.sample_rate = self.data_config.sample_rate
        self.n_mels = self.data_config.n_mels
        self.n_fft = self.data_config.n_fft
        self.win_length = self.data_config.win_length
        self.hop_length = self.data_config.hop_length
        
        self.token_processor = TokenProcessor(self.data_config.tokenizer)
        
        feature_type = self.data_config.get('feature_type', 'mel')
        if feature_type == 'gammatone':
            self.feature_extractor = Gammatonegram(sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_bins=self.n_mels, window='hann', power=2, fmin=20, fmax=self.sample_rate // 2)
        elif feature_type == 'mel':
            self.feature_extractor = MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_mels=self.n_mels)
        else:
            raise ValueError(f"Unsupported feature_type: {feature_type}")
        
        self.compute_stats_only = compute_stats_only
        if not self.compute_stats_only:
            self.student_aug_config = self.data_config.get('student_augmentation')
            self.teacher_aug_config = self.data_config.get('teacher_augmentation')
            self._init_resources()

        self.normalization_enabled = self.data_config.get('normalization', {}).get('enabled', False)
        self.mean = None
        self.std = None
        if self.normalization_enabled:
            self.mean = torch.tensor(precomputed_mean, dtype=torch.float32) if precomputed_mean is not None else None
            self.std = torch.tensor(precomputed_std, dtype=torch.float32) if precomputed_std is not None else None

    def _init_resources(self):
        """Initialize shared resources for augmentation like noise and RIR files."""
        self.noise_paths = None
        self.rir_data = None

        # Load resources if they are defined in either student or teacher config
        student_noise = self.student_aug_config and self.student_aug_config.get('noise_mixing')
        teacher_noise = self.teacher_aug_config and self.teacher_aug_config.get('noise_mixing')
        if student_noise or teacher_noise:
            # Assume student config holds the path
            noise_dir = self.student_aug_config.noise_dir if student_noise else self.teacher_aug_config.noise_dir
            self.noise_paths = self._load_noise_files(noise_dir)

        student_rir = self.student_aug_config and self.student_aug_config.get('rir_mixing')
        teacher_rir = self.teacher_aug_config and self.teacher_aug_config.get('rir_mixing')
        if student_rir or teacher_rir:
            # Assume student config holds the path and params
            if student_rir:
                self.rir_data = self._load_rir_data(self.student_aug_config.rir_dir, self.student_aug_config.RT_list)
            else:
                self.rir_data = self._load_rir_data(self.teacher_aug_config.rir_dir, self.teacher_aug_config.RT_list)

    def _load_scp(self, scp_path):
        items = []
        with open(scp_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    items.append({'audio_path': parts[0], 'token': torch.tensor(list(map(int, parts[1].split())))})
        return items

    def _load_noise_files(self, noise_scp):
        with open(noise_scp, 'r') as f:
            return [line.strip() for line in f]

    def _load_rir_data(self, rir_path, RT_list):
        try:
            with open(rir_path, 'r') as f:
                rir_list = [line.strip() for line in f]
            if not rir_list: return None
            
            rir_data = []
            for file in rir_list:
                if os.path.exists(file):
                    RIR, _ = librosa.load(file, sr=self.sample_rate, mono=True)
                    rir_data.append(RIR / np.sqrt(np.sum(RIR**2) + 1e-9))
            return rir_data if rir_data else None
        except Exception as e:
            print(f"Error loading RIR data: {e}")
            return None

    
    def _apply_noise_mixing(self, waveform, aug_config):
        if aug_config and aug_config.get('noise_mixing') and self.noise_paths and random.random() < aug_config.noise_prob:
            noise_path = random.choice(self.noise_paths)
            noise, noise_sr = torchaudio.load(noise_path)
            if noise_sr != self.sample_rate:
                noise = Resample(orig_freq=noise_sr, new_freq=self.sample_rate)(noise)
            
            if noise.shape[1] < waveform.shape[1]:
                noise = noise.repeat(1, int(np.ceil(waveform.shape[1] / noise.shape[1])))
            noise = noise[:, :waveform.shape[1]]
            
            noise_level = random.uniform(*aug_config.noise_level)
            return (1 - noise_level) * waveform + noise_level * noise
        return waveform

    def _apply_rir_mixing(self, waveform, aug_config):
        if aug_config and aug_config.get('rir_mixing') and self.rir_data and random.random() < aug_config.rir_prob:
            rir_np = random.choice(self.rir_data)
            rir_tensor = torch.from_numpy(rir_np).float().unsqueeze(0)
            reverberated = F.conv1d(waveform.unsqueeze(0), rir_tensor, padding='same')
            return reverberated.squeeze(0)
        return waveform

    def _apply_specaugment(self, features, aug_config):
        if aug_config and aug_config.get('specaugment'):
            time_masking = TimeMasking(time_mask_param=aug_config.time_mask_param, p=1.0)
            freq_masking = FrequencyMasking(freq_mask_param=aug_config.freq_mask_param)
            for _ in range(aug_config.n_time_masks):
                features = time_masking(features)
            for _ in range(aug_config.n_freq_masks):
                features = freq_masking(features)
        return features

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio_path = item['audio_path']
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            logging.error(f"Error loading audio file {audio_path}: {e}")
            # Return a dummy item or skip
            return self.__getitem__((idx + 1) % len(self))

        if sample_rate != self.sample_rate:
            waveform = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(waveform)

        # --- Create Student and Teacher Features ---
        student_features, feat_len = self._get_augmented_features(waveform.clone(), self.student_aug_config)
        teacher_features, _ = self._get_augmented_features(waveform.clone(), self.teacher_aug_config)
        
        target = item['token']
        target_len = len(target)

        return student_features, teacher_features, feat_len, target, target_len, audio_path

    def _get_augmented_features(self, waveform, aug_config):
        # Apply waveform-level augmentations
        if not self.compute_stats_only:
            waveform = self._apply_noise_mixing(waveform, aug_config)
            waveform = self._apply_rir_mixing(waveform, aug_config)

        # Extract features
        features = self.feature_extractor(waveform)
        feat_len = features.shape[2]
        features = torch.log(features + 1e-6)

        # Normalize if enabled
        if self.normalization_enabled and self.mean is not None and self.std is not None:
            features = (features - self.mean.unsqueeze(0).unsqueeze(2)) / (self.std.unsqueeze(0).unsqueeze(2) + 1e-5)

        # Apply feature-level augmentations
        if not self.compute_stats_only:
            features = self._apply_specaugment(features, aug_config)
            
        return features.squeeze(0).transpose(0, 1), feat_len
