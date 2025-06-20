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
        # Load data paths
        self.scp_path = self.data_config.scp_dir + f"{subset}_token.scp"
        self.items = self._load_scp(self.scp_path)
        
        self.sample_rate = self.data_config.sample_rate
        self.n_mels = self.data_config.n_mels
        self.n_fft = self.data_config.n_fft
        self.win_length = self.data_config.win_length
        self.hop_length = self.data_config.hop_length
        
        # Token processor
        self.token_processor = TokenProcessor(self.data_config.tokenizer)
        
        # Feature extractor
        self.feature_extractor = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        self.augmentation = self.data_config.augmentation
        self.compute_stats_only = compute_stats_only
        if not self.compute_stats_only:
            self._init_augmentations()
        else: 
            self.noise_paths = None
            self.rir_data = None
            self.speed_perturbation = None 
        self.mean = None
        self.std = None
        if precomputed_mean is not None and precomputed_std is not None:
            self.mean = torch.tensor(precomputed_mean, dtype=torch.float32)
            self.std = torch.tensor(precomputed_std, dtype=torch.float32)
        
        
    def _init_augmentations(self):
        """Initialize augmentation methods based on config"""
        # Noise mixing augmentation
        if self.augmentation.noise_mixing:
            self.noise_paths = self._load_noise_files(self.augmentation.noise_dir)
            self.noise_prob = self.augmentation.noise_prob
            self.noise_level = self.augmentation.noise_level
        else:
            self.noise_paths = None
            
        # RIR mixing augmentation
        if self.augmentation.rir_mixing:
            self.rir_dir = self.augmentation.rir_dir
            self.rir_prob = self.augmentation.rir_prob
            self.RT_list = self.augmentation.RT_list
            self.rir_data = self._load_rir_data(self.rir_dir, self.RT_list)
        else:
            self.rir_data = None
            
        # SpecAugment
        if self.augmentation.specaugment:
            self.time_mask_param = self.augmentation.time_mask_param
            self.freq_mask_param = self.augmentation.freq_mask_param
            self.n_time_masks = self.augmentation.n_time_masks
            self.n_freq_masks = self.augmentation.n_freq_masks
            
            self.time_masking = TimeMasking(time_mask_param=self.time_mask_param, p=1.0)
            self.freq_masking = FrequencyMasking(freq_mask_param=self.freq_mask_param)
        
        # Speed perturbation
        if self.augmentation.speed_perturb:
            self.speed_factors = self.augmentation.speed_factors
            self.speed_prob = self.augmentation.speed_prob
            self.speed_perturbation = SpeedPerturbation(self.sample_rate, self.speed_factors)
    
        if self.augmentation.gaussian_noise:
            self.gaussian_noise_prob = self.augmentation.gnoise_prob
            self.gaussian_noise_std = self.augmentation.gnoise_std
        
    def _load_scp(self, scp_path):
        """Load dataset manifest file"""
        items = []
        with open(scp_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = line.strip().split('|')
                if len(item) >= 2:
                    audio_path = item[0]
                    token = item[1]
                    token_tensors = torch.tensor(list(map(int, token.split(' '))))
                    items.append({
                        'audio_path': audio_path,
                        'token': token_tensors
                    })
        return items
    
    def _load_noise_files(self, noise_scp):
        """Load noise files for augmentation"""
        noise_files = []
        # if noise_dir and os.path.exists(noise_dir):
        #     for file in os.listdir(noise_dir):
        #         if file.endswith(('.wav', '.flac', '.mp3')):
        #             noise_files.append(os.path.join(noise_dir, file))
        with open(noise_scp, '+r') as f:
            for line in f.readlines():
                noise_files.append(line.split('\n')[0])
        return noise_files

    
    def _load_rir_data(self, rir_path, RT_list):
        """
        Load Room Impulse Response data from file
        
        Args:
            rir_path: Path to RIR scp file
            RT_list: List of RT60 values
            
        Returns:
            List of loaded RIR files
        """
        try:
            with open(rir_path, 'r') as f:
                rir_list = [line.strip() for line in f.readlines()]
            
            if not rir_list:
                print("Warning: No RIR files found in", rir_path)
                return None
                
            rir_data = []
            for file in rir_list:
                if os.path.exists(file):
                    RIR, sr = librosa.load(file, sr=self.sample_rate, mono=True)
                    RIR = RIR / np.sqrt(np.sum(RIR**2) + 1e-9)
                    rir_data.append(RIR)
                else:
                    print(f"Warning: RIR file not found: {file}")
                    
            return rir_data if rir_data else None
            
        except Exception as e:
            print(f"Error loading RIR data: {e}")
            return None
    
    def _apply_gaussian_noise(self, features: torch.Tensor) -> torch.Tensor:
        if self.gaussian_noise_prob > 0 and random.random() < self.gaussian_noise_prob:
            noise = torch.randn_like(features) * self.gaussian_noise_std
            features = features + noise
        return features
    
    def _apply_noise_mixing(self, waveform):
        """Add background noise to the audio"""
        if self.noise_paths and random.random() < self.noise_prob:
            noise_path = random.choice(self.noise_paths)
            noise, noise_sr = torchaudio.load(noise_path)
            
            if noise_sr != self.sample_rate:
                resampler = Resample(orig_freq=noise_sr, new_freq=self.sample_rate)
                noise = resampler(noise)
            
            if noise.shape[1] < waveform.shape[1]:
                factor = int(np.ceil(waveform.shape[1] / noise.shape[1]))
                noise = noise.repeat(1, factor)[:, :waveform.shape[1]]
            else:
                noise = noise[:, :waveform.shape[1]]
            
            noise_level = random.uniform(0, self.noise_level)
            waveform = (1 - noise_level) * waveform + noise_level * noise
        return waveform
    
    def _apply_rir_mixing(self, waveform):
        # waveform: [channels, T] tensor (e.g., [1, T])
        if self.rir_data and random.random() < self.rir_prob:
            rir_np = random.choice(self.rir_data) 
            rir_tensor = torch.from_numpy(rir_np).to(waveform.dtype) 

            if len(rir_tensor.shape) > 1:
                rir_tensor = rir_tensor.flatten()
            
            rir_tensor = rir_tensor.unsqueeze(0) 


            try:
                reverberated = torchaudio.functional.convolve(waveform, rir_tensor, mode='full')
                
                start_idx = (reverberated.shape[-1] - waveform.shape[-1]) // 2
                reverberated = reverberated[:, start_idx : start_idx + waveform.shape[-1]]
            except AttributeError: 
                waveform_np = waveform.numpy()
                if len(rir_np.shape) > 1:
                    rir_np = rir_np.flatten()
                
                temp = ss.convolve(waveform_np[0], rir_np, mode='full') 

                start_idx_np = (temp.shape[-1] - waveform_np.shape[-1]) // 2
                temp = temp[start_idx_np : start_idx_np + waveform_np.shape[-1]]

                if np.max(np.abs(temp)) > 1e-6:
                    temp = temp / np.max(np.abs(temp)) * np.max(np.abs(waveform_np))
                
                reverberated = torch.from_numpy(np.expand_dims(temp, axis=0)).to(waveform.dtype) 

            if reverberated.abs().max() > 1e-6: 
                reverberated = reverberated / reverberated.abs().max() * waveform.abs().max()

            return reverberated

        return waveform

          
    def _apply_specaugment(self, spec):
        """Apply SpecAugment to the spectrogram"""
        if hasattr(self, 'time_masking'):
            # Apply time masking
            for _ in range(self.n_time_masks):
                spec = self.time_masking(spec)
            
            # Apply frequency masking
            for _ in range(self.n_freq_masks):
                spec = self.freq_masking(spec)
                
        return spec
    
    def _apply_speed_perturbation(self, waveform):
        """Apply speed perturbation to the audio"""
        new_len = None
        if hasattr(self, 'speed_perturbation') and random.random() < self.speed_prob:
            length = torch.tensor([waveform.shape[-1]], dtype=torch.float32)
            waveform, new_len = self.speed_perturbation(waveform, length)
        return waveform, new_len
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        # Load audio
        waveform, sample_rate = torchaudio.load(item['audio_path'])
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Apply audio-level augmentations
        if not self.compute_stats_only:
            if self.augmentation.get('speed_perturb', False):
                (waveform, _) = self._apply_speed_perturbation(waveform)
                
            if self.augmentation.get('noise_mixing', False):
                waveform = self._apply_noise_mixing(waveform)

            if self.augmentation.get('rir_mixing', False):
                waveform = self._apply_rir_mixing(waveform)
            
        features = self.feature_extractor(waveform)
        feat_len = features.shape[2]        
        # Convert to log mel spectrogram
        features = torch.log(features + 1e-6)
        
        if not self.compute_stats_only:
            if self.augmentation.get('gaussian_noise', False):
                features = self._apply_gaussian_noise(features)
        
        mean_expanded = self.mean.unsqueeze(0).unsqueeze(2) # (80,) -> (1, 80, 1)
        std_expanded = self.std.unsqueeze(0).unsqueeze(2)   # (80,) -> (1, 80, 1)
        
        if self.mean is not None and self.std is not None:
            features = (features - mean_expanded) / (std_expanded + 1e-5)
        # Process target text
        # target = self.token_processor(item['text'])
        # target_len = len(target)
        
        target = item['token']
        target_len = len(target)
        
        return features.squeeze(0).transpose(0, 1), feat_len, target, target_len

