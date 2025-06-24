import os
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample
import librosa
import numpy as np
from tqdm import tqdm
import logging

# 설정값 (config.yaml 파일에서 가져오거나 직접 설정)
class Config:
    def __init__(self):
        self.sample_rate = 16000
        self.n_mels = 80
        self.n_fft = 400
        self.win_length = 400
        self.hop_length = 160
        self.scp_dir = "/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/scp/순천향대test/"  # scp 파일이 있는 디렉토리
        self.subset = "testclean"  # train, dev, test 등
        self.output_dir = "./stats/순천향대test/"  # 평균/표준편차를 저장할 디렉토리

def load_scp(scp_path):
    """Load dataset manifest file"""
    items = []
    with open(scp_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = line.strip().split('|')
            if len(item) >= 1:
                audio_path = item[0]
                items.append(audio_path)
    return items

def compute_mean_std(config):
    """Computes mean and std deviation of the dataset features from a given dataset."""
    logging.info("Calculating mean and std of training features for normalization...")

    # Feature extractor
    feature_extractor = MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels
    )

    scp_path = os.path.join(config.scp_dir, f"{config.subset}_token.scp")
    audio_paths = load_scp(scp_path)

    sums = 0.0
    sum_sqs = 0.0
    count = 0

    for audio_path in tqdm(audio_paths, desc="Calculating Mean/Std"):
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if needed
            if sample_rate != config.sample_rate:
                resampler = Resample(orig_freq=sample_rate, new_freq=config.sample_rate)
                waveform = resampler(waveform)

            # Extract features
            features = feature_extractor(waveform)
            features = torch.log(features + 1e-6)  # log mel spectrogram

            # Sum and count
            sums += features.sum(dim=(0, 2))  # (n_mels,)
            sum_sqs += (features ** 2).sum(dim=(0, 2))  # (n_mels,)
            count += features.shape[0] * features.shape[2]

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    if count == 0:
        logging.warning("No valid features found for mean/std calculation. Setting mean/std to 0/1.")
        mean = torch.zeros(config.n_mels, dtype=torch.float32)
        std = torch.ones(config.n_mels, dtype=torch.float32)
    else:
        mean = sums / count
        std = torch.sqrt((sum_sqs / count) - (mean ** 2))
        std = torch.max(std, torch.tensor(1e-5))

    logging.info(f"Computed training feature mean (first 5 elements): {[f'{x:.4f}' for x in mean[:5]]}")
    logging.info(f"Computed training feature std (first 5 elements): {[f'{x:.4f}' for x in std[:5]]}")

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    mean_file = os.path.join(output_dir, "train_mean.pt")
    std_file = os.path.join(output_dir, "train_std.pt")

    torch.save(mean, mean_file)
    torch.save(std, std_file)
    logging.info(f"Mean saved to: {mean_file}")
    logging.info(f"Std saved to: {std_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config()
    compute_mean_std(config)