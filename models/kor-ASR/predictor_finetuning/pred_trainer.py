"""
🖤🐰 JaeEun Baik, 2025
"""

import os
import logging
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from attrdict import AttrDict # config 로드를 위해 필요

from pred_datamodule import PredictorDataModule # 새 DataModule 임포트
from pred_modelmodule import PredictorModelModule # 새 ModelModule 임포트

# 기본 설정 (기존 trainer.py와 유사)
torch.backends.cudnn.benchmark = True

class PredictorTrainer:
    def __init__(self, config, base_asr_model_ckpt: str):
        self.config = config
        self.base_asr_model_ckpt = base_asr_model_ckpt # 파인튜닝할 ASR 모델 체크포인트 경로

        # DataModule 초기화
        self.datamodule = PredictorDataModule(config)
        
        # Predictor 파인튜닝 ModelModule 초기화
        self.model = PredictorModelModule(config, self.base_asr_model_ckpt)
        
        # 로깅 설정
        logger = WandbLogger(
            project=config.trainer.proj + "_Predictor_Finetune", 
            name=config.trainer.exp_name + "_Predictor_Finetune" 
        )
        
        callbacks = self._setup_callbacks()
        strategy = DDPStrategy(find_unused_parameters=False) 
        
        trainer_kwargs = {
            'max_epochs': config.trainer.num_epochs, 
            'accelerator': 'gpu' if config.trainer.gpus > 0 else 'cpu',
            'devices': config.trainer.gpus if config.trainer.gpus > 0 else None,
            'logger': logger,
            'callbacks': callbacks,
            'log_every_n_steps': config.trainer.log_every_n_steps,
            'val_check_interval': config.trainer.val_check_interval,
            'precision': config.trainer.precision, # BF16/FP16 정밀도
            'accumulate_grad_batches': config.trainer.accumulate_grad_batches,
            'gradient_clip_val': config.trainer.gradient_clip_val,
            'strategy': strategy,
            'reload_dataloaders_every_n_epochs': config.trainer.reload_dataloaders_every_n_epochs,
        }

        self.trainer = pl.Trainer(**trainer_kwargs)

    def _setup_callbacks(self):
        callbacks = []
        checkpoint_dir = os.path.join(os.path.dirname(self.config.checkpoint.model_save_path), "predictor_finetune")
        os.makedirs(checkpoint_dir, exist_ok=True) 

        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='{epoch:02d}-{val_lm_loss:.4f}',
                monitor='val_lm_loss',
                save_top_k=self.config.checkpoint.save_top_k,
                mode='min',
                save_last=True
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        return callbacks

    def finetune(self):
        logging.info(f"Starting Predictor finetuning for {self.config.trainer.max_epochs} epochs")
        # resume_from_checkpoint는 파인튜닝 자체를 이어갈 때 사용
        self.trainer.fit(self.model, self.datamodule, ckpt_path=self.config.trainer.get('resume_finetune_ckpt_path', None))
        
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            logging.info(f"Best finetuned predictor model saved at: {best_model_path}")
            finetune_model_path_file = os.path.join(os.path.dirname(self.config.trainer.model_save_path), 'best_finetuned_predictor_path.txt')
            with open(finetune_model_path_file, 'w') as f:
                f.write(best_model_path)

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from: {config_path}")
        return AttrDict(config)

def main():
    parser = argparse.ArgumentParser(description="Predictor Finetuning Script for RNN-T ASR")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file for finetuning.')
    parser.add_argument('--base_asr_ckpt', type=str, required=True,
                        help='Path to the base ASR model checkpoint to finetune its predictor.')
    args = parser.parse_args()
    
    # 로깅 설정 (메인 스크립트에서)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = PredictorTrainer.load_config(args.config)
    
    finetuner = PredictorTrainer(config, args.base_asr_ckpt)
    finetuner.finetune()

if __name__ == "__main__":
    main()