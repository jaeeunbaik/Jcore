# 🐰🖤
import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from modelmodule import ModelModule
from data.datamodule import ASRDataModule
from attrdict import AttrDict


torch.backends.cudnn.benchmark = True

class Trainer:
    def __init__(self, config):
        """
        Initialize the trainer with configurations
        
        Args:
            config: Complete configuration object
        """
        self.config = config
        
        # Initialize data module with proper data config
        self.datamodule = ASRDataModule(config)
        
        # Initialize model
        self.model = ModelModule(self.config)
        
        # Setup logging
        logger = WandbLogger(
            project=config.trainer.proj,
            name=config.trainer.exp_name
        )
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        strategy = DDPStrategy(find_unused_parameters=False)
        # Initialize PyTorch Lightning trainer
        trainer_kwargs = {
            'max_epochs': config.trainer.num_epochs,
            'accelerator': 'gpu' if config.trainer.gpus > 0 else 'cpu',
            'devices': config.trainer.gpus if config.trainer.gpus > 0 else None,
            'logger': logger,
            'callbacks': callbacks,
            'log_every_n_steps': config.trainer.log_every_n_steps,
            'val_check_interval': config.trainer.val_check_interval,
            'precision': config.trainer.precision,
            'accumulate_grad_batches': config.trainer.accumulate_grad_batches,
            'gradient_clip_val': config.trainer.gradient_clip_val,
            'strategy': strategy,
            'reload_dataloaders_every_n_epochs': config.trainer.reload_dataloaders_every_n_epochs
        }

        self.trainer = PLTrainer(**trainer_kwargs)

    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []

        checkpoint_path = os.path.dirname(self.config.trainer.ckpt_path)
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_path,
                filename='{epoch:02d}-{val_wer:.4f}',
                monitor='val_wer',
                save_top_k=self.config.checkpoint.save_top_k,
                mode='min',
                save_last=True
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        return callbacks

    def train(self):
        """Train the model"""
        print(f"Starting training for {self.config.trainer.num_epochs} epochs")
        
        # ckpt_path = self.config.trainer.ckpt_path
        if self.config.trainer.resume_from_checkpoint:
            self.trainer.fit(self.model, self.datamodule, ckpt_path=self.config.trainer.ckpt_path)
        else:
            self.trainer.fit(self.model, self.datamodule, ckpt_path=None)
        
        # Save best model path for easy reference
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Best model saved at: {best_model_path}")
            # Save the path to a file for easy loading later
            checkpoint_dir = os.path.dirname(self.config.checkpoint.model_save_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, 'best_model_path.txt'), 'w') as f:
                f.write(best_model_path)

    def validate(self, ckpt_path=None):    
        if ckpt_path is None and hasattr(self.trainer, 'checkpoint_callback'):
            # Use best checkpoint by default
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
            if not ckpt_path:
                print("Warning: No checkpoint found. Using current model state.")
        
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading checkpoint from: {ckpt_path}")
        
        self.trainer.validate(self.model, self.datamodule, ckpt_path=ckpt_path)
    
    
    def evaluate(self, ckpt_path=None):
        """
        Evaluate the model on test set
        
        Args:
            ckpt_path: Optional path to a specific checkpoint to load
        """
        if ckpt_path is None and hasattr(self.trainer, 'checkpoint_callback'):
            # Use best checkpoint by default
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
            if not ckpt_path:
                print("Warning: No checkpoint found. Using current model state.")
        
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading checkpoint from: {ckpt_path}")
        
        self.trainer.test(self.model, self.datamodule, ckpt_path=ckpt_path)

    @staticmethod
    def load_config(config_path):
        """
        Load YAML configuration file and resolve relative paths.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            AttrDict: Configuration as an attribute dictionary with resolved paths.
        """
        config_path_abs = os.path.abspath(config_path)
        
        # config.yaml 파일이 있는 디렉토리
        # 예: /home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/models/kor-ASR/
        config_file_dir = os.path.dirname(config_path_abs)
        
        # 여기서 두 폴더 상위 디렉토리로 이동
        # 예: /home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/
        base_dir_for_resolution = os.path.dirname(config_file_dir)# <-- 이 줄을 변경
        
        with open(config_path_abs, 'r') as f:
            config = yaml.safe_load(f)
        
        def resolve_paths(data, current_base_dir): # 인자명 변경: base_dir 대신 current_base_dir
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and (
                        key.endswith('_dir') or 
                        key.endswith('_path') or 
                        key.endswith('_file') or
                        key == 'tokenizer' # tokenizer 키도 상대경로일 수 있으므로 추가
                    ) and value.startswith('..'):
                        # 현재 base_dir_for_resolution을 기준으로 상대 경로를 해석
                        data[key] = os.path.abspath(os.path.join(current_base_dir, value))
                    else:
                        data[key] = resolve_paths(value, current_base_dir) # 재귀 호출 시에도 변경된 base_dir 전달
            elif isinstance(data, list):
                data = [resolve_paths(item, current_base_dir) for item in data]
            return data

        # config 딕셔너리 전체에 대해 경로 해석 적용
        config_resolved = resolve_paths(config, base_dir_for_resolution) # 변경된 base_dir 전달
        
        print(f"Loaded configuration from: {config_path}")
        print(f"Paths resolved relative to: {base_dir_for_resolution}") # 확인을 위한 로그 추가
        return AttrDict(config_resolved)


def main():
    """Main function for running the trainer from command line"""
    parser = argparse.ArgumentParser(description="ASR Training Script")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate', 'test', 'train_test'],
                        help='Run mode: train, test, or train then test')
    parser.add_argument('--ckpt', type=str, help='ckpt path to test')
    args = parser.parse_args()
    
    # Load configuration
    config = Trainer.load_config(args.config)
    
    # Create trainer with the entire config
    trainer = Trainer(config)
    
    # Run in specified mode
    if args.mode in ['train', 'train_test']:
        trainer.train()
    
    if args.mode in ['test', 'train_test']:
        trainer.evaluate(args.ckpt)
        
    if args.mode == 'validate':
        trainer.validate(args.ckpt)


if __name__ == "__main__":
    main()
