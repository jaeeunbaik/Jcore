"""
🖤🐰 JaeEun Baik, 2025
"""

import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import logging
import sentencepiece as spm

from modules.e2e_asr_model import e2eASR # 기존 ASR 모델 임포트

class PredictorModelModule(pl.LightningModule):
    def __init__(self, config, base_asr_model_ckpt_path: str):
        super().__init__()
        self.config = config
        self.base_asr_model_ckpt_path = base_asr_model_ckpt_path

        # 1. 기존 ASR 모델 로드
        logging.info(f"Loading base ASR model from checkpoint: {self.base_asr_model_ckpt_path}")
        
        self.model = e2eASR(config.asr, config.data.tokenizer)
        checkpoint = torch.load(base_asr_model_ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False) # strict=False로 유연하게 로드

        for param in self.model.encoder.parameters():
            param.requires_grad = False
        logging.info("Encoder parameters frozen.")

        for param in self.model.joiner.parameters():
            param.requires_grad = False
        logging.info("Joiner parameters frozen.")
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.eos) 
        self.sp_processor = spm.SentencePieceProcessor()
        self.sp_processor.load(config.data.tokenizer)
        self.char_list = [self.sp_processor.id_to_piece(i) for i in range(config.asr.decoder.odim)]

        self.save_hyperparameters(config) # 하이퍼파라미터 저장

    def forward(self, input_tokens, input_lengths, target_tokens, target_lengths):
        pred_out, _ = self.model.predictor(y=input_tokens, y_lengths=input_lengths)
        
        if not hasattr(self, 'lm_head'):
            self.lm_head = nn.Linear(self.model.predictor.output_linear.out_features, self.model.odim).to(self.device)
        
        lm_logits = self.lm_head(pred_out) # (B, U_in, Vocab_size)

        # CrossEntropyLoss는 logits와 target을 받습니다.
        # logits: (B, U_in, Vocab_size)
        # target_tokens: (B, U_target)
        
        # CE Loss는 2D logits와 1D target을 기대하므로 flatten
        loss = self.criterion(lm_logits.view(-1, lm_logits.size(-1)), target_tokens.view(-1))
        
        return loss

    def training_step(self, batch, batch_idx):
        input_tokens, input_lengths, target_tokens, target_lengths = batch
        loss = self(input_tokens, input_lengths, target_tokens, target_lengths)
        self.log("train_lm_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tokens, input_lengths, target_tokens, target_lengths = batch
        loss = self(input_tokens, input_lengths, target_tokens, target_lengths)
        self.log("val_lm_loss", loss, prog_bar=True, sync_dist=True)
        # WER/CER은 LM 파인튜닝 단계에서는 보통 계산하지 않음.
        # 전체 ASR 모델로 평가해야 하므로 이 단계에서는 불필요.

    def configure_optimizers(self):
        # Predictor (및 lm_head)의 파라미터만 학습합니다.
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), # requires_grad=True인 파라미터만
            lr=np.float64(self.config.optimizer.op_lr), # LM 파인튜닝에 사용할 학습률 (낮게 설정)
            betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6
        )
        # LM 파인튜닝은 보통 간단한 스케줄러 (예: StepLR, CosineAnnealing) 또는 고정 LR 사용
        # Conformer LR 스케줄을 사용한다면 적절히 조정
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.trainer.num_epochs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" # 'step' 단위로 스케줄러 업데이트
            }
        }