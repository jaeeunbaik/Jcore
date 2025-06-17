"""
🖤🐰 JaeEun Baik, 2025
"""
import re
import random 
import logging
import jiwer
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from typing import List

import sentencepiece as spm

from util.utils_text import TokenProcessor, ErrorCalculator, preprocess_text
from modules.e2e_asr_model import e2eASR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_config = config.model.asr
        self.kd_config = config.model.distillation
        self.use_kd = self.kd_config.using_distillation
        # self.kd_loss = KDLoss(model_config.distillation)
        # if self.use_kd:
        #     self.teacher_model = e2eASR(model_config.teacher)
        #     self.student_model = e2eASR(model_config.student)
            
        #     self.model = KDWrapper(self.teacher_model, self.student_model, self.kd_loss, model_config.distillation.target)
        # else:
        self.optim_config = config.optimizer
        self.lr = np.float64(self.optim_config.op_lr)
        self.tokenizer_path = config.data.tokenizer
        self.token_processor = TokenProcessor(config.data.tokenizer)
        self.trainer_config = config.trainer
        self.model = e2eASR(self.model_config, self.tokenizer_path)
        
        self.sp_processor = spm.SentencePieceProcessor()
        self.sp_processor.load(self.tokenizer_path)
        
        self.char_list = [self.sp_processor.id_to_piece(i) for i in range(self.model_config.decoder.odim)]
        self.error_calculator = ErrorCalculator(
            char_list=self.char_list,
            sym_space=self.model_config.sym_space,
            sym_blank=self.model_config.sym_blank,
            report_cer=self.model_config.report_cer,
            report_wer=self.model_config.report_wer
        )
      

    
    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch
        if self.use_kd:
            loss_dict = self.model(x, x_len, y, y_len)
            self.log("train/total_loss", loss_dict["total_loss"])
            self.log("train/kd_loss", loss_dict["kd_loss"])
            self.log("train/asr_loss", loss_dict["student_loss"])
            return loss_dict["total_loss"]
        else:
            loss = self.model(x, x_len, y, y_len)
            log_items = {
                "train/total_loss": loss.get("loss"),
                "train/cer": loss.get("cer"),
                "train/wer": loss.get("wer"),
            }
            
            # None이 아닌 값만 log로 넘김
            for key, value in log_items.items():
                if value is not None:
                    self.log(key, value, prog_bar=True, sync_dist=True)
            
            return loss.get("loss")
    
    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch 
        
        self.model.eval()

        if self.use_kd:
            loss_dict = self.model(x, x_len, y, y_len)
            self.log("val/loss", loss_dict["total_loss"])
            self.log("val/kd_loss", loss_dict["kd_loss"])
            self.log("val/asr_loss", loss_dict["student_loss"])
        else:
            loss_output = self.model(x, x_len, y, y_len) 
            log_items = {
                "val/loss": loss_output.get("loss"),
                "val/ctc_loss": loss_output.get("loss_ctc"),
                "val/att_loss": loss_output.get("loss_att"),
            }

            for key, value in log_items.items():
                if value is not None:
                    self.log(key, value, prog_bar=True, sync_dist=True)
            
            with torch.no_grad():
                recog_args = self.model_config.decoder 
                decoded_raw_output = self.model.recognize(x, x_len, y, y_len, recog_args=recog_args)
            
            predicted_transcriptions: List[str] = []

            # decoded_raw_output의 타입에 따라 처리
            if self.model.decoder_type == 'ctc':
                # CTC decoder (greedy/beamsearch)는 List[str]을 반환함
                if isinstance(decoded_raw_output, list) and all(isinstance(d, str) for d in decoded_raw_output):
                    predicted_transcriptions = decoded_raw_output
                else:
                    logging.warning(f"CTC decoded output is not List[str]. Type: {type(decoded_raw_output)}")
                    predicted_transcriptions = [""] * len(x)

            elif self.model.decoder_type == 'rnnt':
                # RNN-T decoder는 딕셔너리 형태를 반환함 (예: {'greedy_search': [['word1', 'word2'], ['word3']]})
                if isinstance(decoded_raw_output, dict) and decoded_raw_output:
                    # 딕셔너리의 첫 번째 값을 가져와서 List[List[str]] 형태인지 확인
                    first_key_value = next(iter(decoded_raw_output.values()))
                    if isinstance(first_key_value, list) and all(isinstance(s, list) and all(isinstance(w, str) for w in s) for s in first_key_value):
                        # 각 내부 리스트(단어 리스트)를 공백으로 조인하여 하나의 문자열로 만듭니다.
                        predicted_transcriptions = [" ".join(word_list) for word_list in first_key_value]
                    else:
                        logging.warning(f"RNN-T decoded output dict value is not List[List[str]]. Actual type: {type(first_key_value)}")
                        predicted_transcriptions = [""] * len(x)
                else:
                    logging.warning("RNN-T decoded output is an empty dictionary or not a dict.")
                    predicted_transcriptions = [""] * len(x)

            elif self.model.decoder_type == 'transformer':
                # Transformer decoder는 List[str]을 반환함 (greedy/beamsearch)
                if isinstance(decoded_raw_output, list) and all(isinstance(d, str) for d in decoded_raw_output):
                    predicted_transcriptions = decoded_raw_output
                else:
                    logging.warning(f"Transformer decoded output is not List[str]. Type: {type(decoded_raw_output)}")
                    predicted_transcriptions = [""] * len(x)
            
            else:
                logging.warning(f"Unsupported decoder type: {self.model.decoder_type}. Decoded output type: {type(decoded_raw_output)}")
                predicted_transcriptions = [""] * len(x)
            
            
            # # Ground Truth 텍스트 변환
            reference_transcriptions: List[str] = []
            for i in range(len(y)):
                gt_tokens = y[i].tolist()
                gt_text = self.token_processor.id2text(gt_tokens, filter_blank=True)
                reference_transcriptions.append(gt_text)
            
            # # 텍스트 전처리 (WER 계산을 위해 공통적으로 수행)
            try:
                processed_predicted_transcriptions = [preprocess_text(text) for text in predicted_transcriptions]
                processed_reference_transcriptions = [preprocess_text(text) for text in reference_transcriptions]
            except NameError:
                logging.warning("preprocess_text function not found. Using raw transcriptions for WER.")
                processed_predicted_transcriptions = predicted_transcriptions
                processed_reference_transcriptions = reference_transcriptions

            batch_wers = []
            try:
                import jiwer # jiwer 임포트 확인
                for gt, pred in zip(processed_reference_transcriptions, processed_predicted_transcriptions):
                    if gt or pred: 
                        wer = jiwer.wer(gt, pred)
                        batch_wers.append(wer)
                    else:
                        logging.debug(f"Skipping WER for empty GT/PR. GT='{gt}', PR='{pred}'")
            except ImportError:
                logging.error("jiwer library not installed. Cannot calculate WER. Please install it with 'pip install jiwer'.")
                batch_wers = []

            if batch_wers:
                avg_wer = sum(batch_wers) / len(batch_wers)
                self.log("val_wer", avg_wer, prog_bar=True, sync_dist=True)
            else:
                self.log("val_wer", 1.0, prog_bar=True, sync_dist=True) 

            if batch_idx == 0 and len(processed_reference_transcriptions) > 0:
                logging.info(f"\n===== Batch {batch_idx}, Sample 0 (Decoder Type: {self.model.decoder_type}) =====")
                logging.info(f"GT: '{processed_reference_transcriptions[0]}'")
                logging.info(f"PR: '{processed_predicted_transcriptions[0]}'")
                if batch_wers:
                    logging.info(f"WER: {batch_wers[0]:.4f} ({batch_wers[0] * 100:.2f}%)")
                else:
                    logging.info("WER calculation skipped (no valid texts or jiwer not installed).")

                    
                
                
    def on_validation_epoch_start(self):
        """에폭 시작 시 WER 통계 초기화"""
        self.val_wer_samples = []
        self.val_wer_sum = 0
        self.val_wer_count = 0

    def on_validation_epoch_end(self):
        """에폭 종료 시 WER 통계 처리 및 로깅"""
        if self.val_wer_count > 0:
            avg_wer = self.val_wer_sum / self.val_wer_count
            self.log("val_wer", avg_wer)
            
            # 히스토그램 로깅
            if self.logger and hasattr(self.logger, "experiment"):
                import numpy as np
                import wandb
                if self.val_wer_samples:
                    self.logger.experiment.log({
                        "val_wer_histogram": wandb.Histogram(np.array(self.val_wer_samples)),
                        "global_step": self.global_step
                    })
            
            print(f"\n===== 검증 완료 =====")
            print(f"검증 샘플 수: {self.val_wer_count}")
            print(f"평균 WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")    
                
                
    def test_step(self, batch, batch_idx):
        x, x_len, y, y_len = batch
        if self.use_kd:
            logits = self.student_model.encode(x, x_len)
            decoded_trans = self.student_model.recognize(logits)
        else:
            decoded_trans = self.model.recognize(x, x_len, y, y_len, self.model_config.decoder)

        ref_trans: List[str] = []
        for single_y_tokens in y:
            # ignore_id (padding)를 제외하고 실제 토큰만 사용
            # self.model.sos, self.model.eos 도 고려하여 제거해야 할 수 있습니다.
            # 이 부분은 토크나이저 및 데이터셋 구성에 따라 달라질 수 있습니다.
            filtered_tokens = [
                token.item() for token in single_y_tokens 
                if token.item() != self.model.ignore_id and 
                   token.item() != self.model.sos and 
                   token.item() != self.model.eos
            ]
            ref_trans.append(self.token_processor.id2text(filtered_tokens))
        
        cer_batch = self.error_calculator.calculate_cer(decoded_trans, ref_trans)
        wer_batch = self.error_calculator.calculate_wer(decoded_trans, ref_trans)
        
        # 평균 CER/WER 로깅 (배치 단위)
        self.log("test_cer", cer_batch, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_wer", wer_batch, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 필요하다면 추가적인 로깅 또는 결과 반환
        return {
            "decoded_transcriptions": decoded_trans,
            "reference_transcriptions": ref_trans,
            "cer_batch": cer_batch,
            "wer_batch": wer_batch
        }
        
            
    def configure_optimizers(self):
        # optimizer
        # Conformer 논문은 Adam을 사용했으므로, Adam에 베타와 엡실론 값을 명시적으로 넣어주는 것이 좋습니다.
        # LambdaLR 스케줄러 사용 시, 옵티마이저의 초기 lr은 1.0으로 설정하는 것이 일반적입니다.
        initial_lr_for_scheduler = 1.0 

        if self.optim_config.type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=initial_lr_for_scheduler, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        elif self.optim_config.type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=initial_lr_for_scheduler, betas=(0.9, 0.98), eps=1e-9)
            
        # scheduler
        def transformer_lr_schedule(step, d_model, warmup_steps):
            """
            Transformer Learning Rate Schedule.
            논문: peak learning rate 0.05 / sqrt(d)
            """
            if step == 0:
                step = 1  # 0으로 나누는 것을 방지
            
            # Conformer 논문에 명시된 최고 학습률 (0.05 / sqrt(d))을 정확히 반영
            # '0.05' 상수를 곱해줘야 합니다.
            # d_model ** -0.5 == 1 / sqrt(d_model)
            peak_lr_scale_factor = 0.05 * (d_model ** -0.5) 
            
            # 최종 학습률 반환
            return peak_lr_scale_factor * min(step ** -0.5, step * (warmup_steps ** -1.5))

        d_model = self.model_config.encoder.encoder_dim
        warmup_steps = self.optim_config.warmup_steps
        
        def lr_lambda(step):
            # LambdaLR은 optimizer의 lr에 이 함수의 반환값을 곱하므로,
            # transformer_lr_schedule 자체가 절대 학습률을 반환하도록 했습니다.
            # 따라서 optimizer의 lr을 1.0으로 설정해야 이 스케줄이 제대로 작동합니다.
            return transformer_lr_schedule(step, d_model, warmup_steps)

        if self.optim_config.scheduling_type == "cosine-annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer_config.num_epochs)
        elif self.optim_config.scheduling_type == "warmup":
            # 이 "warmup" 타입은 StepLR을 사용하는데, 이는 트랜스포머의 웜업 스케줄과 다릅니다.
            # Conformer 논문의 스케줄을 사용하려면 'lambda' 타입을 선택해야 합니다.
            logging.warning("Using 'warmup' scheduling_type with StepLR. This is NOT the Transformer LR schedule described in the Conformer paper. Consider using 'lambda' type.")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)
        elif self.optim_config.scheduling_type == "lambda":
            # Conformer 논문 스케줄 적용
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2**10,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" # 'step' 단위로 스케줄러를 업데이트하도록 명시
            }
        } 
    