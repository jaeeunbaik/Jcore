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

    def on_train_epoch_end(self):
        # GPU 캐시 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Epoch {self.current_epoch}: Cleared CUDA cache.")
    
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

            if isinstance(decoded_raw_output, list):
                if all(isinstance(d, str) for d in decoded_raw_output):
                    predicted_transcriptions = decoded_raw_output
                else:
                    logging.warning(f"Decoded output is a list but not List[str]. Type: {type(decoded_raw_output[0]) if decoded_raw_output else 'empty'}")
                    predicted_transcriptions = [""] * len(x)

            elif isinstance(decoded_raw_output, dict):
                if decoded_raw_output:
                    # 딕셔너리 값은 List[List[str]] 형태입니다.
                    first_key_value = next(iter(decoded_raw_output.values())) 
                    
                    if isinstance(first_key_value, list) and all(isinstance(s, list) and all(isinstance(w, str) for w in s) for s in first_key_value):
                        # 각 내부 리스트(단어 리스트)를 공백으로 조인하여 하나의 문자열로 만듭니다.
                        predicted_transcriptions = [" ".join(word_list) for word_list in first_key_value]
                    elif isinstance(first_key_value, list) and all(isinstance(s, str) for s in first_key_value):
                        # 이미 List[str] 형태인 경우 (이 경우는 발생하지 않을 가능성이 높음)
                        predicted_transcriptions = first_key_value
                    else:
                        logging.warning(f"Decoded output dict value is not List[List[str]] or List[str]. Actual type: {type(first_key_value)}")
                        predicted_transcriptions = [""] * len(x)
                else:
                    logging.warning("Decoded output is an empty dictionary.")
                    predicted_transcriptions = [""] * len(x)
            else:
                logging.warning(f"Unsupported decoded output type: {type(decoded_raw_output)}")
                predicted_transcriptions = [""] * len(x) 

            # Ground Truth 텍스트 변환
            reference_transcriptions: List[str] = []
            for i in range(len(y)):
                gt_tokens = y[i].tolist()
                gt_text = self.token_processor.id2text(gt_tokens, filter_blank=True)
                reference_transcriptions.append(gt_text)
            
            # 텍스트 전처리 (WER 계산을 위해 공통적으로 수행)
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
            decoded_trans = self.model.recognize(x, x_len, y, self.model_config.decoder)

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
        if self.optim_config.type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        elif self.optim_config.type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            
        # scheduler
        def transformer_lr_schedule(step, d_model, warmup_steps):
            """
            Transformer Learning Rate Schedule.

            Args:
                step (int): 현재 스텝.
                d_model (int): 모델의 차원 (hidden size).
                warmup_steps (int): 워밍업 스텝 수.

            Returns:
                float: 학습률 스케일링 값.
            """
            if step == 0:
                step = 1  # 0으로 나누는 것을 방지
            scale = d_model ** -0.5
            return scale * min(step ** -0.5, step * (warmup_steps ** -1.5))
        d_model = self.model_config.encoder.encoder_dim
        warmup_steps = self.optim_config.warmup_steps
        def lr_lambda(step):
            return transformer_lr_schedule(step, d_model, warmup_steps)

        if self.optim_config.scheduling_type == "cosine-annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer_config.num_epochs)  # T_max : cosine 주기 한 번 도는데 걸리는
        elif self.optim_config.scheduling_type == "warmup":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)
        elif self.optim_config.scheduling_type == "lambda":
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
                "interval": "step"
            }
        }
            
    
    def log_text(self, key: str, value: str, step: int):
        """
        WandB에 텍스트를 로깅하기 위한 헬퍼 함수.
        PyTorch Lightning의 logger.experiment를 통해 WandB API에 접근합니다.
        """
        if self.logger and hasattr(self.logger, "experiment") and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            # self.logger.experiment는 WandB Run 객체입니다.
            # wandb.log()를 사용하여 텍스트를 로깅합니다.
            self.logger.experiment.log({key: value}, step=step)
        else:
            # WandB 로거가 활성화되지 않았거나 다른 로거를 사용하는 경우 콘솔에 출력
            print(f"Log (Step {step}) - {key}: {value}")