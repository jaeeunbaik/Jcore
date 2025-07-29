"""
🖤🐰 JaeEun Baik, 2025
"""
import nlptutti as metrics
import logging
import jiwer
import torch
import wandb
import csv
import numpy as np
import pytorch_lightning as pl
from typing import List

import sentencepiece as spm

from util.utils_text import TokenProcessor, ErrorCalculator, preprocess_text
from modules.e2e_asr_model import e2eASR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelModule(pl.LightningModule):
    def on_train_epoch_start(self):
        if self.current_epoch >= 2:
            datamodule = getattr(self.trainer, 'datamodule', None)
            if datamodule is not None and hasattr(datamodule, 'train_dataset'):
                train_dataset = getattr(datamodule, 'train_dataset')
                aug = getattr(train_dataset, 'augmentation', None)
                if aug is not None:
                    aug.noise_mixing = True
                    logging.info(f"[Augmentation] Noise mixing enabled at epoch {self.current_epoch}")


    def __init__(self, config):
        super().__init__()
        self.model_config = config.model.asr
        self.kd_config = config.model.distillation
        self.use_kd = self.kd_config.using_distillation
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
        self.val_cer_sum = 0
        self.val_cer_count = 0
        self.csv_file = 'test_result.csv'
        self.csv_writer = None
        self.csv_file = None

    
    def training_step(self, batch, batch_idx):
        x, x_len, y, y_len, _ = batch
        
        if self.use_kd:
            loss_dict = self.model(x, x_len, y, y_len)
            self.log("train/total_loss", loss_dict["total_loss"])
            return loss_dict["total_loss"]
        else:
            loss = self.model(x, x_len, y, y_len)
            log_items = {
                "train/total_loss": loss.get("loss")
            }            
            for key, value in log_items.items():
                if value is not None:
                    self.log(key, value, prog_bar=True, sync_dist=True)
            
            return loss.get("loss")
    
    def validation_step(self, batch, batch_idx):
        x, x_len, y, y_len, wav_path = batch 
        
        self.model.eval()

        # 메모리 최적화를 위해 gradient 계산 비활성화
        with torch.no_grad():
            if self.use_kd:
                loss_dict = self.model(x, x_len, y, y_len)
                self.log("val/loss", loss_dict["total_loss"])
            else:
                loss_output = self.model(x, x_len, y, y_len) 
                val_loss = loss_output.get("loss")
                # ↓ key를 "val_loss"로, 그리고 epoch 집계·로거 전송 플래그 추가
                self.log(
                    "val_loss",         # ModelCheckpoint(monitor="val_loss")와 동일
                    val_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
            # 인코딩과 디코딩을 분리하여 메모리 사용량 최적화
            recog_args = self.model_config.decoder 
            # y는 None으로, ylens는 그대로 전달하여 오류 수정
            decoded_raw_output = self.model.recognize(x, x_len, None, y_len, recog_args=recog_args)
            
            predicted_transcriptions: List[str] = []

            if self.model.decoder_type == 'ctc':
                if isinstance(decoded_raw_output, list) and all(isinstance(d, str) for d in decoded_raw_output):
                    predicted_transcriptions = decoded_raw_output
                else:
                    logging.warning(f"CTC decoded output is not List[str]. Type: {type(decoded_raw_output)}")
                    predicted_transcriptions = [""] * len(x)

            elif self.model.decoder_type == 'rnnt':
                if isinstance(decoded_raw_output, dict) and decoded_raw_output:
                    first_key_value = next(iter(decoded_raw_output.values()))
                    if isinstance(first_key_value, list) and all(isinstance(s, list) and all(isinstance(w, str) for w in s) for s in first_key_value):

                        predicted_transcriptions = [" ".join(word_list) for word_list in first_key_value]
                    else:
                        logging.warning(f"RNN-T decoded output dict value is not List[List[str]]. Actual type: {type(first_key_value)}")
                        predicted_transcriptions = [""] * len(x)
                else:
                    logging.warning("RNN-T decoded output is an empty dictionary or not a dict.")
                    predicted_transcriptions = [""] * len(x)

            elif self.model.decoder_type == 'transformer':
                if isinstance(decoded_raw_output, list) and all(isinstance(d, str) for d in decoded_raw_output):
                    predicted_transcriptions = decoded_raw_output
                else:
                    logging.warning(f"Transformer decoded output is not List[str]. Type: {type(decoded_raw_output)}")
                    predicted_transcriptions = [""] * len(x)
            
            else:
                logging.warning(f"Unsupported decoder type: {self.model.decoder_type}. Decoded output type: {type(decoded_raw_output)}")
                predicted_transcriptions = [""] * len(x)
            
            
            reference_transcriptions: List[str] = []
            for i in range(len(y)):
                gt_tokens = y[i].tolist()
                gt_text = self.token_processor.id2text(gt_tokens, filter_blank=True)
                reference_transcriptions.append(gt_text)
            
            try:
                processed_predicted_transcriptions = [preprocess_text(text) for text in predicted_transcriptions]
                processed_reference_transcriptions = [preprocess_text(text) for text in reference_transcriptions]
            except NameError:
                logging.warning("preprocess_text function not found. Using raw transcriptions for WER.")
                processed_predicted_transcriptions = predicted_transcriptions
                processed_reference_transcriptions = reference_transcriptions
                
                
            batch_wers = []
            batch_cers = []
            
            try:
                import jiwer 
                for gt, pred in zip(processed_reference_transcriptions, processed_predicted_transcriptions):
                    if gt or pred: 
                        wer = jiwer.wer(gt, pred)
                        batch_wers.append(wer)
                        
                        # cer = jiwer.cer(gt, pred) 
                        cer = metrics.get_cer(gt, pred)
                        batch_cers.append(cer['cer']) 
                    else:
                        logging.debug(f"Skipping WER/CER for empty GT/PR. GT='{gt}', PR='{pred}'") # 로그 메시지 수정
            except ImportError:
                logging.error("jiwer library not installed. Cannot calculate WER/CER. Please install it with 'pip install jiwer'.") # 로그 메시지 수정
                batch_wers = []
                batch_cers = [] 

            if self.model_config.report_wer: 
                if batch_wers:
                    # accumulate WER without using samples list
                    self.val_wer_sum += sum(batch_wers)
                    self.val_wer_count += len(batch_wers)

            if self.model_config.report_cer:
                if batch_cers:
                    # accumulate CER without using samples list
                    self.val_cer_sum += sum(batch_cers)
                    self.val_cer_count += len(batch_cers) 
                else:
                    pass
                    # self.log("val_cer", 1.0, prog_bar=True, sync_dist=True) 

            if batch_idx == 0 and len(processed_reference_transcriptions) > 0:
                logging.info(f"\n===== Batch {batch_idx}, Sample 0 (Decoder Type: {self.model.decoder_type}) =====")
                logging.info(f"GT: '{processed_reference_transcriptions[0]}'")
                logging.info(f"PR: '{processed_predicted_transcriptions[0]}'")
                if batch_wers: 
                    logging.info(f"WER: {batch_wers[0]:.4f} ({batch_wers[0] * 100:.2f}%)")
                    if batch_cers and self.model_config.report_cer:
                        logging.info(f"CER: {batch_cers[0]:.4f} ({batch_cers[0] * 100:.2f}%)") 
                else:
                    logging.info("WER/CER calculation skipped (no valid texts or jiwer not installed).") 
                
                
    def on_validation_epoch_start(self):
        self.val_wer_sum = 0
        self.val_wer_count = 0
        self.val_cer_sum = 0
        self.val_cer_count = 0
                
                    
    def on_validation_epoch_end(self):
        avg_cer, avg_wer = None, None

        # CER epoch metric 로깅
        if self.model_config.report_cer and self.val_cer_count > 0:
            avg_cer = self.val_cer_sum / self.val_cer_count
            self.log(
                "val_cer_epoch",
                avg_cer,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        # WER epoch metric 로깅
        if self.model_config.report_wer and self.val_wer_count > 0:
            avg_wer = self.val_wer_sum / self.val_wer_count
            self.log(
                "val_wer_epoch",
                avg_wer,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        # 2) print에서 logs 대신 avg_cer, avg_wer 사용
        if self.trainer.is_global_zero:
            print(f"\n===== Epoch {self.current_epoch} Validation Done "
                  f"(CER={avg_cer}, WER={avg_wer}) =====")
            
    def on_test_epoch_start(self):
        self.test_cer_sum = 0
        self.test_cer_count = 0
        if self.trainer.is_global_zero:
            # CSV 파일 초기화 로직을 test_step에서 여기로 이동
            self.csv_file = open("test_result.csv", "w", encoding="utf-8", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["WAV_Path", "GT", "PR", "CER"])

    def test_step(self, batch, batch_idx):
        x, x_len, y, ylens, wav_path = batch  # y, y_len은 사용하지 않으므로 _로 받음
        self.model.eval()
        with torch.no_grad():
            # 테스트 시점에서는 Loss 계산이 불필요하므로 관련 코드 제거
            # 오직 recognize만 수행
            recog_args = self.model_config.decoder
            decoded_raw_output = self.model.recognize(x, x_len, None, ylens, recog_args=recog_args)

            predicted_transcriptions: List[str] = []
            if self.model.decoder_type == 'rnnt':
                if isinstance(decoded_raw_output, dict) and decoded_raw_output:
                    first_key_value = next(iter(decoded_raw_output.values()))
                    if isinstance(first_key_value, list) and all(isinstance(s, list) and all(isinstance(w, str) for w in s) for s in first_key_value):
                        predicted_transcriptions = [" ".join(word_list) for word_list in first_key_value]
                    else:
                        predicted_transcriptions = [""] * len(x)
                else:
                    predicted_transcriptions = [""] * len(x)
            else:
                # 다른 디코더 타입에 대한 처리
                pass

            # GT를 가져오는 부분은 에러 계산을 위해 그대로 둡니다.
            # 하지만 dataloader가 y를 반환하지 않는 경우를 대비해야 합니다.
            # 이 예제에서는 batch에 y가 포함되어 있다고 가정합니다.
            y, y_len = batch[2], batch[3]
            reference_transcriptions: List[str] = []
            for i in range(len(y)):
                gt_tokens = y[i].tolist()
                gt_text = self.token_processor.id2text(gt_tokens, filter_blank=True)
                reference_transcriptions.append(gt_text)

            processed_predicted_transcriptions = [preprocess_text(text) for text in predicted_transcriptions]
            processed_reference_transcriptions = [preprocess_text(text) for text in reference_transcriptions]

            for i in range(len(processed_reference_transcriptions)):
                gt = processed_reference_transcriptions[i]
                pr = processed_predicted_transcriptions[i]
                path = wav_path[i]
                cer = metrics.get_cer(gt, pr)["cer"] if (gt or pr) else 1.0
                
                # CER 누적
                self.test_cer_sum += cer
                self.test_cer_count += 1

                # 로그 및 CSV 저장
                logging.info(f"[Test] {path}\n  GT: {gt}\n  PR: {pr}\n  CER: {cer:.4f}")
                if self.trainer.is_global_zero:
                    self.csv_writer.writerow([path, gt, pr, f"{cer:.4f}"])

    def on_test_epoch_end(self):
        # 전체 평균 CER 계산
        avg_cer = self.test_cer_sum / self.test_cer_count if self.test_cer_count > 0 else 0.0
        
        # PyTorch Lightning 로거에 최종 메트릭 기록
        self.log("test_avg_cer", avg_cer, prog_bar=True, logger=True, sync_dist=True)

        # CSV 파일 마무리
        if self.trainer.is_global_zero and self.csv_file is not None:
            self.csv_writer.writerow(["", "", "평균 CER:", f"{avg_cer:.4f}"])
            self.csv_file.close()
            logging.info(f"Test results saved to test_result.csv with average CER: {avg_cer:.4f}")
        
        
        
    def configure_optimizers(self):
        initial_lr_for_optimizer = 1.0 

        if self.optim_config.type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=initial_lr_for_optimizer, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        elif self.optim_config.type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=initial_lr_for_optimizer, betas=(0.9, 0.98), eps=1e-9)
            
        def transformer_lr_schedule_factor(step, d_model, warmup_steps, initial_k_value):
            """
            Conformer 논문의 Learning Rate Schedule에 따라 '스케일 팩터'를 반환합니다.
            최종 LR = optimizer_initial_lr * transformer_lr_schedule_factor(...)
            Args:
                initial_k_value (float): 논문에서의 'k' 또는 'peak_lr'에 해당하는 값. (YAML의 op_lr)
            """
            if step == 0:
                step = 1  
            
            return initial_k_value * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

        d_model = self.model_config.encoder.encoder_dim
        warmup_steps = self.optim_config.warmup_steps
        k_value_from_config = self.optim_config.op_lr 
        
        def lr_lambda(step):
            return transformer_lr_schedule_factor(step, d_model, warmup_steps, k_value_from_config)

        if self.optim_config.scheduling_type == "lambda":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
