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
        x, x_len, y, y_len, _ = batch
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
        x, x_len, y, y_len, wav_path = batch 
        
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
                    avg_wer = sum(batch_wers) / len(batch_wers)
                    # self.log("val_wer", avg_wer, prog_bar=True, sync_dist=True)
                    if self.trainer.is_global_zero: 
                        self.val_wer_samples.extend(batch_wers)
                        self.val_wer_sum += sum(batch_wers) 
                        self.val_wer_count += len(batch_wers)
                else:
                    pass
                    # self.log("val_wer", 1.0, prog_bar=True, sync_dist=True)
            
            if self.model_config.report_cer:
                if batch_cers:
                    avg_cer = sum(batch_cers) / len(batch_cers)
                    self.log("val_cer", avg_cer, prog_bar=True, sync_dist=True)
                    if self.trainer.is_global_zero:
                        self.val_cer_samples.extend(batch_cers)
                        self.val_cer_sum += sum(batch_cers)
                        self.val_cer_count += len(batch_cers) 
                else:
                    self.log("val_cer", 1.0, prog_bar=True, sync_dist=True) 

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
        self.val_wer_samples = []
        self.val_wer_sum = 0
        self.val_wer_count = 0
        self.val_cer_samples = []
        self.val_cer_sum = 0
        self.val_cer_count = 0
        

    def on_validation_epoch_end(self):
        if self.val_wer_count > 0:
            avg_wer_epoch = self.val_wer_sum / self.val_wer_count
            # self.log("val_wer", avg_wer_epoch, sync_dist=True)
            
            if self.logger and hasattr(self.logger, "experiment"):
                if self.val_wer_samples:
                    self.logger.experiment.log({
                        "val_wer_histogram": wandb.Histogram(np.array(self.val_wer_samples)),
                        "global_step": self.global_step
                    })
        else:
            self.log("val_wer_epoch", 1.0, sync_dist=True)
            
        if self.val_cer_count > 0:
            avg_cer_epoch = self.val_cer_sum / self.val_cer_count 
            self.log("val_cer_epoch", avg_cer_epoch, sync_dist=True)
            
            if self.logger and hasattr(self.logger, "experiment"):
                if self.val_cer_samples:
                    self.logger.experiment.log({
                        "val_cer_histogram": wandb.Histogram(np.array(self.val_cer_samples)), 
                        "global_step": self.global_step
                    })
        else:
            self.log("val_cer_epoch", 1.0, sync_dist=True)
            
        if self.trainer.is_global_zero: 
            print(f"\n===== 검증 완료 =====")
            if self.val_wer_count > 0:
                print(f"검증 샘플 수: {self.val_wer_count}")
                print(f"평균 WER (epoch-end): {avg_wer_epoch:.4f} ({avg_wer_epoch*100:.2f}%)")
            if self.val_cer_count > 0 and self.model_config.report_cer: 
                print(f"평균 CER (epoch-end): {avg_cer_epoch:.4f} ({avg_cer_epoch*100:.2f}%)")
                
                
    def test_step(self, batch, batch_idx):
        x, x_len, y, y_len, wav_path = batch
        if self.use_kd:
            logits = self.student_model.encode(x, x_len)
            decoded_trans = self.student_model.recognize(logits)
        else:
            decoded_trans = self.model.recognize(x, x_len, y, y_len, self.model_config.decoder)

        ref_trans: List[str] = []
        for single_y_tokens in y:
            filtered_tokens = [
                token.item() for token in single_y_tokens 
                if token.item() != self.model.ignore_id and 
                   token.item() != self.model.sos and 
                   token.item() != self.model.eos
            ]
            ref_trans.append(self.token_processor.id2text(filtered_tokens))
        
        predicted_transcriptions_test: List[str] = []
        if isinstance(decoded_trans, dict):
            if decoded_trans:
                key_used = next(iter(decoded_trans.keys()))
                if isinstance(decoded_trans[key_used], list) and \
                   all(isinstance(s, list) and all(isinstance(w, str) for w in s) for s in decoded_trans[key_used]):
                    predicted_transcriptions_test = [" ".join(word_list) for word_list in decoded_trans[key_used]]
                elif isinstance(decoded_trans[key_used], list) and \
                     all(isinstance(s, str) for s in decoded_trans[key_used]): # 이미 List[str]인 경우
                    predicted_transcriptions_test = decoded_trans[key_used]
                else:
                    logging.warning(f"RNN-T decoded output dict value is unexpected type for {key_used}.")
                    predicted_transcriptions_test = [""] * len(x)
            else:
                logging.warning("Decoded output is an empty dictionary.")
                predicted_transcriptions_test = [""] * len(x)
        elif isinstance(decoded_trans, list) and all(isinstance(d, str) for d in decoded_trans):
            predicted_transcriptions_test = decoded_trans
        else:
            logging.warning(f"Unsupported decoded output type: {type(decoded_trans)}. Falling back to empty strings.")
            predicted_transcriptions_test = [""] * len(x)

        try:
            processed_predicted_transcriptions = [preprocess_text(text) for text in predicted_transcriptions_test]
            processed_reference_transcriptions = [preprocess_text(text) for text in ref_trans]
        except NameError:
            logging.warning("preprocess_text function not found. Using raw transcriptions for WER/CER in test_step.")
            processed_predicted_transcriptions = predicted_transcriptions_test
            processed_reference_transcriptions = ref_trans
            
        if self.trainer.is_global_zero: 
            logging.info(f"\n===== Test Step Batch {batch_idx} Results =====")
            for i in range(len(processed_reference_transcriptions)):
                current_gt = processed_reference_transcriptions[i]
                current_pr = processed_predicted_transcriptions[i]
                logging.info(f"  Sample {i+1} GT: '{current_gt}'")
                logging.info(f"  Sample {i+1} PR: '{current_pr}'")
                if self.model_config.report_wer or self.model_config.report_cer:
                    if current_gt or current_pr:
                        sample_wer = jiwer.wer(current_gt, current_pr)
                        # sample_cer = jiwer.cer(current_gt, current_pr)
                        sample_cer = metrics.get_cer(current_gt, current_pr)['cer']
                        logging.info(f"  Sample {i+1} WER: {sample_wer:.4f} ({sample_wer*100:.2f}%)")
                        logging.info(f"  Sample {i+1} CER: {sample_cer:.4f} ({sample_cer*100:.2f}%)")
                    else:
                        logging.info(f"  Sample {i+1} WER/CER skipped (empty GT/PR).")
            logging.info(f"========================================\n")
        batch_wers = []
        batch_cers = [] 
        
        try:
            for gt, pred in zip(processed_reference_transcriptions, processed_predicted_transcriptions):
                if gt or pred: 
                    wer = jiwer.wer(gt, pred)
                    batch_wers.append(wer)
                    
                    # cer = jiwer.cer(gt, pred) 
                    cer = metrics.get_cer(gt, pred)
                    batch_cers.append(cer['cer']) 
                else:
                    logging.debug(f"Skipping WER/CER for empty GT/PR. GT='{gt}', PR='{pred}' in test_step.") 
        except ImportError:
            logging.error("jiwer library not installed. Cannot calculate WER/CER in test_step. Please install it with 'pip install jiwer'.") 
            batch_wers = []
            batch_cers = [] 

        if self.model_config.report_wer: 
            if batch_wers:
                avg_wer_batch = sum(batch_wers) / len(batch_wers) 
                self.log("test_wer_batch", avg_wer_batch, on_step=True, on_epoch=False, prog_bar=True, logger=True) 
                self.log("test_wer", avg_wer_batch, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
            else:
                self.log("test_wer_batch", 1.0, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                self.log("test_wer", 1.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.model_config.report_cer: 
            if batch_cers:
                avg_cer_batch = sum(batch_cers) / len(batch_cers) 
                self.log("test_cer_batch", avg_cer_batch, on_step=True, on_epoch=False, prog_bar=True, logger=True) 
                self.log("test_cer", avg_cer_batch, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
            else:
                self.log("test_cer_batch", 1.0, on_step=True, on_epoch=False, prog_bar=True, logger=True)
                self.log("test_cer", 1.0, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # CSV 파일에 결과를 저장하기 위한 준비
        if batch_idx == 0 and self.trainer.is_global_zero:
            self.csv_file = open("test_results.csv", "w", encoding="utf-8", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["WAV_Path", "GT", "PR", "CER"])  # 헤더 작성
            self.total_cer = 0.0
            self.num_samples = 0

        # 각 샘플에 대한 결과 저장
        if self.trainer.is_global_zero:
            for i in range(len(processed_reference_transcriptions)):
                current_gt = processed_reference_transcriptions[i]
                current_pr = processed_predicted_transcriptions[i]
                path = wav_path[i]
                
                if current_gt or current_pr:
                    sample_cer = metrics.get_cer(current_gt, current_pr)['cer']
                    self.csv_writer.writerow([path, current_gt, current_pr, f"{sample_cer:.4f}"])
                    self.total_cer += sample_cer  # CER 합계 업데이트
                    self.num_samples += 1  # 샘플 수 업데이트
                else:
                    self.csv_writer.writerow([path, current_gt, current_pr, "N/A"])

        return {
            "decoded_transcriptions": predicted_transcriptions_test,
            "reference_transcriptions": ref_trans,
            "cer_batch": avg_cer_batch if self.model_config.report_cer and batch_cers else None, 
            "wer_batch": avg_wer_batch if self.model_config.report_wer and batch_wers else None, 
        }


    def on_test_epoch_end(self):
        if hasattr(self, "csv_file") and self.csv_file:
            if hasattr(self, "total_cer") and hasattr(self, "num_samples"):
                avg_cer = self.total_cer / self.num_samples if self.num_samples > 0 else 0.0
                self.csv_writer.writerow(["", "", "평균 CER:", f"{avg_cer:.4f}"])  # 마지막 줄에 평균 CER 추가
            else:
                self.csv_writer.writerow(["", "", "평균 CER:", "N/A"])
            self.csv_file.close()
            logging.info("Test results saved to test_results.csv")
        
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
    